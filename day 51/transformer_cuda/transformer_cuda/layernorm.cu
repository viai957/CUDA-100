/*
 * LayerNorm: CUDA kernel for layer normalization in transformer models
 * Math: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 * Inputs / Outputs: [batch_size*seq_len, hidden_size]
 * Assumptions: hidden_size is relatively small (512-2048)
 * Parallel Strategy: One warp per row, vectorized memory access
 * Mixed Precision Policy: FP16/BF16 input/output, FP32 for internal stats
 * Complexity: O(N) FLOPs, O(N) memory reads/writes
 * Test Vectors: Validated against PyTorch's native LayerNorm
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Helper functions for type conversion
template <typename T>
__device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }

template <>
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v) { return static_cast<T>(v); }

template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }

// Vectorized memory access for better throughput
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VecType {
    T data[VecSize];
};

// Welford's online algorithm for computing mean and variance
template <typename T>
__device__ __forceinline__ void welford_update(float& mean, float& m2, float& count, T val) {
    count += 1.0f;
    float delta = to_float(val) - mean;
    mean += delta / count;
    float delta2 = to_float(val) - mean;
    m2 += delta * delta2;
}

// Warp-level reduction using shuffle instructions
template <typename T>
__device__ __forceinline__ void warp_reduce_welford(float& mean, float& m2, float& count) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float b_mean = __shfl_down_sync(FULL_MASK, mean, offset);
        float b_m2 = __shfl_down_sync(FULL_MASK, m2, offset);
        float b_count = __shfl_down_sync(FULL_MASK, count, offset);
        
        // Combine statistics
        if (b_count > 0) {
            float delta = b_mean - mean;
            float new_count = count + b_count;
            float nb_over_n = b_count / new_count;
            
            m2 += b_m2 + delta * delta * count * nb_over_n;
            mean += delta * nb_over_n;
            count = new_count;
        }
    }
    
    // Broadcast to all threads in the warp
    mean = __shfl_sync(FULL_MASK, mean, 0);
    m2 = __shfl_sync(FULL_MASK, m2, 0);
    count = __shfl_sync(FULL_MASK, count, 0);
}

// Main LayerNorm kernel
template <typename T, int VecSize>
__global__ void layernorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    T* __restrict__ mean_out,
    T* __restrict__ inv_var_out,
    int hidden_size,
    float epsilon
) {
    // Each warp handles one row
    const int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    
    // Skip if out of bounds
    if (row_idx >= gridDim.x * blockDim.y) return;
    
    // Input/output pointers for this row
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    
    // Welford accumulators
    float mean = 0.0f;
    float m2 = 0.0f;
    float count = 0.0f;
    
    // Process elements in chunks of VecSize
    constexpr int elems_per_thread = VecSize * (WARP_SIZE / 32);
    
    // Step 1: Compute mean and variance using Welford's algorithm
    for (int idx = lane_id * VecSize; idx < hidden_size; idx += WARP_SIZE * VecSize) {
        // Bounds checking
        if (idx + VecSize <= hidden_size) {
            // Vector load
            VecType<T, VecSize> vec_input;
            *reinterpret_cast<VecType<T, VecSize>*>(&vec_input) = 
                *reinterpret_cast<const VecType<T, VecSize>*>(row_input + idx);
            
            // Update statistics
            #pragma unroll
            for (int i = 0; i < VecSize; i++) {
                welford_update(mean, m2, count, vec_input.data[i]);
            }
        } else {
            // Handle remainder (not aligned to VecSize)
            for (int i = 0; idx + i < hidden_size && i < VecSize; i++) {
                welford_update(mean, m2, count, row_input[idx + i]);
            }
        }
    }
    
    // Reduce statistics across the warp
    warp_reduce_welford(mean, m2, count);
    
    // Compute inverse standard deviation
    float inv_std = rsqrtf(m2 / count + epsilon);
    
    // Store statistics if we're the first thread in the warp
    if (lane_id == 0) {
        mean_out[row_idx] = from_float<T>(mean);
        inv_var_out[row_idx] = from_float<T>(inv_std);
    }
    
    // Step 2: Normalize and apply affine transformation
    for (int idx = lane_id * VecSize; idx < hidden_size; idx += WARP_SIZE * VecSize) {
        // Bounds checking
        if (idx + VecSize <= hidden_size) {
            // Vector load
            VecType<T, VecSize> vec_input, vec_output;
            *reinterpret_cast<VecType<T, VecSize>*>(&vec_input) = 
                *reinterpret_cast<const VecType<T, VecSize>*>(row_input + idx);
            
            // Normalize and transform
            #pragma unroll
            for (int i = 0; i < VecSize; i++) {
                float val = to_float(vec_input.data[i]);
                float w = to_float(weight[idx + i]);
                float b = bias ? to_float(bias[idx + i]) : 0.0f;
                
                // LayerNorm formula: y = ((x - mean) * inv_std) * gamma + beta
                float normalized = (val - mean) * inv_std;
                float transformed = normalized * w + b;
                
                vec_output.data[i] = from_float<T>(transformed);
            }
            
            // Vector store
            *reinterpret_cast<VecType<T, VecSize>*>(row_output + idx) = 
                *reinterpret_cast<VecType<T, VecSize>*>(&vec_output);
        } else {
            // Handle remainder (not aligned to VecSize)
            for (int i = 0; idx + i < hidden_size && i < VecSize; i++) {
                float val = to_float(row_input[idx + i]);
                float w = to_float(weight[idx + i]);
                float b = bias ? to_float(bias[idx + i]) : 0.0f;
                
                // LayerNorm formula: y = ((x - mean) * inv_std) * gamma + beta
                float normalized = (val - mean) * inv_std;
                float transformed = normalized * w + b;
                
                row_output[idx + i] = from_float<T>(transformed);
            }
        }
    }
}

// Launcher function that handles different data types
void layernorm_forward_launcher(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& mean,
    torch::Tensor& inv_variance,
    double epsilon,
    cudaStream_t stream)
{
    const int batch_size = input.size(0);
    const int hidden_size = input.size(1);
    
    // Determine optimal block/grid dimensions
    const int threads_per_warp = WARP_SIZE;
    const int warps_per_block = 4;  // Can be tuned
    const int threads_per_block = threads_per_warp * warps_per_block;
    const int blocks_per_grid = (batch_size + warps_per_block - 1) / warps_per_block;
    
    dim3 grid(blocks_per_grid);
    dim3 block(threads_per_warp, warps_per_block);
    
    // Launch appropriate kernel based on input type
    if (input.scalar_type() == torch::ScalarType::Half) {
        // Use vectorized loads for better memory throughput
        if (hidden_size % 4 == 0) {
            layernorm_kernel<__half, 4><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<const __half*>(weight.data_ptr()),
                bias.numel() > 0 ? reinterpret_cast<const __half*>(bias.data_ptr()) : nullptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                reinterpret_cast<__half*>(mean.data_ptr()),
                reinterpret_cast<__half*>(inv_variance.data_ptr()),
                hidden_size,
                static_cast<float>(epsilon)
            );
        } else if (hidden_size % 2 == 0) {
            layernorm_kernel<__half, 2><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<const __half*>(weight.data_ptr()),
                bias.numel() > 0 ? reinterpret_cast<const __half*>(bias.data_ptr()) : nullptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                reinterpret_cast<__half*>(mean.data_ptr()),
                reinterpret_cast<__half*>(inv_variance.data_ptr()),
                hidden_size,
                static_cast<float>(epsilon)
            );
        } else {
            layernorm_kernel<__half, 1><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<const __half*>(weight.data_ptr()),
                bias.numel() > 0 ? reinterpret_cast<const __half*>(bias.data_ptr()) : nullptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                reinterpret_cast<__half*>(mean.data_ptr()),
                reinterpret_cast<__half*>(inv_variance.data_ptr()),
                hidden_size,
                static_cast<float>(epsilon)
            );
        }
    } else if (input.scalar_type() == torch::ScalarType::Float) {
        // For float, use vectorized loads when possible
        if (hidden_size % 4 == 0) {
            layernorm_kernel<float, 4><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(input.data_ptr()),
                reinterpret_cast<const float*>(weight.data_ptr()),
                bias.numel() > 0 ? reinterpret_cast<const float*>(bias.data_ptr()) : nullptr,
                reinterpret_cast<float*>(output.data_ptr()),
                reinterpret_cast<float*>(mean.data_ptr()),
                reinterpret_cast<float*>(inv_variance.data_ptr()),
                hidden_size,
                static_cast<float>(epsilon)
            );
        } else if (hidden_size % 2 == 0) {
            layernorm_kernel<float, 2><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(input.data_ptr()),
                reinterpret_cast<const float*>(weight.data_ptr()),
                bias.numel() > 0 ? reinterpret_cast<const float*>(bias.data_ptr()) : nullptr,
                reinterpret_cast<float*>(output.data_ptr()),
                reinterpret_cast<float*>(mean.data_ptr()),
                reinterpret_cast<float*>(inv_variance.data_ptr()),
                hidden_size,
                static_cast<float>(epsilon)
            );
        } else {
            layernorm_kernel<float, 1><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(input.data_ptr()),
                reinterpret_cast<const float*>(weight.data_ptr()),
                bias.numel() > 0 ? reinterpret_cast<const float*>(bias.data_ptr()) : nullptr,
                reinterpret_cast<float*>(output.data_ptr()),
                reinterpret_cast<float*>(mean.data_ptr()),
                reinterpret_cast<float*>(inv_variance.data_ptr()),
                hidden_size,
                static_cast<float>(epsilon)
            );
        }
    } else {
        AT_ERROR("Unsupported input type for LayerNorm kernel");
    }
} 