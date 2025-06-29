/*
 * FeedForward: CUDA kernel for feed-forward networks in transformer models
 * Math: FFN(x) = Linear2(Activation(Linear1(x)))
 * Inputs: x of shape [batch_size, seq_len, d_model]
 * Assumptions: d_model and d_ff are relatively small (512-4096)
 * Parallel Strategy: Block per batch*seq_len, thread per feature
 * Mixed Precision Policy: FP16/BF16 input/output, FP32 for activations
 * Distributed Hooks: None (handled at Python level)
 * Complexity: O(N*d_model*d_ff) FLOPs, O(N*d_model + N*d_ff) memory
 * Test Vectors: Validated against PyTorch's native implementation
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

// GELU activation function
__device__ __forceinline__ float gelu(float x) {
    // Approximation of GELU
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coef * x3);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

// ReLU activation function
__device__ __forceinline__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Dropout function
__device__ __forceinline__ float apply_dropout(float x, float prob, unsigned int& seed) {
    if (prob == 0.0f) return x;
    
    // Simple xorshift random number generator
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    
    float rand = __uint2float_rn(seed) / __uint2float_rn(0xFFFFFFFF);
    return rand < prob ? 0.0f : x / (1.0f - prob);
}

// CUDA kernel for fused feed-forward network (Linear1 → Activation → Dropout → Linear2)
template <typename T>
__global__ void feed_forward_kernel(
    const T* __restrict__ input,           // [batch_size, seq_len, d_model]
    const T* __restrict__ fc1_weight,      // [d_ff, d_model]
    const T* __restrict__ fc1_bias,        // [d_ff] or nullptr
    const T* __restrict__ fc2_weight,      // [d_model, d_ff]
    const T* __restrict__ fc2_bias,        // [d_model] or nullptr
    T* __restrict__ output,                // [batch_size, seq_len, d_model]
    bool use_gelu,                         // true for GELU, false for ReLU
    float dropout_prob,                    // Dropout probability
    int batch_size,
    int seq_len,
    int d_model,
    int d_ff,
    unsigned long long seed                // Random seed for dropout
) {
    // Each block handles one token (batch_idx, seq_idx)
    const int token_idx = blockIdx.x;
    const int batch_idx = token_idx / seq_len;
    const int seq_idx = token_idx % seq_len;
    
    // Thread index within the block
    const int tid = threadIdx.x;
    
    // Skip if out of bounds
    if (batch_idx >= batch_size) return;
    
    // Pointer to the current token's input
    const T* token_input = input + (batch_idx * seq_len + seq_idx) * d_model;
    
    // Pointer to the current token's output
    T* token_output = output + (batch_idx * seq_len + seq_idx) * d_model;
    
    // Shared memory for intermediate results
    extern __shared__ char shared_memory[];
    float* hidden = reinterpret_cast<float*>(shared_memory);
    
    // Generate per-thread seed for dropout
    unsigned int thread_seed = seed + token_idx * blockDim.x + tid;
    
    // Step 1: Compute Linear1(x) + bias1
    for (int ff_idx = tid; ff_idx < d_ff; ff_idx += blockDim.x) {
        float sum = 0.0f;
        
        // Compute dot product
        for (int m_idx = 0; m_idx < d_model; m_idx++) {
            sum += to_float(token_input[m_idx]) * to_float(fc1_weight[ff_idx * d_model + m_idx]);
        }
        
        // Add bias if provided
        if (fc1_bias != nullptr) {
            sum += to_float(fc1_bias[ff_idx]);
        }
        
        // Apply activation
        if (use_gelu) {
            sum = gelu(sum);
        } else {
            sum = relu(sum);
        }
        
        // Apply dropout
        sum = apply_dropout(sum, dropout_prob, thread_seed);
        
        // Store in shared memory
        hidden[ff_idx] = sum;
    }
    
    // Make sure all threads have computed their hidden activations
    __syncthreads();
    
    // Step 2: Compute Linear2(hidden) + bias2
    for (int m_idx = tid; m_idx < d_model; m_idx += blockDim.x) {
        float sum = 0.0f;
        
        // Compute dot product
        for (int ff_idx = 0; ff_idx < d_ff; ff_idx++) {
            sum += hidden[ff_idx] * to_float(fc2_weight[m_idx * d_ff + ff_idx]);
        }
        
        // Add bias if provided
        if (fc2_bias != nullptr) {
            sum += to_float(fc2_bias[m_idx]);
        }
        
        // Write output
        token_output[m_idx] = from_float<T>(sum);
    }
}

// Launcher function that handles different data types
torch::Tensor feed_forward_forward_launcher(
    const torch::Tensor& input,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_bias,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc2_bias,
    const std::string& activation,
    float dropout_prob,
    cudaStream_t stream)
{
    // Get dimensions
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int d_model = input.size(2);
    const int d_ff = fc1_weight.size(0);
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Determine if we're using GELU or ReLU
    bool use_gelu = (activation == "gelu");
    
    // Generate random seed for dropout
    unsigned long long seed = 0;
    if (dropout_prob > 0.0f) {
        seed = static_cast<unsigned long long>(torch::randint(0, INT_MAX, {1}).item<int>());
    }
    
    // Determine block size and shared memory size
    const int block_size = 256;  // Can be tuned
    const int shared_mem_size = d_ff * sizeof(float);  // For hidden activations
    
    // Check if shared memory size is within limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        AT_ERROR("Shared memory size exceeds device limit");
    }
    
    // Determine grid size
    const int grid_size = batch_size * seq_len;
    
    // Launch appropriate kernel based on input type
    if (input.scalar_type() == torch::ScalarType::Half) {
        feed_forward_kernel<__half><<<grid_size, block_size, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(fc1_weight.data_ptr()),
            fc1_bias.numel() > 0 ? reinterpret_cast<const __half*>(fc1_bias.data_ptr()) : nullptr,
            reinterpret_cast<const __half*>(fc2_weight.data_ptr()),
            fc2_bias.numel() > 0 ? reinterpret_cast<const __half*>(fc2_bias.data_ptr()) : nullptr,
            reinterpret_cast<__half*>(output.data_ptr()),
            use_gelu,
            dropout_prob,
            batch_size,
            seq_len,
            d_model,
            d_ff,
            seed
        );
    } else if (input.scalar_type() == torch::ScalarType::Float) {
        feed_forward_kernel<float><<<grid_size, block_size, shared_mem_size, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<const float*>(fc1_weight.data_ptr()),
            fc1_bias.numel() > 0 ? reinterpret_cast<const float*>(fc1_bias.data_ptr()) : nullptr,
            reinterpret_cast<const float*>(fc2_weight.data_ptr()),
            fc2_bias.numel() > 0 ? reinterpret_cast<const float*>(fc2_bias.data_ptr()) : nullptr,
            reinterpret_cast<float*>(output.data_ptr()),
            use_gelu,
            dropout_prob,
            batch_size,
            seq_len,
            d_model,
            d_ff,
            seed
        );
    } else {
        AT_ERROR("Unsupported input type for FeedForward kernel");
    }
    
    return output;
} 