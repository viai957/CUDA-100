/*
 * Conv1d: Optimized CUDA kernel for 1D convolution operation
 * Math: Y[b, o, p] = Σ_i Σ_k X[b, i, p + k] * W[o, i, k] + b[o]
 * Inputs / Outputs:
 *   - input: Input tensor (batch_size, in_channels, in_width) - half precision
 *   - weight: Weight tensor (out_channels, in_channels, kernel_size) - half precision
 *   - bias: Bias tensor (out_channels) - half precision (optional)
 *   - output: Output tensor (batch_size, out_channels, out_width) - half precision
 *   - out_width = (in_width + 2*padding - (kernel_size - 1) - 1) / stride + 1
 * Assumptions:
 *   - Forward pass only
 *   - Supports padding and striding
 *   - Optimized for small kernel sizes (3, 5, 7)
 *   - Support for dilation can be added but not implemented here
 * Parallel Strategy:
 *   - 3D grid of thread blocks (batch_size, out_channels, out_width)
 *   - Each thread block computes one output element
 *   - Shared memory for input and weight reuse
 * Mixed Precision Policy:
 *   - Input/output/weights in FP16
 *   - Internal accumulation in FP32 for numerical stability
 * Distributed Hooks: N/A (single GPU implementation)
 * Complexity:
 *   - FLOPs: batch_size * out_channels * out_width * in_channels * kernel_size
 *   - Bytes moved: (batch_size * in_channels * in_width + 
 *                   out_channels * in_channels * kernel_size +
 *                   batch_size * out_channels * out_width) * sizeof(half)
 * Test Vectors:
 *   - batch_size=2, in_channels=4, in_width=8, out_channels=6, kernel_size=3, stride=1, padding=1
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <nvToolsExt.h>

// Error checking macro
#define CK(call) \
  { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      std::exit(EXIT_FAILURE); \
    } \
  }

// Constants for convolution
#define MAX_KERNEL_SIZE 7
#define THREADS_PER_BLOCK 256

/**
 * CUDA kernel for 1D convolution
 * This kernel is optimized for small kernel sizes common in Whisper (3x3)
 * Each thread computes one output element
 */
template<int KERNEL_SIZE, int THREADS_X, bool WITH_BIAS>
__global__ void Conv1dForwardKernel(
    const half* __restrict__ input,      // [batch_size, in_channels, in_width]
    const half* __restrict__ weight,     // [out_channels, in_channels, kernel_size]
    const half* __restrict__ bias,       // [out_channels]
    half* __restrict__ output,           // [batch_size, out_channels, out_width]
    int batch_size,
    int in_channels,
    int in_width,
    int out_channels,
    int out_width,
    int stride,
    int padding)
{
    // Each thread processes one output element
    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int out_width_start = blockIdx.z * THREADS_X;
    int out_width_idx = out_width_start + threadIdx.x;
    
    if (out_width_idx >= out_width) return;
    
    // Each thread loads its portion of the kernel weights into shared memory
    __shared__ float s_weight[MAX_KERNEL_SIZE];
    
    // Input position corresponding to output
    int in_width_idx = out_width_idx * stride - padding;
    
    // Compute convolution for this output element
    float result = 0.0f;
    
    // Loop over input channels
    for (int in_c = 0; in_c < in_channels; in_c++) {
        // Load weights for this input channel into shared memory
        if (threadIdx.x < KERNEL_SIZE) {
            s_weight[threadIdx.x] = __half2float(weight[(out_channel_idx * in_channels + in_c) * KERNEL_SIZE + threadIdx.x]);
        }
        
        __syncthreads();
        
        // Perform convolution
        for (int k = 0; k < KERNEL_SIZE; k++) {
            int in_idx = in_width_idx + k;
            if (in_idx >= 0 && in_idx < in_width) {
                float in_val = __half2float(input[(batch_idx * in_channels + in_c) * in_width + in_idx]);
                result += in_val * s_weight[k];
            }
        }
        
        __syncthreads();
    }
    
    // Add bias if needed
    if (WITH_BIAS) {
        result += __half2float(bias[out_channel_idx]);
    }
    
    // Write result to output
    output[(batch_idx * out_channels + out_channel_idx) * out_width + out_width_idx] = __float2half(result);
}

/**
 * More efficient kernel for the special case of kernel_size=3, stride=1, padding=1,
 * which appears in the Whisper encoder.
 */
template<bool WITH_BIAS>
__global__ void Conv1dK3S1P1Kernel(
    const half* __restrict__ input,      // [batch_size, in_channels, in_width]
    const half* __restrict__ weight,     // [out_channels, in_channels, 3]
    const half* __restrict__ bias,       // [out_channels]
    half* __restrict__ output,           // [batch_size, out_channels, out_width]
    int batch_size,
    int in_channels,
    int in_width,
    int out_channels,
    int out_width)
{
    // Each thread processes one output element
    int batch_idx = blockIdx.x;
    int out_c_idx = blockIdx.y;
    int out_w_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (out_w_idx >= out_width) return;
    
    // Input width index (with padding of 1)
    int in_w_idx = out_w_idx; // For stride=1
    
    // Each thread loads its weights into registers
    __shared__ float s_weight[3 * 32]; // Support up to 32 input channels per thread block
    
    // Load weights into shared memory
    for (int i = threadIdx.x; i < in_channels * 3; i += blockDim.x) {
        int in_c = i / 3;
        int k = i % 3;
        if (in_c < in_channels) {
            s_weight[i] = __half2float(weight[(out_c_idx * in_channels + in_c) * 3 + k]);
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    float result = 0.0f;
    
    for (int in_c = 0; in_c < in_channels; in_c++) {
        // Handle left padding
        float x_left = (in_w_idx > 0) ? __half2float(input[(batch_idx * in_channels + in_c) * in_width + in_w_idx - 1]) : 0.0f;
        
        // Current position (always valid with padding=1)
        float x_center = __half2float(input[(batch_idx * in_channels + in_c) * in_width + in_w_idx]);
        
        // Handle right padding
        float x_right = (in_w_idx < in_width - 1) ? __half2float(input[(batch_idx * in_channels + in_c) * in_width + in_w_idx + 1]) : 0.0f;
        
        // Get weights from shared memory
        float w0 = s_weight[in_c * 3 + 0];
        float w1 = s_weight[in_c * 3 + 1];
        float w2 = s_weight[in_c * 3 + 2];
        
        // Compute dot product for this input channel
        result += x_left * w0 + x_center * w1 + x_right * w2;
    }
    
    // Add bias if needed
    if (WITH_BIAS) {
        result += __half2float(bias[out_c_idx]);
    }
    
    // Write result to output
    output[(batch_idx * out_channels + out_c_idx) * out_width + out_w_idx] = __float2half(result);
}

/**
 * Host-side launcher for the 1D convolution
 */
void conv1d_forward_launcher(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int batch_size,
    int in_channels,
    int in_width,
    int out_channels,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream = nullptr)
{
    nvtxRangePush("conv1d_forward");
    
    // Special case for common Whisper encoder configuration
    if (kernel_size == 3 && stride == 1 && padding == 1) {
        // Grid dimensions for the optimized kernel
        dim3 block(256);
        dim3 grid(batch_size, out_channels, (out_width + block.x - 1) / block.x);
        
        if (bias != nullptr) {
            Conv1dK3S1P1Kernel<true><<<grid, block, 0, stream>>>(
                input, weight, bias, output,
                batch_size, in_channels, in_width, out_channels, out_width
            );
        } else {
            Conv1dK3S1P1Kernel<false><<<grid, block, 0, stream>>>(
                input, weight, nullptr, output,
                batch_size, in_channels, in_width, out_channels, out_width
            );
        }
    } else {
        // General case with dynamic kernel size
        constexpr int threads_x = 32;
        dim3 block(threads_x);
        dim3 grid(batch_size, out_channels, (out_width + threads_x - 1) / threads_x);
        
        // Launch appropriate kernel based on kernel size and whether bias is provided
        switch(kernel_size) {
            case 3:
                if (bias != nullptr) {
                    Conv1dForwardKernel<3, threads_x, true><<<grid, block, 0, stream>>>(
                        input, weight, bias, output,
                        batch_size, in_channels, in_width, out_channels, out_width,
                        stride, padding
                    );
                } else {
                    Conv1dForwardKernel<3, threads_x, false><<<grid, block, 0, stream>>>(
                        input, weight, nullptr, output,
                        batch_size, in_channels, in_width, out_channels, out_width,
                        stride, padding
                    );
                }
                break;
            case 5:
                if (bias != nullptr) {
                    Conv1dForwardKernel<5, threads_x, true><<<grid, block, 0, stream>>>(
                        input, weight, bias, output,
                        batch_size, in_channels, in_width, out_channels, out_width,
                        stride, padding
                    );
                } else {
                    Conv1dForwardKernel<5, threads_x, false><<<grid, block, 0, stream>>>(
                        input, weight, nullptr, output,
                        batch_size, in_channels, in_width, out_channels, out_width,
                        stride, padding
                    );
                }
                break;
            case 7:
                if (bias != nullptr) {
                    Conv1dForwardKernel<7, threads_x, true><<<grid, block, 0, stream>>>(
                        input, weight, bias, output,
                        batch_size, in_channels, in_width, out_channels, out_width,
                        stride, padding
                    );
                } else {
                    Conv1dForwardKernel<7, threads_x, false><<<grid, block, 0, stream>>>(
                        input, weight, nullptr, output,
                        batch_size, in_channels, in_width, out_channels, out_width,
                        stride, padding
                    );
                }
                break;
            default:
                fprintf(stderr, "Unsupported kernel size %d. Must be 3, 5, or 7.\n", kernel_size);
                std::exit(EXIT_FAILURE);
        }
    }
    
    // Check for errors
    CK(cudaGetLastError());
    
    nvtxRangePop();
}

/**
 * Calculate output dimensions for a 1D convolution
 */
int calculate_out_width(int in_width, int kernel_size, int stride, int padding) {
    return (in_width + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
}

#ifdef UNIT_TEST
int main() {
    // Test parameters
    const int batch_size = 2;
    const int in_channels = 4;
    const int in_width = 8;
    const int out_channels = 6;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;
    
    // Calculate output dimensions
    const int out_width = calculate_out_width(in_width, kernel_size, stride, padding);
    
    printf("Input shape: [%d, %d, %d]\n", batch_size, in_channels, in_width);
    printf("Weight shape: [%d, %d, %d]\n", out_channels, in_channels, kernel_size);
    printf("Output shape: [%d, %d, %d]\n", batch_size, out_channels, out_width);
    
    // Host memory allocation
    size_t input_size = batch_size * in_channels * in_width;
    size_t weight_size = out_channels * in_channels * kernel_size;
    size_t bias_size = out_channels;
    size_t output_size = batch_size * out_channels * out_width;
    
    half *h_input = new half[input_size];
    half *h_weight = new half[weight_size];
    half *h_bias = new half[bias_size];
    half *h_output = new half[output_size];
    half *h_expected = new half[output_size];
    
    // Initialize test data
    for (int i = 0; i < input_size; ++i) {
        h_input[i] = __float2half(0.1f * (i % 10));
    }
    for (int i = 0; i < weight_size; ++i) {
        h_weight[i] = __float2half(0.01f * (i % 20));
    }
    for (int i = 0; i < bias_size; ++i) {
        h_bias[i] = __float2half(0.5f * i);
    }
    
    // Device memory allocation
    half *d_input, *d_weight, *d_bias, *d_output;
    CK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CK(cudaMalloc(&d_weight, weight_size * sizeof(half)));
    CK(cudaMalloc(&d_bias, bias_size * sizeof(half)));
    CK(cudaMalloc(&d_output, output_size * sizeof(half)));
    
    // Copy data to device
    CK(cudaMemcpy(d_input, h_input, input_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_weight, h_weight, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bias, h_bias, bias_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Launch kernel
    conv1d_forward_launcher(d_input, d_weight, d_bias, d_output,
                           batch_size, in_channels, in_width,
                           out_channels, out_width, kernel_size,
                           stride, padding);
    
    // Copy results back to host
    CK(cudaMemcpy(h_output, d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Calculate expected output on CPU for verification
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ow = 0; ow < out_width; ++ow) {
                float result = __half2float(h_bias[oc]);
                
                // Input position corresponding to output
                int iw_start = ow * stride - padding;
                
                // Perform convolution
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int iw = iw_start + k;
                        if (iw >= 0 && iw < in_width) {
                            float in_val = __half2float(h_input[(b * in_channels + ic) * in_width + iw]);
                            float w_val = __half2float(h_weight[(oc * in_channels + ic) * kernel_size + k]);
                            result += in_val * w_val;
                        }
                    }
                }
                
                h_expected[(b * out_channels + oc) * out_width + ow] = __float2half(result);
            }
        }
    }
    
    // Verify results
    bool pass = true;
    float max_diff = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        float expected = __half2float(h_expected[i]);
        float actual = __half2float(h_output[i]);
        float diff = std::abs(expected - actual);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-3f) {
            pass = false;
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected, actual);
        }
    }
    
    if (pass) {
        printf("✓ Conv1d test passed! Max difference: %f\n", max_diff);
    } else {
        printf("✗ Conv1d test failed! Max difference: %f\n", max_diff);
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_weight;
    delete[] h_bias;
    delete[] h_output;
    delete[] h_expected;
    
    CK(cudaFree(d_input));
    CK(cudaFree(d_weight));
    CK(cudaFree(d_bias));
    CK(cudaFree(d_output));
    
    return pass ? 0 : 1;
}
#endif

/* Profiling example & performance tips 
 * 
 * To profile this kernel:
 * 1. nsys profile --stats=true ./conv1d_test
 * 
 * Performance tips:
 * - For small kernel sizes, consider using register-based approaches
 * - For large batch sizes, consider using im2col + GEMM approach
 * - For large input channels, consider tiling input channels to utilize shared memory better
 * - For repeated calls with same shapes, consider using cuDNN or persistent kernels
 */ 