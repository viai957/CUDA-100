/*
 * GELU: Optimized CUDA kernel for GELU activation function
 * Math: GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of the standard normal distribution
 *       Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * Inputs / Outputs:
 *   - input: Input tensor (any shape) - half precision
 *   - output: Output tensor (same shape as input) - half precision
 * Assumptions:
 *   - Forward pass only
 *   - Vectorized memory access for better throughput
 *   - Handles arbitrary tensor sizes
 * Parallel Strategy:
 *   - 1D grid of thread blocks
 *   - Each thread processes multiple elements based on vector width
 * Mixed Precision Policy:
 *   - Input/output in FP16
 *   - Internal computation in FP32 for numerical stability
 * Distributed Hooks: N/A (single GPU implementation)
 * Complexity:
 *   - FLOPs: ~10 * N (where N is the number of elements)
 *   - Bytes moved: 2 * N * sizeof(half) (read input + write output)
 * Test Vectors:
 *   - input = [-2.0, -1.0, 0.0, 1.0, 2.0]
 *   - expected output ≈ [-0.046, -0.159, 0.0, 0.841, 1.954]
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

// Constants for GELU approximation
#define GELU_COEF_1 0.7978845608028654f   // sqrt(2/π)
#define GELU_COEF_2 0.044715f

// Vector width for memory access
#define VECTOR_WIDTH 4

// GELU kernel with vectorized memory access
template<int THREADS_PER_BLOCK>
__global__ void GeluForwardKernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int n)
{
    // Each thread processes VECTOR_WIDTH elements at a time
    using Vec4Half = float4;
    
    // Get thread global index
    int idx = blockIdx.x * THREADS_PER_BLOCK * VECTOR_WIDTH + threadIdx.x * VECTOR_WIDTH;
    
    // Process elements in chunks of VECTOR_WIDTH
    if (idx + VECTOR_WIDTH <= n) {
        // Load VECTOR_WIDTH elements at once
        Vec4Half in_vec;
        half* in_ptr = (half*)&in_vec;
        
        // Load vector
        #pragma unroll
        for (int i = 0; i < VECTOR_WIDTH; i++) {
            in_ptr[i] = input[idx + i];
        }
        
        // Process each element in the vector
        Vec4Half out_vec;
        half* out_ptr = (half*)&out_vec;
        
        #pragma unroll
        for (int i = 0; i < VECTOR_WIDTH; i++) {
            // Convert to float for computation
            float x = __half2float(in_ptr[i]);
            
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            float x_cubed = x * x * x;
            float inner = GELU_COEF_1 * (x + GELU_COEF_2 * x_cubed);
            float tanh_inner = tanhf(inner);
            float result = 0.5f * x * (1.0f + tanh_inner);
            
            // Convert back to half
            out_ptr[i] = __float2half(result);
        }
        
        // Store vector
        #pragma unroll
        for (int i = 0; i < VECTOR_WIDTH; i++) {
            output[idx + i] = out_ptr[i];
        }
    }
    else if (idx < n) {
        // Handle remaining elements (tail)
        for (int i = 0; i < VECTOR_WIDTH && idx + i < n; i++) {
            float x = __half2float(input[idx + i]);
            float x_cubed = x * x * x;
            float inner = GELU_COEF_1 * (x + GELU_COEF_2 * x_cubed);
            float tanh_inner = tanhf(inner);
            float result = 0.5f * x * (1.0f + tanh_inner);
            output[idx + i] = __float2half(result);
        }
    }
}

/**
 * Host-side launcher for the GELU activation function
 */
void gelu_forward_launcher(
    const half* input,
    half* output,
    int n,
    cudaStream_t stream = nullptr)
{
    nvtxRangePush("gelu_forward");
    
    // Kernel configuration
    constexpr int THREADS_PER_BLOCK = 256;
    int blocks = (n + THREADS_PER_BLOCK * VECTOR_WIDTH - 1) / (THREADS_PER_BLOCK * VECTOR_WIDTH);
    
    // Launch kernel
    GeluForwardKernel<THREADS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        input, output, n
    );
    
    // Check for errors
    CK(cudaGetLastError());
    
    nvtxRangePop();
}

#ifdef UNIT_TEST
int main() {
    // Test parameters
    const int n = 5;
    
    // Host memory allocation
    half *h_input = new half[n];
    half *h_output = new half[n];
    half *h_expected = new half[n];
    
    // Initialize test data
    // input = [-2.0, -1.0, 0.0, 1.0, 2.0]
    h_input[0] = __float2half(-2.0f);
    h_input[1] = __float2half(-1.0f);
    h_input[2] = __float2half(0.0f);
    h_input[3] = __float2half(1.0f);
    h_input[4] = __float2half(2.0f);
    
    // Expected output ≈ [-0.046, -0.159, 0.0, 0.841, 1.954]
    h_expected[0] = __float2half(-0.046f);
    h_expected[1] = __float2half(-0.159f);
    h_expected[2] = __float2half(0.0f);
    h_expected[3] = __float2half(0.841f);
    h_expected[4] = __float2half(1.954f);
    
    // Device memory allocation
    half *d_input, *d_output;
    CK(cudaMalloc(&d_input, n * sizeof(half)));
    CK(cudaMalloc(&d_output, n * sizeof(half)));
    
    // Copy data to device
    CK(cudaMemcpy(d_input, h_input, n * sizeof(half), cudaMemcpyHostToDevice));
    
    // Launch kernel
    gelu_forward_launcher(d_input, d_output, n);
    
    // Copy results back to host
    CK(cudaMemcpy(h_output, d_output, n * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool pass = true;
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = __half2float(h_expected[i]);
        float actual = __half2float(h_output[i]);
        float diff = std::abs(expected - actual);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-2f) {  // Larger tolerance due to approximation
            pass = false;
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected, actual);
        }
    }
    
    if (pass) {
        printf("✓ GELU test passed! Max difference: %f\n", max_diff);
    } else {
        printf("✗ GELU test failed! Max difference: %f\n", max_diff);
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_expected;
    
    CK(cudaFree(d_input));
    CK(cudaFree(d_output));
    
    return pass ? 0 : 1;
}
#endif

/* Profiling example & performance tips 
 * 
 * To profile this kernel:
 * 1. nsys profile --stats=true ./gelu_test
 * 
 * Performance tips:
 * - For larger tensors, consider using larger vector widths (8 or 16)
 * - For small tensors, reduce thread block size to increase occupancy
 * - Consider using shared memory for repeated access patterns
 * - For inference-only workloads, consider using lookup tables for GELU
 */ 