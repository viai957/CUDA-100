#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

/**
 * CUDA kernel for performing element-wise vector addition
 * @param inputA First input vector in device memory
 * @param inputB Second input vector in device memory
 * @param output Result vector in device memory
 * @param numElements Number of elements in each vector
 */
__global__ void vectorAdditionKernel(const float* inputA, const float* inputB, 
                                    float* output, const unsigned int numElements) {
    // Calculate global thread ID
    const unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't access beyond array bounds
    if (globalIdx < numElements) {
        output[globalIdx] = inputA[globalIdx] + inputB[globalIdx];
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            return EXIT_FAILURE; \
        } \
    } while(0)

int main() {
    // Vector dimensions
    constexpr unsigned int kNumElements = 1 << 20; // 1M elements
    constexpr size_t kMemSize = kNumElements * sizeof(float);
    
    // Allocate and initialize host vectors
    std::vector<float> hostA(kNumElements, 1.0f);
    std::vector<float> hostB(kNumElements);
    std::vector<float> hostResult(kNumElements);
    
    // Initialize vector B with index values
    for (unsigned int i = 0; i < kNumElements; ++i) {
        hostB[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float *deviceA = nullptr;
    float *deviceB = nullptr;
    float *deviceResult = nullptr;
    
    CUDA_CHECK(cudaMalloc(&deviceA, kMemSize));
    CUDA_CHECK(cudaMalloc(&deviceB, kMemSize));
    CUDA_CHECK(cudaMalloc(&deviceResult, kMemSize));
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Transfer data to device
    CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), kMemSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB.data(), kMemSize, cudaMemcpyHostToDevice));
    
    // Launch kernel with optimal thread configuration
    constexpr unsigned int kThreadsPerBlock = 256;
    const unsigned int kBlocks = (kNumElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    
    vectorAdditionKernel<<<kBlocks, kThreadsPerBlock>>>(deviceA, deviceB, deviceResult, kNumElements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Transfer results back to host
    CUDA_CHECK(cudaMemcpy(hostResult.data(), deviceResult, kMemSize, cudaMemcpyDeviceToHost));
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    
    // Display last few results for verification
    std::cout << "Vector addition completed in " << duration.count() << " ms\n";
    std::cout << "Showing last 5 results for verification:\n";
    for (unsigned int i = kNumElements - 5; i < kNumElements; ++i) {
        std::cout << "Result[" << i << "] = " << hostResult[i] << std::endl;
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceResult));
    
    return EXIT_SUCCESS;
}