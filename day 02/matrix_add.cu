#include <iostream>
#include <cuda_runtime.h>

// Error checking macro inspired by modern CUDA best practices
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Modern structure for matrix dimensions and properties
struct MatrixDims {
    int rows;
    int cols;
    int pitch;  // For memory alignment
    
    __host__ __device__ size_t size() const { 
        return rows * cols * sizeof(float); 
    }
    
    __host__ __device__ bool isValid() const { 
        return rows > 0 && cols > 0; 
    }
};

// Optimized kernel with shared memory and efficient memory access patterns
template<unsigned int BLOCK_SIZE_X = 32, unsigned int BLOCK_SIZE_Y = 32>
__global__ void MatrixAddOptimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const MatrixDims dims
) {
    // Shared memory for tile-based computation
    __shared__ float tile_A[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float tile_B[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE_Y + ty;
    const int col = blockIdx.x * BLOCK_SIZE_X + tx;
    
    if (row < dims.rows && col < dims.cols) {
        // Coalesced memory access
        const int idx = row * dims.cols + col;
        tile_A[ty][tx] = A[idx];
        tile_B[ty][tx] = B[idx];
        
        __syncthreads();  // Ensure all threads have loaded their data
        
        // Compute result with potential for future optimizations
        C[idx] = tile_A[ty][tx] + tile_B[ty][tx];
    }
}

class MatrixOperations {
public:
    static void AddMatrices(const float* h_A, const float* h_B, float* h_C, 
                           const MatrixDims& dims) {
        float *d_A, *d_B, *d_C;
        
        // Modern error-checked allocation
        CUDA_CHECK(cudaMalloc(&d_A, dims.size()));
        CUDA_CHECK(cudaMalloc(&d_B, dims.size()));
        CUDA_CHECK(cudaMalloc(&d_C, dims.size()));
        
        // Async memory transfers
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, dims.size(), 
                                 cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, dims.size(), 
                                 cudaMemcpyHostToDevice, stream));
        
        // Optimal thread block configuration
        constexpr unsigned int BLOCK_SIZE = 32;
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks(
            (dims.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (dims.rows + BLOCK_SIZE - 1) / BLOCK_SIZE
        );
        
        // Launch kernel with template parameters
        MatrixAddOptimized<BLOCK_SIZE, BLOCK_SIZE>
            <<<numBlocks, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, dims);
        
        CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, dims.size(), 
                                 cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

int main() {
    const int N = 1024;  // Larger size for better GPU utilization
    MatrixDims dims{N, N, N * sizeof(float)};
    
    // Modern C++ memory management
    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 2.0f);
    std::vector<float> h_C(N * N, 0.0f);
    
    // Perform addition
    MatrixOperations::AddMatrices(h_A.data(), h_B.data(), h_C.data(), dims);
    
    // Verify results (only printing a small subset)
    printf("Matrix Addition Results (top-left 5x5 corner):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    
    return 0;
}