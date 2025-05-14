#include <cuda_runtime.h>
#include <iostream>

#define TITLE_SIZE 16

// Example 3x2 @2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// Matrix A = (3 x 2) 
// [[1, 2],
//  [3, 4],
//  [5, 6]] 

// Matrix B = (2 x 4)
// [[1, 2, 3, 4],
//  [5, 6, 7, 8]]

// Matrix C = (3 x 4)
// [[1*1 + 2*5, 1*2 + 2*6, 1*3 + 2*7, 1*4 + 2*8],
//  [3*1 + 4*5, 3*2 + 4*6, 3*3 + 4*7, 3*4 + 4*8],
//  [5*1 + 6*5, 5*2 + 6*6, 5*3 + 6*7, 5*4 + 6*8]]

// Matrix C = (3 x 4)
// [[11, 14, 17, 20],
//  [23, 30, 37, 44],
//  [35, 46, 57, 68]]

// TILE_SIZE = 16
// BLOCK_SIZE = 1024
// A[M,K] = (3,2)
// B[K,N] = (2,4)
// C[M,N] = (3,4)

// A[3,2] = [[1, 2],
//           [3, 4],
//           [5, 6]]

// B[2,4] = [[1, 2, 3, 4],
//           [5, 6, 7, 8]]

// C[3,4] = [[0, 0, 0, 0],
//           [0, 0, 0, 0],
//           [0, 0, 0, 0]]

// Block (32, 32)
// Threads per block (1024)
// Total blocks = (M / 32) * (N / 32)
// Total threads = 1024

// Each block computes a 16x16 submatrix of C
// 16 = TILE_SIZE

// Total number of blocks = (3 / 16) * (4 / 16) = 0

// Total number of threads = 1024

// Each block will compute a 16x16 = 256 elements of C  

// Each thread will compute a single element of C
// Total number of threads = 256

// Each thread will load a 2x4 submatrix of A and B
// 2 = K / TILE_SIZE
// 4 = N / TILE_SIZE

// Each thread will compute a single element of C
// Total number of threads = 256

// Each thread will compute a single element of C
// Total number of threads = 256

// Each thread will compute a single element of C
// Total number of threads = 256

// Each thread will compute a single element of C
// Total number of threads = 256


// CUDA KERNEL
__global__ void matrixMulTiled(float* A, float *B, float *C, int M, int N, int K){
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++){
        if (row < M && tile * TILE_SIZE + tx < K){
            sharedA[tx][ty] = A[row * K + tile * TILE_SIZE + tx];
        }
        else{
            sharedA[tx][ty] = 0.0f;
        }
        if (col < N && tile * TILE_SIZE + ty < K){
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        }
        else{
            sharedB[ty][tx] = 0.0f;
        }
        __syncthread();

        for (int k = 0; k < TILE_SIZE; k++){
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();

        if (row < M && col < N){
            C[row * N + col] = sum;
        }
    }
}

int main(){
    // Define matrix dimentions
    const int M = 1024; // Number of rows in A and C
    const int N = 1024; // Number of columns in B and C
    const int K = 1024; // Number of columns in A and rows in B

    // Caculate matrix sizes in bytes
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Declare device pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Kernel launch code 
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Synchronoze device
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::cout << "Matrix multiplication completed successfully!" << std::endl;
    return 0;
}