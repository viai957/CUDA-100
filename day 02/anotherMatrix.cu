#include <iostream>
#include <cuda_runtime.h>

__device__ float randomFunc(float x, float y){
    return x + y * 0.5f;
}

__global__ void matrixAddKernel(float *A, float *B, float *C, const int size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size){
        C[col + size * row] = randomFunc(A[col + size * row], B[row + size * col]);
    }
}

int main(){
    int N = 1024;
    int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int size = sizeof(float) * N * N;

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    A = new float[N * N];
    B = new float[N * N];
    C = new float[N * N];

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Initialize matrices
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
        }
    }

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print a small portion of the result (printing all would be too much)
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}