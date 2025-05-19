#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void LayerNorm(const float* A, float *B, int rows, int cols){
    // Calculate row Index
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols){
        // Copy row and column data to shared memory
        extern __shared__ float shared[];
        float *row_data = shared;
        float *col_data = shared + blockDim.x; 

        // Copy row data to shared memory
        for (int c = threadIdx.y; c < cols; c += blockDim.y){
            row_data[c] = A[row * cols + c];
        }
        __syncthreads();

        // Compute mean
        float mean = 0.0f;
        for (int c = 0; c < cols; c++){
            mean += row_data[c];
        }
        mean /= cols;

        // Compute variance
        float variance = 0.0f;
        for (int c = 0; c < cols; c++){
            variance += (row_data[c] - mean) * (row_data[c] - mean);
        }
        variance /= cols;
        float stddev = sqrtf(variance + 1e-7);

        // Normalize
        for (int c = threadIdx.y; c < cols; c += blockDim.y){
            B[row * cols + c] = (row_data[c] - mean) / stddev;
        }
    }
}

int mean(){
    const int rows = 10, cols = 10;
    float *A, *B;

    // Allocate host memory
    A = (float *)malloc(rows * cols * sizeof(float));
    B = (float *)malloc(rows * cols * sizeof(float));

    // Initialize input matrix
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            A[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Allocate device memory
    float *d_a, *d_b;
    cudaMalloc(&d_a, rows * cols * sizeof(float));
    cudaMalloc(&d_b, rows * cols * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    int blocksize = 256;
    int gridsize = (rows + blocksize - 1) / blocksize;
    size_t shared_memory_size = cols * sizeof(float);
    LayerNorm<<<gridsize, blocksize, shared_memory_size>>>(d_a, d_b, rows, cols);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(B, d_b, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print  results
    printf("A:\n");
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            printf("%.2f", B[i * cols + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(A);
    free(B);

    return 0;
}