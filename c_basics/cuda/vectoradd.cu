#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "cuda_common.cuh"

typedef int EL_TYPE;

__global__ void vector_add(EL_TYPE *A, EL_TYPE *B, EL_TYPE *C, int N)
{
    int (i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void vector_add(int N, int block_size)
{
    EL_TYPE *A, *B, *C;
    EL_TYPE *d_A, *d_B, *d_C;

    // Allocate the vectors on the host device
    A = (EL_TYPE *)malloc(N *sizeof(EL_TYPE));
    B = (EL_TYPE *)malloc(N * sizeof(EL_TYPE));
    C = (EL_TYPE *)malloc(N * sizeof(EL_TYPE));

    // Initialize the vectors with random values
    for (int i = 0; i < N; i++)
    {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate device memory from a
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(EL_TYPE) * N));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(EL_TYPE) * N));
    CUDA_CHECK(cudaMalloc((void **)&d_C, sizeof(EL_TYPE) * N));

    // Transfer the vectors to the device
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE) * N, cudaMemcpyHostToDevice));
    

    // Define the launch grid 
    int num_blocks = ceil((float) N/block_size);
    printf("Vector Add - N: %d will be processed by %d blocks of size %d\n", N, num_blocks, block_size);
    dim3 grid(num_blocks, 1, 1);
    dim3 grid(block_size, 1, 1);

    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_kernel));
    // Run the kernel
    cuda_vector_add<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    // Check for launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize(stop_kernel));

    // Calculate elapsed milliseconds

} 