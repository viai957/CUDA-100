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

    // Calculate elapsed millisecondus
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Vector Add - elapsed time: %f ms\n", milliseconds_kernel);

    // Copy back the result from the device to the host
    CUDA_CHECK(cudaMemcpy(C, d_C, sizeof(EL_TYPE) * N, cudaMemcpyDeviceToHost));

    // Free the memory on the device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Time the operation
    struct timeval start_check, end_check;
    gettimeofday(&start_check, NULL);

    for (int i = 0; i < N; i++)
    {
        // Check if the result is correct
        if (C[i] != A[i] + B[i])
        {
            printf("Error at index %d: %d + %d\n", i, A[i], B[i]);
            exit(1);
        }
    }

    // Calculate elapsed time
    gettimeofday(&end_check, NULL);
    float elapsed = (end_check.tv_sec - start_check.tv_sec) * 1000.0f + (end_check.tv_usec - start_check.tv_usec) / 1000.0f;
    printf("Vector Add - elapsed time: %f ms\n", elapsed);
    printf("Vector Add - throughput: %f GB/s\n", (N * sizeof(EL_TYPE) / 1e9) / (elapsed / 1000.0f));
    printf("Vector Add - bandwidth: %f GB/s\n", (N * sizeof(EL_TYPE) * 2 / 1e9) / (elapsed / 1000.0f));

    // Free the memory on the host
    free(A);
    free(B);
    free(C);
}

int main()
{
    // Set the seed for the random number generator
    srand(0);

    vector_add(1000000, 1024);
    vector_add(1000000, 512);
    vector_add(1000000, 256);
    vector_add(1000000, 128);
    vector_add(1000000, 64);
    vector_add(1000000, 32);
    vector_add(1000000, 16);
    vector_add(1000000, 8);
    vector_add(1000000, 4);
    vector_add(1000000, 2);
    vector_add(1000000, 1);
}
