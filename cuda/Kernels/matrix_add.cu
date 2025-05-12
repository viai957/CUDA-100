#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "cuda_common.cuh"

typedef float EL_TYPE;

__global__ void matrix_add(EL_TYPE *A, EL_TYPE *B, EL_TYPE *C, int num_rows, int num_cols)
{
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < num_rows && col_index < num_cols)
    {
        size_t index = static_cast<size_t>(row_index) * num_cols + col_index; // A[row_index][col_index]
        C[index] = A[index] + B[index];
    }
}

void matrix_add(int num_cols, int num_rows, int rows_block_size, int cols_block_size)
{
    EL_TYPE *A, *B, *C;
    EL_TYPE *d_A, *d_B, *d_C;

    // Allocate the matrix on the host device
    A = (EL_TYPE *)malloc(sizeof(EL_TYPE) * num_cols * num_rows);
    B = (EL_TYPE *)malloc(sizeof(EL_TYPE) * num_cols * num_rows);
    C = (EL_TYPE *)malloc(sizeof(EL_TYPE) * num_cols * num_rows);

    // Initialize the matrices with random values
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            size_t index = static_cast<size_t>(i) * num_cols + j; // A[i][j]
            A[index] = rand() % 100;
            B[index] = rand() % 100;
        }
    }
     
    // Allocate memory on the device 
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(EL_TYPE) * num_cols * num_rows));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(EL_TYPE) * num_cols * num_rows));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(EL_TYPE) * num_cols * num_rows));

    // Transfer the matrices to the device 
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE) * num_cols * num_rows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE) * num_cols * num_rows, cudaMemcpyHostToDevice));


    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_kernel, 0));


    // Define the launch grid 
    int num_blocks_rows = (num_rows + rows_block_size - 1) / rows_block_size;
    int_num_blocks_cols = (num_cols + cols_block_size - 1) / cols_block_size;
    printf("Matrix Add - M: %d, N: %d will be processed by (%d x %d) blocks of size (%d x %d)\n", num_rows, num_cols, num_blocks_rows, num_blocks_cols, rows_block_size, cols_block_size);
    dim3 grid(num_blocks_cols, num_blocks_rows);
    dim3 block(cols_block_size, rows_block_size);
    // run the kernel
    matrix_add<<<grid, block>>>(d_A, d_B, d_C, num_rows, num_cols);

    // check for launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    // Calculate elapsed time milliseconds
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Matrix Add - Elapsed time: %f ms\n", milliseconds_kernel);

    // Copy back the result from the device to the host
    CUDA_CHECK(cudaMemcpy(C, d_C, sizeof(EL_TYPE) * num_cols * num_rows, cudaMemcpyDeviceToHost));

    // Free the memory on the device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B ));
    CUDA_CHECK(cudaFree(d_C));

    // Time the operation
    struct timeval start_check, end_check;
    gettimeofday(&start_check, NULL);

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            size_t index = static_cast<size_t>(i) * num_cols + j; // A[i][j]
            if (C[index] != A[index] + B[index])
            {
                printf("Error at index (%d, %d): %.2f != %.2f + %.2f\n", i, j, A[index], B[index], C[index]);
                exit(1);
            }
        }
    }
    
    // Calculate the elaapsed time
    gettimeofday(&end_check, NULL);
    float elapsed = (end_check.tv_sec - start_check.tv_sec) * 1000.0 + (end_check.tv_usec - start_check.tv_usec) / 1000.0;
    printf("Matrix Add - Check elapsed time: %f ms\n", elapsed);

    printf("Matrix Add - Result OK\n");

    // Free the memory on the host
    free(A);
    free(B);
    free(OUT);
}

int main()
{
    // set seed for random number generation
    srand(0);

    matrix_add(1024, 1024, 1024, 1024);
    matrix_add(1024, 1024, 1024, 1);
    matrix_add(1024, 1024, 1, 1024);
    matrix_add(1024, 1024, 1, 1);
}