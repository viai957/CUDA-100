#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000 // Number of elements to sort = 10 million
#define BLOCK_SIZE 1024 // Number of threads per block

// CPU bubble sort implementation
void bubble_sort_cpu(int *arr, int n){
    for (int i = 0; i < n-1; i++){
        for (int j = 0; j < n-i-1; j++){
            if (arr[j] > arr[j+1]) {
                // Swap elements
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

PARALLEL BUBBLE SORT (A)
For k = 0 to n-2
If k is even then
    for i = 0 to (n/2)-1 do in parallel
        If A[2i] > A[2i+1] then
            Exchange A[2i] ↔ A[2i+1]
Else
    for i = 0 to (n/2)-2 do in parallel
        If A[2i+1] > A[2i+2] then
            Exchange A[2i+1] ↔ A[2i+2]
Next k

// CUDA kernel for bubble sort
__global__ void bubble_sort_kernel(int *arr, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one element
    if (idx < n-1){
        // Compare and swap if needed
        if (arr[idx] > arr[idx+1]){
            // Swap elements
            int temp = arr[idx];
            arr[idx] = arr[idx+1];
            arr[idx+1] = temp;
        }
    }
    __syncthreads(); // Ensure all swaps are complete before next iteration
}

// Function to measure execution time
double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    int *h_arr, *d_arr;
    int size = N * sizeof(int);

    // Allocate host memory
    h_arr = (int *)malloc(size);

    // initialize array with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++){
        h_arr[i] = rand() % 10000000; // Random values between 0 and 9999999
    }

    // Allocate device memory
    cudaMalloc(&d_arr, size);

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 grid(num_blocks);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++){
        double start_time = get_time();
        bubble_sort_cpu(h_arr, N);
        double end_time = get_time();
        cpu_total_time += (end_time - start_time);
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    // Benchmarking GPU performance
    printf("Running GPU implementation of bubble sort...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 5; i++){
        double start_time = get_time();
        // Run multiple iterations of the kernel to get a more accurate measurement
        for (int j = 0; j < N; j++){
            bubble_sort_kernel<<<grid, block>>>(d_arr, N);
            cudaDeviceSynchronize();
        }
        double end_time = get_time();
        gpu_total_time += (end_time - start_time);
    }
    double gpu_avg_time = gpu_total_time / 5.0;

    // Print results
    printf("CPU average time: %.5f seconds\n", cpu_avg_time);
    printf("GPU average time: %.5f seconds\n", gpu_avg_time);
    printf("Speedup: %.2f\n", cpu_avg_time / gpu_avg_time);

    // Copy sorted data back to host
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Verify sorted data
    for (int i = 0; i < N-1; i++){
        if (h_arr[i] > h_arr[i+1]){
            printf("Sorting failed!\n");
            break;
        }
    }

    // Free up host and device memory
    free(h_arr);
    cudaFree(d_arr);

    return 0;
}
