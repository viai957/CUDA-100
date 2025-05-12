#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdint.h>
#include <chrono>
#include "cuda_common.cuh"

/*
 * Matrix Multiplication Benchmark
 * 
 * Purpose:
 * This program performs matrix multiplication C = A * B on both CPU and GPU,
 * compares their performance, and benchmarks execution time, throughput, and GFLOPS.
 * 
 * Parameters:
 * - M: Number of rows in matrix A and C
 * - K: Number of columns in matrix A and rows in matrix B
 * - N: Number of columns in matrix B and C
 * 
 * Benchmarking:
 * - Measures average execution time over multiple runs for CPU and GPU implementations.
 * - Calculates total floating-point operations (2 * M * N * K).
 * - Computes GFLOPS (billion floating-point operations per second).
 * - Estimates effective memory bandwidth.
 */

#define M 256 // Number of rows in A and C
#define K 512 // Number of columns in A, and rows in B
#define N 256 // Number of columns in B and C
#define BLOCK_SIZE 32 // Size of the block for matrix multiplication

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

// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int num_rows_a, int num_cols_a, int num_cols_b){
    for (int i = 0; i < num_rows_a; i++){
        for (int j = 0; j < num_cols_b; j++){
            float sum = 0.0f;
            for (int k = 0; k < num_cols_a; k++){
                sum += A[i * num_cols_a + k] * B[k * num_cols_b + j];
            }
            C[i * num_cols_b + j] = sum;
        }
    }
}

// GPU matrix multiplication
__global__ void matmul_kernel(float *A, float *B, float *C, int num_rows_a, int num_cols_a, int num_cols_b){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows_a && col < num_cols_b){
        float sum = 0.0f;
        for (int l = 0; l < num_cols_a; l++){
            sum += A[row * num_cols_a + l] * B[l * num_cols_b + col];
        }
        C[row * num_cols_b + col] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols){
    for (int i = 0; i < rows * cols; i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time using CLOCK_MONOTONIC (not used after chrono inclusion)
double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Calculate GFLOPS given execution time in seconds
double calculate_gflops(double time_sec){
    double total_ops = 2.0 * M * N * K; // 2 * M * N * K floating point operations
    return (total_ops / time_sec) / 1e9;
}

int main(){
    // Print matrix dimensions and block size for clarity
    printf("Matrix dimensions: M = %d, K = %d, N = %d\n", M, K, N);
    printf("CUDA block size: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C_cpu = (float *)malloc(size_C);
    h_C_gpu = (float *)malloc(size_C);

    // Initialize matrices with random values
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device memory with error checking
    cudaError_t err;
    err = cudaMalloc(&d_A, size_A);
    if (err != cudaSuccess) { printf("cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc(&d_B, size_B);
    if (err != cudaSuccess) { printf("cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc(&d_C, size_C);
    if (err != cudaSuccess) { printf("cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Copy data from host to device with error checking
    err = cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy d_A failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy d_B failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE); // Ceiling division

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++){
        // Measure CPU time using chrono high_resolution_clock
        auto start_time = std::chrono::high_resolution_clock::now();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        cpu_total_time += elapsed.count();
    }
    double cpu_avg_time = cpu_total_time / 20;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++){
        // Measure GPU time using chrono high_resolution_clock
        auto start_time = std::chrono::high_resolution_clock::now();
        matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("Kernel launch failed: %s\n", cudaGetErrorString(err)); return -1; }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); return -1; }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        gpu_total_time += elapsed.count();
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Copy result back from device to host for validation (optional)
    err = cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy device to host failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Calculate total floating point operations
    double total_ops = 2.0 * M * N * K;

    // Calculate GFLOPS for CPU and GPU
    double cpu_gflops = total_ops / cpu_avg_time / 1e9;
    double gpu_gflops = total_ops / gpu_avg_time / 1e9;

    // Calculate effective memory bandwidth (bytes moved per multiplication)
    // For matrix multiplication, we read A (M*K), B (K*N), and write C (M*N)
    // Assuming each element is read/written once per multiplication
    double total_bytes = (M * K + K * N + M * N) * sizeof(float);
    double cpu_bandwidth = total_bytes / cpu_avg_time / (1e9); // GB/s
    double gpu_bandwidth = total_bytes / gpu_avg_time / (1e9); // GB/s

    // Print results with detailed benchmarking info
    printf("\n===== Benchmark Results =====\n");
    printf("CPU average time: %.6f seconds\n", cpu_avg_time);
    printf("GPU average time: %.6f seconds\n", gpu_avg_time);
    printf("Speedup (CPU time / GPU time): %.2f\n", cpu_avg_time / gpu_avg_time);
    printf("Total floating point operations: %.0f\n", total_ops);
    printf("CPU GFLOPS: %.2f\n", cpu_gflops);
    printf("GPU GFLOPS: %.2f\n", gpu_gflops);
    printf("CPU effective memory bandwidth: %.2f GB/s\n", cpu_bandwidth);
    printf("GPU effective memory bandwidth: %.2f GB/s\n", gpu_bandwidth);

    // Final summary table-like output
    printf("\n%-15s %-15s %-15s %-15s\n", "Metric", "CPU", "GPU", "Speedup");
    printf("%-15s %-15.6f %-15.6f %-15.2f\n", "Time (sec)", cpu_avg_time, gpu_avg_time, cpu_avg_time / gpu_avg_time);
    printf("%-15s %-15.2f %-15.2f %-15.2f\n", "GFLOPS", cpu_gflops, gpu_gflops, gpu_gflops / cpu_gflops);
    printf("%-15s %-15.2f %-15.2f %-15.2f\n", "Bandwidth(GB/s)", cpu_bandwidth, gpu_bandwidth, gpu_bandwidth / cpu_bandwidth);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
