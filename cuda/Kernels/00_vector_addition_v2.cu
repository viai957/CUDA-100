#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000 // Number of elements in the vectors = 10 million
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 4
// 16 * 16 * 8 = 2048 threads per block

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n){
    for (int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for 1D vector addition
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
        // one add, one store
    }
}

// CUDA kernel for 3D vector addition
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    // 3 add, 3 multiplies, 3 stores

    if (i < nx && j < ny && k < nz){
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz){
            c[idx] = a[idx] + b[idx];
        }
    }
    // 3 add, 3 multiplies, 3 stores
}

// Initialize vector with random values
void init_vector(float *vec, int n){
    for (init i = 0; i < n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N * sizeof(float);

    // Allocate memory on host
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1d = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);


    // Allocate memory on device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimentions for 1D vector addition
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    // Define grid and block dimentions for 3D vector addition
    int nx = 100, ny = 100, nz = 100; // N = 10000000 = 100 * 100 * 100
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++){
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_1d_total_time += end_time - start_time;
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 100.0;

    // Verify 1D results immediatly
    cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++){
        if (fabs(h_c_gpu_1d[i] - (h_a[i] + h_b[i])) > 1e-4){
            correct_1d = false;
            std::cout << i << "cpu: " << h_a[i] + h_b[i] << " != " << h_c_gpu_1d[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");


    // Benchmark GPU 3D vector addition
    printf("Benchmarking GPU 3D vector addition...\n");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < N; i++){
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

    // Print results
    printf("CPU Avg Time: %f milliseconds\n", cpu_total_time * 1000.0);
    printf("GPU 1D Avg Time: %f milliseconds\n", gpu_1d_avg_time * 1000.0);
    printf("GPU 3D Avg Time: %f milliseconds\n", gpu_3d_avg_time * 1000.0);
    printf("Speedup: (CPU vs GPU 1D): %fx\n", cpu_total_time / gpu_1d_avg_time);
    printf("Speedup: (CPU vs GPU 3D): %fx\n", cpu_total_time / gpu_3d_avg_time);
    printf("Speedup: (GPU 1D vs GPU 3D): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);


    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);  
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}

