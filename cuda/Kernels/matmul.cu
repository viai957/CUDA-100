#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 // Number of rows in A and C
#define K 512 // Number of columns in A and rows in B
#define N 128 // Number of columns in B and C
#define BLOCK_SIZE 32 // Number of threads per block

// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// A = [[1, 2],
//      [3, 4],
//      [5, 6]]
// B = [[1, 2, 3, 4],
//      [5, 6, 7, 8]]
// C = [[1*1 + 2*5, 1*2 + 2*6, 1*3 + 2*7, 1*4 + 2*8],
//      [3*1 + 4*5, 3*2 + 4*6, 3*3 + 4*7, 3*4 + 4*8],
//      [5*1 + 6*5, 5*2 + 6*6, 5*3 + 6*7, 5*4 + 6*8]]

// CPU implementation
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            float sum = 0.0f;
            for (int l = 0; l < k; l++){
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul_gpu(float *a, float *b, float *c, int n, int m, int k){
    int row = blokcIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n){
        float sum =0.0f;
        for (int l = 0; l < k; l++){
            sum += a[row * k + l] * b[l * n + col];
        }
        c[row * n + j] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int *rows, int cols){
    for (int i = 0; i < rows; i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
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

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device memory
}
