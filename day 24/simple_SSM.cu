#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void spmv_csr_kernel(int num_rows, const float *values, const int *column_indices, const int *row_offsets, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        for (int i = row_offsets[row]; i < row_offsets[row + 1]; i++) {
            dot += values[i] * x[column_indices[i]];
        }
        y[row] = dot;
    }
}

void spmv_csr(int num_rows, int nnz, float *h_values, int *h_column_indices, int *h_row_offsets, float *h_x, float *h_y) {
    float *d_values;
    float*d_x;
    float *d_y;
    int *d_column_indices; 
    int *d_row_offsets;

    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_column_indices, nnz * sizeof(int));
    cudaMalloc(&d_row_offsets, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_x, num_rows * sizeof(float));
    cudaMalloc(&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, h_row_offsets, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    spmv_csr_kernel<<<gridSize, blockSize>>>(num_rows, d_values, d_column_indices, d_row_offsets, d_x, d_y);
    
    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_column_indices);
    cudaFree(d_row_offsets);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int num_rows = 3;
    int nnz = 4;
    float values[] = {1, 2, 3, 4};
    int column_indices[] = {0, 2, 1, 2};
    int row_offsets[] = {0, 1, 3, 4};
    float x[] = {1, 2, 3};
    float y[3] = {0};

    spmv_csr(num_rows, nnz, values, column_indices, row_offsets, x, y);

    std::cout << "Rezultat SpMV: ";
    for (int i = 0; i < num_rows; i++) {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}