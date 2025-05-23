#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

__global__ void Attention(float *Q, float *K, float *V, float *O, int seq_len, int dim)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float softmaxscale = 1.0f / sqrtf(dim); // Use sqrtf for single-precision

    if (row < seq_len && col < dim)
    {
        float sum = 0.0f;
        for (int k = 0; k < dim; k++)
        {
            sum += Q[row * dim + k] * K[col * dim + k];
        }

        O[row * dim + col] = sum * softmaxscale;

        __syncthreads();

        float max_value = -1e30f;
        for (int k = 0; k < dim; k++)
        {
            max_value = fmaxf(max_value, O[row * dim + k]);
        }
        __syncthreads();

        float denum = 0.0f;
        for (int k = 0; k < dim; k++)
        {
            O[row * dim + k] = expf(O[row * dim + k] - max_value);
            denum += O[row * dim + k];
        }

        __syncthreads();

        for (int k = 0; k < dim; k++)
        {
            O[row * dim + k] = O[row * dim + k] / denum;
        }

        sum = 0.0f;
        for (int k = 0; k < dim; k++)
        {
            sum += O[row * dim + k] * V[col + k * dim];
        }
        O[row * dim + col] = sum;
    }
}

void cpu_attention(float *Q, float *K, float *V, float *O, int seq_len, int dim)
{
    float softmaxscale = 1.0f / sqrtf(dim);

    for (int row = 0; row < seq_len; row++)
    {
        for (int col = 0; col < dim; col++)
        {
            float sum = 0.0f;
            for (int k = 0; k < dim; k++)
            {
                sum += Q[row * dim + k] * K[k * dim + col];
            }
            O[row * dim + col] = sum * softmaxscale;

            float max_value = -1e30f;
            for (int k = 0; k < dim; k++)
            {
                max_value = fmaxf(max_value, O[row * dim + k]);
            }

            float denum = 0.0f;
            for (int k = 0; k < dim; k++)
            {
                O[row * dim + k] = expf(O[row * dim + k] - max_value);
                denum += O[row * dim + k];
            }

            for (int k = 0; k < dim; k++)
            {
                O[row * dim + k] = O[row * dim + k] / denum;
            }
            sum = 0.0f;
            for (int k = 0; k < dim; k++)
            {
                sum += O[row * dim + k] * V[k * dim + col];
            }
            O[row * dim + col] = sum;
        }
    }
}

int main()
{
    int seq_len = 128;
    int dim = 64;

    // Allocate memory using malloc (C-style)
    float *h_Q = (float *)malloc(seq_len * dim * sizeof(float));
    float *h_K = (float *)malloc(seq_len * dim * sizeof(float));
    float *h_V = (float *)malloc(seq_len * dim * sizeof(float));
    float *h_O_cuda = (float *)malloc(seq_len * dim * sizeof(float));
    float *h_O_cpu = (float *)malloc(seq_len * dim * sizeof(float));

    // Initialize input data (example)
    for (int i = 0; i < seq_len * dim; i++)
    {
        h_Q[i] = 1;
        h_K[i] = 1;
        h_V[i] = 1;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, seq_len * dim * sizeof(float));
    cudaMalloc(&d_K, seq_len * dim * sizeof(float));
    cudaMalloc(&d_V, seq_len * dim * sizeof(float));
    cudaMalloc(&d_O, seq_len * dim * sizeof(float));

    cudaMemcpy(d_Q, h_Q, seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = dim;
    dim3 blockDim(16, 16);
    dim3 gridDim((dim + 15) / 16, (seq_len + 15) / 16);

    // CUDA execution
    auto start = std::chrono::high_resolution_clock::now();
    Attention<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, seq_len, dim);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cudaMemcpy(h_O_cuda, d_O, seq_len * dim * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU execution
    start = std::chrono::high_resolution_clock::now();
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, seq_len, dim);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "CUDA execution time: " << duration.count() << " ms" << std::endl;

    // Verification (example)
    float tolerance = 1e-5f;
    bool same = true;
    for (int i = 0; i < 10; i++)
    {
        if (std::abs(h_O_cuda[i] - h_O_cpu[i]) > tolerance)
        {
            std::cout << "Difference at index " << i << ": CUDA=" << h_O_cuda[i] << ", CPU=" << h_O_cpu[i] << std::endl;
            same = false;
            break;
        }
    }
    if (same)
        std::cout << "Results are approximately the same." << std::endl;
    else
        std::cout << "Results are different." << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O_cuda);
    free(h_O_cpu);

    return 0;
}