#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256

__global__ void reductionKernelOptimized(const float *g_in, float *g_out, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (BLOCK_SIZE * 2) + tid;
    
    float mySum = 0.0f;
    if (idx < n)
        mySum = g_in[idx];
    if (idx + BLOCK_SIZE < n)
        mySum += g_in[idx + BLOCK_SIZE];
    
    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
        g_out[blockIdx.x] = sdata[0];
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_array = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_array[i] = 1.0f;
    }

    float *d_in, *d_out;
    hipMalloc(&d_in, size);
    int numBlocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    hipMalloc(&d_out, numBlocks * sizeof(float));

    hipMemcpy(d_in, h_array, size, hipMemcpyHostToDevice);

    size_t sharedMemSize = BLOCK_SIZE * sizeof(float);
    hipLaunchKernelGGL(reductionKernelOptimized, dim3(numBlocks), dim3(BLOCK_SIZE), sharedMemSize, 0, d_in, d_out, n);
    hipDeviceSynchronize();

    float *h_partialSums = (float*)malloc(numBlocks * sizeof(float));
    hipMemcpy(h_partialSums, d_out, numBlocks * sizeof(float), hipMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partialSums[i];
    }
    printf("Reduction result: %f (expected %f)\n", sum, (float)n);

    free(h_array);
    free(h_partialSums);
    hipFree(d_in);
    hipFree(d_out);
    
    return 0;
}