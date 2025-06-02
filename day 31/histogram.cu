#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16  // 16x16 thread block
#define HIST_SIZE 256  // Grayscale histogram bins

__global__ void histogram_equalization(unsigned char *d_img, unsigned char *d_out, int width, int height) {
    __shared__ unsigned int hist_shared[HIST_SIZE];  // Shared memory for histogram
    __shared__ float cdf_shared[HIST_SIZE];          // Shared memory for CDF

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int index = y * width + x;
    
    // Initialize shared histogram
    if (tx < HIST_SIZE / BLOCK_SIZE && ty == 0) {
        hist_shared[tx * BLOCK_SIZE] = 0;
    }
    __syncthreads();

    // First pass: compute local histogram using atomic operations
    if (x < width && y < height) {
        atomicAdd(&hist_shared[d_img[index]], 1);
    }
    __syncthreads();

    // Merge local histograms into global memory
    __shared__ unsigned int hist_global[HIST_SIZE]; 
    if (tx == 0 && ty == 0) {
        for (int i = 0; i < HIST_SIZE; i++) {
            atomicAdd(&hist_global[i], hist_shared[i]);
        }
    }
    __syncthreads();

    // Compute CDF (Cumulative Distribution Function)
    if (tx == 0 && ty == 0) {
        float sum = 0;
        for (int i = 0; i < HIST_SIZE; i++) {
            sum += hist_global[i];
            cdf_shared[i] = sum;
        }

        // Normalize the CDF
        float min_cdf = cdf_shared[0];
        for (int i = 0; i < HIST_SIZE; i++) {
            cdf_shared[i] = ((cdf_shared[i] - min_cdf) / (width * height - min_cdf)) * 255.0f;
        }
    }
    __syncthreads();

    // Apply equalization
    if (x < width && y < height) {
        d_out[index] = (unsigned char)cdf_shared[d_img[index]];
    }
}