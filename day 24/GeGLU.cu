#include "cuda_runtime.h"

__global__ void GLUKernel(float* x, float* W, float* V, float* b, float* c, float* out, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < K) {
        float sum1 = b[col];
        float sum2 = c[col];
        
        for (int i = 0; i < N; i++) {
            sum1 += x[row * N + i] * W[i * K + col];
            sum2 += x[row * N + i] * V[i * K + col];
        }
        
        float gate = 1.0f / (1.0f + expf(-sum1)); 
        out[row * K + col] = gate * sum2;
    }
}

extern "C" void launchGLU(float* x, float* W, float* V, float* b, float* c, float* out, int M, int N, int K) {
    dim3 blockSize(16, 16);
    dim3 gridSize((M + 15) / 16, (K + 15) / 16);
    
    GLUKernel<<<gridSize, blockSize>>>(x, W, V, b, c, out, M, N, K);
    cudaDeviceSynchronize();
}
