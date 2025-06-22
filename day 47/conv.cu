#include <cuda_runtime.h>

void convLayer_forward(int M, int C, int H, int W, int K, float *X, float *W, float *Y)
{
    int m, c, h, w, p, q;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    for (m = 0; m < M; m++)
    { // for each feature maps
        for (h = 0; h < H_out; h++)
        { // for each output element
            for (w = 0; w < W_out; w++)
            {
                Y[m, h, w] = 0;
                for (c = 0; c < C; c++)
                { // sum over all input feature maps
                    for (p = 0; p < K; p++)
                    { // KxK filter
                        for (q = 0; q < K; q++)
                        {
                            Y[m, h, w] += X[c, h + p, w + p] * W[m, c, p, q];
                        }
                    }
                }
            }
        }
    }
}

void poolingLayer_forward(int M, int H, int W, int K, float *Y, float *S)
{
    int m, h, w, p, q;
    for (m = 0; m < M; m++)         // for each output feature maps
        for (h = 0; h < H / K; h++) // for each output element
            for (w = 0; w < W / K; w++)
            {
                S[m, h, w] = 0.;
                for (p = 0; p < K; p++)
                { // loop over KxK input samples
                    for (q = 0; q < K; q++)
                        S[m, h, w] = S[m, h, w] + Y[m, K * w + p, K * h + q] / (K * K);
                }
                // add bias and apply non-linear activation
                S[m, h, w] = sigmoid(S[m, h, w] + b[m]);
            }
}

void convLayer_backward_xgrad(int M, int C, int H_in, int W_in, int K,
                              float *dE_dY, float *W, float *dE_dX)
{
    int m, c, h, w, p, q;
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;
    for (c = 0; c < C; c++)
        for (h = 0; h < H_in; h++)
            for (w = 0; w < W_in; w++)
                dE_dX[c, h, w] = 0.;
    for (m = 0; m < M; m++)
        for (h = 0; h < H_out; h++)
            for (w = 0; w < W_out; w++)
                for (c = 0; c < C; c++)
                    for (p = 0; p < K; p++)
                        for (q = 0; q < K; q++)
                            dE_dX[c, h + p, w + q] += dE_dY[m, h, w] * W[m, c, p, q];
}

void convLayer_backward_wgrad(int M, int C, int H, int W, int K,
                              float *dE_dY, float *X, float *dE_dW)
{
    int m, c, h, w, p, q;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    for (m = 0; m < M; m++)
        for (c = 0; c < C; c++)
            for (p = 0; p < K; p++)
                for (q = 0; q < K; q++)
                    dE_dW[m, c, p, q] = 0.;
    for (m = 0; m < M; m++)
        for (h = 0; h < H_out; h++)
            for (w = 0; w < W_out; w++)
                for (c = 0; c < C; c++)
                    for (p = 0; p < K; p++)
                        for (q = 0; q < K; q++)
                            dE_dW[m, c, p, q] += X[c, h + p, w + q] * dE_dY[m, c, h, w];
}
# define TILE_WIDTH 16
__global__ void ConvLayerForward_Kernel(int C, int W_grid, int K, float *X, float *W, float *Y)
{
    int n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float *X_shared = &shmem[0];
    float *W_shared = &shmem[X_tile_width * X_tile_width];
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x; // h0 and w0 used as shorthand for threadIdx.x and threadIdx.y
    w0 = threadIdx.y;
    h_base = (blockIdx.z / W_grid) * TILE_SIZE; // vertical base out data index for the block
    w_base = (blockIdx.z % W_grid) * TILE_SIZE; // horizontal base out data index for the block
    h = h_base + h0;
    w = w_base + w0;
    float acc = 0.;
    int c, i, j, p, q;
    for (c = 0; c < C; c++)
    { 
        if ((h0 < K) && (w0 < K))
            W_shared[h0, w0] = W[m, c, h0, w0];
        __syncthreads();
        for (i = h; i < h_base + X_tile_width; i += TILE_WIDTH)
        {
            for (j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
                X_shared[i - h_base, j - w_base] = X[n, c, h, w];
        }
        __syncthreads();
        for (p = 0; p < K; p++)
        {
            for (q = 0; q < K; q++)
                acc = acc + X_shared[h + p, w + q] * W_shared[p, q];
        }
        __syncthreads();
    }
    Y[n, m, h, w] = acc;
}