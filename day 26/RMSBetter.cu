#include <cuda_runtime.h>

#define CEILING(x, y) (((x) + (y) - 1) / (y))

#define blockdimy 128

__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void RMSKernel_V2(float *input, float *output, const int w, const int h)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_data[32];

    float sum = 0.0f;
    if (row < h && col < w)
    {
        
    }
}