#include <cuda_runtime.h>

#define CEILING(x, y) (((x) + (y) - 1) / (y))

#define blockdimy 128

__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset, 32);
    }
    return val;
}

__global__ void RMSKernel_V2(float *input, float *output, const int w, const int h)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float shared_data[32];

    float sum = 0.0f;

    if (row < h && col < w)
    {
        float4 val = reinterpret_cast<float4 *>(&input[row * w + col * 4])[0];
        sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    __syncthreads();

    sum = warpReduceSum(sum);

    __syncthreads();

    if (threadIdx.x % 32 == 0)
    {
        shared_data[threadIdx.x / 32] = sum;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float final_sum = 0.0f;
        for (int i = 0; i < blockDim.x / 32; ++i)
        {
            final_sum += shared_data[i];
        }
        output[row] = input[row] / sqrt(final_sum / float(w));
    }
}

void RMSV2(float *input, float *output, int w, int h)
{
    dim3 block_size = dim3(1, 32, 1);
    dim3 grid_size = dim3(h, 1, 1);
    RMSKernel_V2<<<grid_size, block_size>>>(input, output, w, h);
}
