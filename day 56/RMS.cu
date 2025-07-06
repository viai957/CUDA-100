#include <cuda_runtime.h>

// Define the CEILING macro
#define CEILING(x, y) (((x) + (y) - 1) / (y))

#define blockdimy 128

__global__ void RMSKernel1_V1(float *input, float *output, const int w, const int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < h && col < w)
    {
        float sum = 0;
        for (int i = 0; i < w; ++i)
        {
            sum += input[row * w + i] * input[row * w + i];
        }
        sum = sqrt((float)1 / w * sum);

        output[row + w * col] = input[row * w + col] / sum;
    }
}


void RMSV1(float *input, float *output, int w, int h)
{

    dim3 block_size = dim3(32, 32);
    dim3 grid_size = dim3(CEILING(w, 32), CEILING(32, h));
    RMSKernel1_V1<<<grid_size, block_size>>>(input, output, w, h);
}



