#include <cuda_runtime.h>

// Define the ceil macro
#define CEILING(x, y) (((x) + (y) - 1) / (y))
#define blockdimy 128

__global__ void RMSKernel1_V1(float *input, float *output, const int w, const int h)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < h && col < w)
    {
        float sum = 0.0f;
        for (int i = 0; i < w; i++)
        {
            sum += input[row * w + i] * input[row * w + i];
        }
        sum = sqrt((float)1 / w * sum);

        output[row + w * col] = input[row * w + col] / sum;
    }
}

void RMSV1(float *input, float * output, int w, int h)
{
    dim3 block_size = dim3(32, 32);
    dim3 grid_size = dim3(CEILING(w, 32), CEILING(h, 32));
    RMSKernel1_V1<<<grid_size, block_size>>>(input, output, w, h);
    cudaDeviceSynchronize();
}

int main()
{
    int w = 1024;
    int h = 1024;

    float *h_input = new float[w * h];
    float *h_output = new float[w * h];

    for (int i = 0; i < w * h; i++)
    {
        h_input[i] = 1;
    }
}