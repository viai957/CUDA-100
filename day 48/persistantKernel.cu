#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#define IMAGESIZE 1024  // 32 x 32 x 1
#define NUM_IMAGES 1024 // Total number of images
#define THREADS_PER_BLOCK 256

__device__ float dummyKernel(const float *image)
{
    float sum = 0.0f;
    for (int i = 0; i < IMAGESIZE; i++)
    {
        sum += image[i];
    }
    return sum;
}

__global__ void persistentKernel(const float *ImageData, float *output, const int *taskQueue,
                                 int numTask, volatile int *QueueHead,
                                 volatile bool *doneFlag)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (!(*doneFlag))
    {
        int taskIndex = atomicAdd((int *)QueueHead, 1);
        if (taskIndex < numTask)
        {
            int imageIndex = taskQueue[taskIndex];
            int imageOffset = imageIndex * IMAGESIZE;
            float result = dummyKernel(&ImageData[imageOffset]);
            output[imageIndex] = result;
        }
        else
        {
            __nanosleep(100);
            if (atomicAdd((int *)QueueHead, 0) >= numTask)
            {
                *doneFlag = true;
            }
        }
    }
}

__global__ void normalKernel(const float *ImageData, float *output)
{
    int imageIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (imageIndex < NUM_IMAGES)
    {
        int imageOffset = imageIndex * IMAGESIZE;
        float result = dummyKernel(&ImageData[imageOffset]);
        output[imageIndex] = result;
    }
}

int main()
{
    size_t imageDataSize = NUM_IMAGES * IMAGESIZE * sizeof(float);
    float *h_ImageData = new float[NUM_IMAGES * IMAGESIZE];
    float *h_output = new float[NUM_IMAGES];
    int *h_taskQueue = new int[NUM_IMAGES];

    for (int i = 0; i < NUM_IMAGES; i++)
    {
        for (int j = 0; j < IMAGESIZE; j++)
        {
            h_ImageData[i * IMAGESIZE + j] = static_cast<float>(rand()) / RAND_MAX;
        }
        h_taskQueue[i] = i;
    }

    // Device allocations.
    float *d_ImageData, *d_output;
    int *d_taskQueue;
    int *d_QueueHead;
    bool *d_doneFlag;
    cudaMalloc(&d_ImageData, imageDataSize);
    cudaMalloc(&d_output, NUM_IMAGES * sizeof(float));
    cudaMalloc(&d_taskQueue, NUM_IMAGES * sizeof(int));
    cudaMalloc(&d_QueueHead, sizeof(int));
    cudaMalloc(&d_doneFlag, sizeof(bool));

    cudaMemcpy(d_ImageData, h_ImageData, imageDataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_taskQueue, h_taskQueue, NUM_IMAGES * sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    bool falseFlag = false;
    cudaMemcpy(d_QueueHead, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_doneFlag, &falseFlag, sizeof(bool), cudaMemcpyHostToDevice);

    int blocks = (THREADS_PER_BLOCK > NUM_IMAGES ? 1 : (NUM_IMAGES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    cudaEvent_t startPersistent, stopPersistent;
    cudaEventCreate(&startPersistent);
    cudaEventCreate(&stopPersistent);
    cudaEventRecord(startPersistent);

    persistentKernel<<<blocks, THREADS_PER_BLOCK>>>(d_ImageData, d_output, d_taskQueue, NUM_IMAGES, d_QueueHead, d_doneFlag);
    cudaEventRecord(stopPersistent);
    cudaEventSynchronize(stopPersistent);
    float persistentTime;
    cudaEventElapsedTime(&persistentTime, startPersistent, stopPersistent);

    cudaMemcpy(h_output, d_output, NUM_IMAGES * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Persistent Kernel Execution Time: " << persistentTime << " ms" << std::endl;

    cudaMemset(d_output, 0, NUM_IMAGES * sizeof(float));

    cudaEvent_t startNormal, stopNormal;
    cudaEventCreate(&startNormal);
    cudaEventCreate(&stopNormal);
    cudaEventRecord(startNormal);

    normalKernel<<<blocks, THREADS_PER_BLOCK>>>(d_ImageData, d_output);
    cudaEventRecord(stopNormal);
    cudaEventSynchronize(stopNormal);
    float normalTime;
    cudaEventElapsedTime(&normalTime, startNormal, stopNormal);

    cudaMemcpy(h_output, d_output, NUM_IMAGES * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Normal Kernel Execution Time: " << normalTime << " ms" << std::endl;

    cudaFree(d_ImageData);
    cudaFree(d_output);
    cudaFree(d_taskQueue);
    cudaFree(d_QueueHead);
    cudaFree(d_doneFlag);
    delete[] h_ImageData;
    delete[] h_output;
    delete[] h_taskQueue;

    cudaEventDestroy(startPersistent);
    cudaEventDestroy(stopPersistent);
    cudaEventDestroy(startNormal);
    cudaEventDestroy(stopNormal);

    return 0;
}