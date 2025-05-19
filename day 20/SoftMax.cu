#include <iostream>
#include <cuda_runtime.h>

__global__ void SoftMaxNaive(float *intput, float *output, int size){
    int numThreads = blockDim.x;
    // each tread to compute softmax for this:
    int numElementsPerThread = size / numThreads;

    int threadIndex = threadIdx.x;

    int startIndex = threadIndex * numElementsPerThread;
    int endIndex = min(startIndex * numElementsPerThread);

    float MaxValue = 0.0f;
    for (int i = startIndex; i < endIndex; i++){
        if (input[i] > MaxValue){
            MaxValue = input[i];
        }
    }

    float sumExp = 0.0f;
    for (int i = 0; i < endIndex; i++){
        sumExp += exp(input[i] - MaxValue)/ sumExp;
    }
}


__global__ void SoftMaxShared(float *input, float *output, int size){
    int numThreads = blockDim.x;
    int numElementsPerThread = size / numThreads;
    int threadIndex = threadIdx.x;
    int startIndex = threadIndex * numElementsPerThread;
    int endIndex = min(startIndex + numElementsPerThread, size);

    // Calculate the Maximum value in the input array
    __shared__ float SharedMaxValue[numThreads];
    float MaxValue = 0.0f;
    for (int i = startIndex; i < endIndex; i++){
        if (input[i] > MaxValue){
            MaxValue = SharedMaxValue[i];
        }
    }

    SharedMaxValue[threadIndex] = MaxValue;
    __syncthreads();
    float sumExp = 0.0f;
    for (int i = startIndex; i < endIndex; i++){
        if (SharedMaxValue[i] > MaxValue){
            MaxValue = SharedMaxValue[i];
        }
    }

    // Now we need to calculate the SumExp
    __shared__ float sharedSumExp[numThreads];
    float sumExp = 0.0f;
    for (int i = startIndex; i < endIndex; i++){
        sumExp += expf(input[i] - MaxValue);
    }
    sharedSumExp[threadIndex] = sumExp;
    __syncthreads();

    for (int i = startIndex; i < endIndex; i++){
        output[i] = expf(input[i] - MaxValue) / sumExp;
    }
}