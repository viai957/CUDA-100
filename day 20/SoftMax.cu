#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

#define MAX_THREADS 1024

__global__ void SoftMaxNaive(float *input, float *output, int size){
    int numThreads = blockDim.x;
    int numElementsPerThread = size / numThreads;
    int threadIndex = threadIdx.x;
    int startIndex = threadIndex * numElementsPerThread;
    int endIndex = min(startIndex + numElementsPerThread, size);

    float MaxValue = -FLT_MAX;
    for (int i = startIndex; i < endIndex; i++){
        if (input[i] > MaxValue){
            MaxValue = input[i];
        }
    }
    // Optionally, reduce MaxValue across threads (not done in naive)
    float sumExp = 0.0f;
    for (int i = startIndex; i < endIndex; i++){
        sumExp += expf(input[i] - MaxValue);
    }
    for (int i = startIndex; i < endIndex; i++){
        output[i] = expf(input[i] - MaxValue) / sumExp;
    }
}

__global__ void SoftMaxShared(float *input, float *output, int size){
    int numThreads = blockDim.x;
    int numElementsPerThread = (size + numThreads - 1) / numThreads;
    int threadIndex = threadIdx.x;
    int startIndex = threadIndex * numElementsPerThread;
    int endIndex = min(startIndex + numElementsPerThread, size);

    // Calculate the Maximum value in the input array (per thread)
    extern __shared__ float shared[];
    float* SharedMaxValue = shared;
    float* sharedSumExp = &shared[numThreads];

    float MaxValue = -FLT_MAX;
    for (int i = startIndex; i < endIndex; i++){
        if (input[i] > MaxValue){
            MaxValue = input[i];
        }
    }
    SharedMaxValue[threadIndex] = MaxValue;
    __syncthreads();
    // Reduce max across threads
    if (threadIndex == 0) {
        float blockMax = SharedMaxValue[0];
        for (int i = 1; i < numThreads; i++) {
            if (SharedMaxValue[i] > blockMax) blockMax = SharedMaxValue[i];
        }
        SharedMaxValue[0] = blockMax;
    }
    __syncthreads();
    MaxValue = SharedMaxValue[0];

    float sumExp = 0.0f;
    for (int i = startIndex; i < endIndex; i++){
        sumExp += expf(input[i] - MaxValue);
    }
    sharedSumExp[threadIndex] = sumExp;
    __syncthreads();
    // Reduce sumExp across threads
    if (threadIndex == 0) {
        float blockSum = 0.0f;
        for (int i = 0; i < numThreads; i++) {
            blockSum += sharedSumExp[i];
        }
        sharedSumExp[0] = blockSum;
    }
    __syncthreads();
    sumExp = sharedSumExp[0];
    for (int i = startIndex; i < endIndex; i++){
        output[i] = expf(input[i] - MaxValue) / sumExp;
    }
}

int main() {
    // Minimal main function to allow linking. Add kernel launch/test code here if needed.
    return 0;
}