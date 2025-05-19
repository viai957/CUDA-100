#include <iostream>
#include <cmath>

__global__ void sm_roll_call(){
    const int threadIndex = threadIdx.x;

    uint streamingMultiprocessorId;
    asm("mov.u32 %0, %%smid;" : "=r"(streamingMultiprocessorId));

    printf("Thread %d on SM %d\n", threadIndex, streamingMultiprocessorId);

}

int main(){
    sm_roll_call<<<4, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}