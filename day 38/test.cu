#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>

__global__ void array_increment(int *input){
    const int threadIndex = threadIdx.x;
    input[threadIndex] += 1;
}

const int arraySize = 1024;

int *array = (int*)malloc(arraySize * sizeof(int))
for (int i = 0; i < arraySize; i++){
    array[i] = i * 10;
}

int* d_array;
cudaMalloc((void**)&d_array, arraySize * sizeof(int));
cudaMemcpy(d_array, array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

array_increment<<<1, arraySize>>>(d_array);
cudaMemcpy(array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

void printArray(int* array, int arraySize){
    print("[");
    for (int i = 0; i < arraySize; i++){
        printf("%d", array[i]);
        if (i < arraySize - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main() {
    const int arraySize = 1024;

    // Allocate host memory for the input array
    int* array = (int*)malloc(arraySize * sizeof(int));

    // Initialize the input array
    for (int i = 0; i < arraySize; i++){
        array[i] = i * 10;
    }

    printf("Before increment:\n");
    printArray(array, arraySize);

    // Allocate GPU memory for the input array
    int* d_array;
    cudaMalloc((void**)&d_array, arraySize * sizeof(int));

    // Copy the input array from host memoery to GPU memory
    cudaMemcpy(d_array, array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    array_increment<<<1, arraySize>>>(d_array);

    // Copy the result array from GPU memory back to host memory
    cudaMemcpy(array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    printf("After increment:\n");
    printArray(array, arraySize);
    // Free GPU memory
    cudaFree(d_array);
    // Free host memory
    free(array);
    return 0;
}
