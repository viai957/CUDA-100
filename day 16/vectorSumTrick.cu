#include <iostream.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void sum(const float *v_p, float *out, const int n){
    extern __shared__ float sdata[]; // shared memory for partial sums
    int ti = threadIdx.x; // thread index in the block
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global index in the input array

    // Load data into shared memory
    shared_memory[ti] = (i < n) ? v_p[i] : 0.0f;
    __syncthreads(); // synchronize threads in the block

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (ti < s) {
            shared_memory[ti] += shared_memory[ti + s];
        }
        __syncthreads(); // synchronize threads in the block
    }

    // Write the result for this block to global memory
    if (ti == 0) {
        out[blockIdx.x] = shared_memory[0];
    }
}

void chunkedPartialSums(const float *array, const int size){ 
    const int ThreadsPerBlock = 1024; // 1024 threads per block
    const int Chunks = (size + ThreadsPerBlock - 1) / ThreadsPerBlock; // Calculate number of chunks

    dim3 BlockDim(ThreadsPerBlock); // Set the block size to 256
    dim3 GridDim(Chunks); // Set the grid size to the number of chunks

    size_t SharedMemory = ThreadsPerBlock * sizeof(float); // Allocate shared memory for each block
    size_t SizeBytes = size * sizeof(float); // Calculate the size in bytes
    size_t SizePartial = Chunks * sizeof(float); // Calculate the size of the partial sums array

    float *DeviceInput, *DevicePartial, *HostPartial;
    cudaMalloc((void**)&DeviceInput, SizeBytes); // Allocate device memory for input array
    cudaMalloc((void**)&DevicePartial, SizePartial); // Allocate device memory for partial sums array
    cudaMemcpy(DeviceInput, array, SizeBytes, cudaMemcpyHostToDevice); // Copy input array to device
    cudaMemset(DevicePartial, 0, SizePartial); // Initialize partial sums array to zero
    sum<<<GridDim, BlockDim, SharedMemory>>>(DeviceInput, DevicePartial, size); // Launch kernel
    cudaDeviceSynchronize(); // Wait for kernel to finish
    HostPartial = (float*)malloc(SizePartial); // Allocate host memory for partial sums
    cudaMemcpy(HostPartial, DevicePartial, SizePartial, cudaMemcpyDeviceToHost); // Copy partial sums to host
    
    cudaError_t err = cudaGetLastError(); // Check for errors
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    // Calculate the final sum on the host
    float *PartialSum = (float*)malloc(Chunks * sizeof(float)); // Allocate host memory for final sum
    float FinalSum = 0.0f; // Initialize final sum to zero
    for (int i = 0; i < Chunks; i++) {
        FinalSum += HostPartial[i]; // Sum the partial sums
    }  
    std::cout << "Final sum: " << FinalSum << std::endl; // Print the final sum
    free(HostPartial); // Free host memory for partial sums
    cudaFree(DeviceInput); // Free device memory for input array
    cudaFree(DevicePartial); // Free device memory for partial sums
    free(PartialSum); // Free host memory for final sum
    cudaDeviceReset(); // Reset the device
    std::cout << "Device reset." << std::endl; // Print device reset message
    std::cout << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl; // Print any CUDA errors   
}

int main(int argc, char *argv[]) {
    int size = 1024 * 1024; // Size of the input array
    float *array = (float*)malloc(size * sizeof(float)); // Allocate host memory for input array

    // Initialize the input array with random values
    for (int i = 0; i < size; i++) {
        array[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now(); // Start timer
    chunkedPartialSums(array, size); // Call the function to calculate partial sums
    auto end = std::chrono::high_resolution_clock::now(); // End timer

    std::chrono::duration<double> elapsed = end - start; // Calculate elapsed time
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl; // Print elapsed time

    free(array); // Free host memory for input array
    return 0; // Return success
}