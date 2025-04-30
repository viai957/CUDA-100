#include <iostream>
#include <cstdlib>
#include <cassert>

// Custom structure for holding computation dimensions
struct ComputeDimensions {
    size_t matrix_rows;
    size_t matrix_cols;
    size_t vector_size;
    
    ComputeDimensions(size_t n) : 
        matrix_rows(n), matrix_cols(n), vector_size(n) {}
        
    bool validate() const {
        return (matrix_cols == vector_size) && (matrix_rows > 0);
    }
};

// Custom error handling wrapper
inline void validateCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d - %s\n", 
                file, line, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_VALIDATE(x) validateCuda(x, __FILE__, __LINE__)

// Memory management wrapper class
template<typename T>
class DeviceMemoryManager {
private:
    T* ptr_;
    size_t size_;

public:
    DeviceMemoryManager(size_t count) : size_(count * sizeof(T)) {
        CUDA_VALIDATE(cudaMalloc(&ptr_, size_));
    }
    
    ~DeviceMemoryManager() {
        if (ptr_) CUDA_VALIDATE(cudaFree(ptr_));
    }
    
    void copyFromHost(const T* host_data) {
        CUDA_VALIDATE(cudaMemcpy(ptr_, host_data, size_, 
                                cudaMemcpyHostToDevice));
    }
    
    void copyToHost(T* host_data) {
        CUDA_VALIDATE(cudaMemcpy(host_data, ptr_, size_, 
                                cudaMemcpyDeviceToHost));
    }
    
    T* get() { return ptr_; }
};

// Enhanced kernel with better memory access patterns
__global__ void enhancedMatrixVectorProduct(
    const float* __restrict__ matrix,
    const float* __restrict__ vector,
    float* __restrict__ result,
    const int dimension
) {
    // Shared memory for vector elements
    extern __shared__ float shared_vector[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Collaborative loading of vector elements into shared memory
    for (int i = tid; i < dimension; i += blockDim.x) {
        shared_vector[i] = vector[i];
    }
    __syncthreads();
    
    if (gid < dimension) {
        float dot_product = 0.0f;
        const float* row = matrix + gid * dimension;
        
        // Unrolled loop for better instruction-level parallelism
        #pragma unroll 4
        for (int j = 0; j < dimension; j++) {
            dot_product += row[j] * shared_vector[j];
        }
        result[gid] = dot_product;
    }
}

class MatrixVectorMultiplier {
private:
    ComputeDimensions dims_;
    static constexpr int THREADS_PER_BLOCK = 256;
    
public:
    MatrixVectorMultiplier(size_t n) : dims_(n) {
        assert(dims_.validate() && "Invalid dimensions specified");
    }
    
    void initialize(float* matrix, float* vector) {
        for (size_t i = 0; i < dims_.matrix_rows; ++i) {
            for (size_t j = 0; j < dims_.matrix_cols; ++j) {
                matrix[i * dims_.matrix_cols + j] = 1.0f;
            }
            vector[i] = 2.0f;
        }
    }
    
    void compute(const float* matrix, const float* vector, float* result) {
        // Create device memory managers
        DeviceMemoryManager<float> d_matrix(dims_.matrix_rows * dims_.matrix_cols);
        DeviceMemoryManager<float> d_vector(dims_.vector_size);
        DeviceMemoryManager<float> d_result(dims_.matrix_rows);
        
        // Transfer data to device
        d_matrix.copyFromHost(matrix);
        d_vector.copyFromHost(vector);
        
        // Calculate grid dimensions
        const int grid_size = (dims_.matrix_rows + THREADS_PER_BLOCK - 1) 
                             / THREADS_PER_BLOCK;
        
        // Launch kernel with shared memory
        enhancedMatrixVectorProduct<<<grid_size, THREADS_PER_BLOCK, 
                                    dims_.vector_size * sizeof(float)>>>
            (d_matrix.get(), d_vector.get(), d_result.get(), dims_.matrix_rows);
        
        CUDA_VALIDATE(cudaDeviceSynchronize());
        
        // Retrieve results
        d_result.copyToHost(result);
    }
    
    void displayResults(const float* matrix, const float* vector, 
                       const float* result) const {
        std::cout << "\nMatrix A:\n";
        for (size_t i = 0; i < dims_.matrix_rows; ++i) {
            for (size_t j = 0; j < dims_.matrix_cols; ++j) {
                std::cout << matrix[i * dims_.matrix_cols + j] << " ";
            }
            std::cout << "\n";
        }
        
        std::cout << "\nVector B:\n";
        for (size_t i = 0; i < dims_.vector_size; ++i) {
            std::cout << vector[i] << " ";
        }
        
        std::cout << "\nResult C:\n";
        for (size_t i = 0; i < dims_.matrix_rows; ++i) {
            std::cout << result[i] << " ";
        }
        std::cout << "\n";
    }
};

int main() {
    constexpr size_t MATRIX_SIZE = 10;
    
    // Host memory allocation
    std::vector<float> matrix(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> vector(MATRIX_SIZE);
    std::vector<float> result(MATRIX_SIZE, 0.0f);
    
    // Create multiplier object
    MatrixVectorMultiplier multiplier(MATRIX_SIZE);
    
    // Initialize data
    multiplier.initialize(matrix.data(), vector.data());
    
    // Perform computation
    multiplier.compute(matrix.data(), vector.data(), result.data());
    
    // Display results
    multiplier.displayResults(matrix.data(), vector.data(), result.data());
    
    return 0;
}