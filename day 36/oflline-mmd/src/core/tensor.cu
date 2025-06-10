#pragma once
#include <cuda_runtime.h>
#include <cassert>

template<typename T = float>
class DeviceTensor {
public:
    T* data_;
    int64_t size_;
    DeviceTensor(int64_t n) : size_(n) {
        cudaMalloc(&data_, n * sizeof(T));
    }
    ~DeviceTensor() { cudaFree(data_); }
    // Disable copy, allow move:
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;
    DeviceTensor(DeviceTensor&& o) noexcept : data_(o.data_), size_(o.size_) { o.data_ = nullptr; }
};
