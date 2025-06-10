// ============================================================================
// src/core/math.cu
// Low-level CUDA math helpers & fused element-wise kernels for Mirror-Descent RL
// ============================================================================

#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>
#include <float.h>
#include "tensor.hpp"          // DeviceTensor

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t _err = (expr);                                          \
        if (_err != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(_err));     \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(expr)                                                  \
    do {                                                                    \
        cublasStatus_t _err = (expr);                                       \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n",                \
                         __FILE__, __LINE__, int(_err));                    \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// In-register helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fast_relu(float x) { return x > 0.f ? x : 0.f; }

// Warp-wide max + sum reductions (for log-sum-exp/softmax)
template<int THREADS>
__device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = THREADS / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

template<int THREADS>
__device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = THREADS / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// ---------------------------------------------------------------------------
// Kernel 1: add bias (+ optional ReLU)  y_ij += b_i
// ---------------------------------------------------------------------------
template<int TPB=256, bool RELU>
__global__ void bias_relu_kernel(float* __restrict y,
                                 const float* __restrict b,
                                 int out, int batch)
{
    int idx = blockIdx.x * TPB + threadIdx.x;          // flat index (o + i*out)
    int N   = out * batch;
    if (idx >= N) return;

    int o = idx % out;
    float v = y[idx] + b[o];
    if constexpr(RELU) v = fast_relu(v);
    y[idx] = v;
}

inline void launch_bias_relu(float* y,
                             const float* b,
                             int out, int batch,
                             bool relu,
                             cudaStream_t s = 0)
{
    constexpr int TPB = 256;
    int blocks = (out * batch + TPB - 1) / TPB;
    if (relu)
        bias_relu_kernel<TPB, true ><<<blocks, TPB, 0, s>>>(y, b, out, batch);
    else
        bias_relu_kernel<TPB, false><<<blocks, TPB, 0, s>>>(y, b, out, batch);
}

// ---------------------------------------------------------------------------
// Kernel 2: fused dW / db reduction
//   dW = grad_y  @  x^T      (outer product summed over batch)
//   db = sum_batch(grad_y)
// One thread-block reduces one output neuron across the whole batch.
// ---------------------------------------------------------------------------
template<int TPB=256>
__global__ void fused_dw_db_kernel(const float* __restrict grad_y, // [B, O]
                                   const float* __restrict  x,     // [B, I]
                                   float* __restrict dW,           // [O, I]
                                   float* __restrict db,           // [O]
                                   int B, int O, int I)
{
    int o = blockIdx.x;                    // output neuron this block owns
    int tid = threadIdx.x;

    // Shared scratch for partials
    extern __shared__ float sh[];
    float* sh_dw = sh;                     // TPB*I elements (strided write)
    float  sum_db = 0.f;

    // Loop over batch in strides of TPB
    for (int b0 = 0; b0 < B; b0 += TPB) {
        int b = b0 + tid;
        if (b < B) {
            float gy = grad_y[b * O + o];
            sum_db += gy;
            // compute gy * x[b, :]  -> I partials
            #pragma unroll 4
            for (int i = 0; i < I; ++i)
                sh_dw[tid * I + i] = gy * x[b * I + i];
        }
        __syncthreads();

        // Reduce partial outer products across warp
        for (int stride = TPB / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                #pragma unroll 4
                for (int i = 0; i < I; ++i)
                    sh_dw[tid * I + i] += sh_dw[(tid + stride) * I + i];
            }
            __syncthreads();
        }
    }

    // Thread 0 writes results
    if (tid == 0) {
        db[o] = sum_db;
        float* row = dW + o * I;
        #pragma unroll 4
        for (int i = 0; i < I; ++i)
            row[i] += sh_dw[i];            // atomics not needed (one block per o)
    }
}

// Host wrapper
inline void launch_dw_db(const float* grad_y,
                         const float* x,
                         float* dW, float* db,
                         int B, int O, int I,
                         cudaStream_t s = 0)
{
    constexpr int TPB = 256;
    size_t shmem = TPB * I * sizeof(float);
    fused_dw_db_kernel<TPB><<<O, TPB, shmem, s>>>(grad_y, x, dW, db, B, O, I);
}

// ---------------------------------------------------------------------------
// Convenience: softmax + logsumexp (row-wise) for small A (<=1024)
// ---------------------------------------------------------------------------
template<int TPB=32>
__global__ void rowwise_logsoftmax(float* __restrict out,  // in-place logits
                                   float* __restrict lse,  // [B]
                                   int B, int A)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;

    if (b >= B) return;
    float* row = out + b * A;

    // compute max
    float m = -FLT_MAX;
    for (int a = tid; a < A; a += TPB)
        m = fmaxf(m, row[a]);
    m = warp_reduce_max<TPB>(m);

    // compute sum(exp)
    float sum = 0.f;
    for (int a = tid; a < A; a += TPB)
        sum += __expf(row[a] - m);
    sum = warp_reduce_sum<TPB>(sum);
    if (tid == 0) lse[b] = __logf(sum) + m;

    // write softmax in-place
    for (int a = tid; a < A; a += TPB)
        row[a] = __expf(row[a] - m) / sum;
}

// ---------------------------------------------------------------------------
// Small utility for allocating zero-initialised tensors on device
// ---------------------------------------------------------------------------
inline float* device_malloc_zero(size_t n_bytes) {
    float* p;
    CUDA_CHECK(cudaMalloc(&p, n_bytes));
    CUDA_CHECK(cudaMemset(p, 0, n_bytes));
    return p;
}