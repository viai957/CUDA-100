#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdlib>

# define kWarpSize 32

/* -------------------------------------------------------*/
#define CK(call) {
    cudaError_t err = call;
    if (err != cudaSuccess){
        fprinf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    }
}

/* Helpers */
template<typename T> __device__ __forceinline__ T Div(T a, T b) { return a/b;}
template<typename T> __device__ __forceinline__ T Rsqrt(T x) { return rsqrtf(x);}

/* PackType: vectorised storage aligned to sizeof(T)*N  
Combine a single scalar into running stats*/
template<typename T, int N>
struct alignas(sizeof(T) * N) PackType { T elem[N]; };

/* Two Welford "combinbe" overloads */
template<typename T>
__device__ __forceinline__ void WelfordCombine(T x, T* mean, T* m2, T* count)
{
    if (*count == 0) {
        *mean = x;
        *m2 = static_cast<T>(0);
        *count = static_cast<T>(1);
        return;
    }
    T delta = x - mean;
    (*count) += static_cast<T>(1);
    T delta_n = delta / (*count);
    *mean += delta_n;
    *m2 += delta * (x - mean);
}

/* (B) Combine two Welford accumulators (b ⊕ a → a) */
template<typename T>
__device__ __forceinline__ void WelfordCombine(T b_mean, T b_m2, T b_count, T* a_mean, T* a_m2, T* a_count)
{
    if (b_count == 0) return;
    if (*a_count == 0){
        *a_mean = b_mean;
        *a_m2 = b_m2;
        *a_count = b_count;
        return;
    }
    T delta = b_mean - *a_mean;
    T tot_count = *a_count + b_count;
    *a_mean += delta * b_count / tot_count;
    *a_m2 += b_m2 + delta * delta * (*a_count * b_count) / tot_count;
}

/* Warp-level reduction helpers (WelfordWarpReduce) */
/*
 * WelfordWarpReduce uses warp shuffle instructions (__shfl_down_sync)
 * to efficiently combine Welford accumulators across threads within a warp.
 *
 * Why we use __shfl instead of shared memory:
 * - __shfl_* is hardware-accelerated and operates directly on registers
 * - It avoids memory access, synchronization, and arbitration latency
 * - Intra-warp threads execute in lockstep, so no explicit __syncthreads() needed
 * - Latency is ~1–2 cycles vs 10–30+ cycles for shared memory
 *
 * Shared memory would only be used if we needed communication across warps,
 * or required inter-thread buffering. For purely warp-local reduction,
 * warp shuffles are optimal.
 */
template<typename T, int thread_group_width = kWarpSize>
__device__ __forceinline__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count)
{
    *mean = thread_mean;
    *m2 = thread_m2;
    *count = thread_count;

    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1){
        T b_mean = __shft_down_sync(0xffffffff, *mean, mask);
        T b_m2   = __shft_down_sync(0xffffffff, *m2, mask);
        T b_count = __shft_down_sync(0xffffffff, *count, mask);
        WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
    }
}

template<typename T, int thread_group_width = kWarpSize>
__device__ __forceinline__ void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count)
{
    WelfordWarpReaduce<T, thread_group_width>(thread_mean, thread_m2, thread_count, mean, m2, count);
    *mean = __shft_sync(0xffffffff, *mean, 0, thread_group_width);
    *m2 = __shft_sync(0xffffffff, *m2, 0, thread_group_width);
    *count = __shft_sync(0xffffffff, *count, 0, thread_group_width);
}

/* ------------------------------------------------------------------ */
/* Accessor functor for global memory load/store                      */
/* ------------------------------------------------------------------ */
template<typename T>
struct GlobalMem {
    const T* src;
    T* dst;
    int64_t cols;

    template<int N>
    __device__ __forceinline__ void load(T* out, int64_t row, int64_t col) const {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = src[row * cols + col + i];
        }
    
    template<int N>
    __device__ __forceinline__ void store(const T* in, int64_t row, int64_t cols) const {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            dst[row * cols + col + i] = in[i];
        }
    }
    }
};


/* ------------------------------------------------------------------ */
/* LayerNorm kernel  */
/* ------------------------------------------------------------------ */
template<typename LOAD, typename STORE, typename ComputeType,
        int pack_size, int cols_per_thread,
        int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store,
                                  int64_t rows, int64_t cols,
                                  double epsilon,
                                  ComputeType* mean, ComputeType* inv_variance)
{
    static_assert(cols_per_thread % pack_size == 0, "");
    static_assert(thread_group_width <= kWarpSize, "");
    static_assert(kWarpSize % thread_group_width == 0, "");

    constexpr int num_packs = cols_per_thread / pack_size;
    ComputeType buf[rows_per_access][cols_per_thread];

    int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t num_global_thread_group = gridDim.x * blockDim.y;
    int lane_id = threadIdx.x;

    for (int64_t row = )
}
