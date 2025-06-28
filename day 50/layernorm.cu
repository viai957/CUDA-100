/*******************************************************************/
/*  layer_norm_welford.cu                                          */
/*  Numerically-stable LayerNorm (rows×cols tensor) using          */
/*  warp-level Welford reduction and vectorised memory accesses.   */
/*******************************************************************/
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#define kWarpSize 32

/* ------------------------------------------------------------------ */
/* Simple error-check wrapper                                         */
/* ------------------------------------------------------------------ */
#define CK(call)                                                                 \
  {                                                                              \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
      fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  }

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
template<typename T> __device__ __forceinline__ T Div(T a, T b) { return a / b; }
template<typename T> __device__ __forceinline__ T Rsqrt(T x)      { return rsqrtf(x); }

/* ------------------------------------------------------------------ */
/* PackType: vectorised storage aligned to sizeof(T)*N                */
/* ------------------------------------------------------------------ */
template<typename T, int N>
struct alignas(sizeof(T) * N) PackType { T elem[N]; };

/* ------------------------------------------------------------------ */
/* Two Welford “combine” overloads                                    */
/* ------------------------------------------------------------------ */
/*
 * Welford's Online Algorithm (intuitively):
 * 
 * Let’s say:
 * - n is the updated count.
 * - μₙ is the new mean.
 * - μₙ₋₁ is the previous mean.
 * - x is the new data point.
 * 
 * Then the mean update:
 *   μₙ = μₙ₋₁ + (x - μₙ₋₁) / n
 * 
 * Then the M2 (sum of squared deviations) update:
 *   M2ₙ = M2ₙ₋₁ + (x - μₙ₋₁)(x - μₙ)
 */
/* (A) Combine a single scalar into running stats */
template<typename T>
__device__ __forceinline__ void WelfordCombine(T x, T* mean, T* m2, T* count)
{
  if (*count == 0) {
    *mean = x;
    *m2   = static_cast<T>(0);
    *count = static_cast<T>(1);
    return;
  }
  T delta   = x - *mean;
  (*count) += static_cast<T>(1);
  T delta_n = delta / (*count);
  *mean    += delta_n;
  *m2      += delta * (x - *mean);
}

/* (B) Combine two Welford accumulators (b ⊕ a → a) */
template<typename T>
__device__ __forceinline__ void WelfordCombine(T b_mean, T b_m2, T b_count,
                                               T* a_mean, T* a_m2, T* a_count)
{
  if (b_count == 0) return;
  if (*a_count == 0) {               // a is empty → just copy b
    *a_mean  = b_mean;
    *a_m2    = b_m2;
    *a_count = b_count;
    return;
  }
  T delta        = b_mean - *a_mean;
  T tot_count    = *a_count + b_count;
  *a_mean       += delta * b_count / tot_count;
  *a_m2         += b_m2 + delta * delta * (*a_count * b_count) / tot_count;
  *a_count       = tot_count;
}

/* ------------------------------------------------------------------ */
/* Warp-level reduction helpers                                       */
/* ------------------------------------------------------------------ */
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
__device__ __forceinline__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count,
                                                  T* mean, T* m2, T* count)
{
  *mean  = thread_mean;
  *m2    = thread_m2;
  *count = thread_count;

  for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
    T b_mean  = __shfl_down_sync(0xffffffff, *mean,  mask);
    T b_m2    = __shfl_down_sync(0xffffffff, *m2,    mask);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<typename T, int thread_group_width = kWarpSize>
__device__ __forceinline__ void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count,
                                                     T* mean, T* m2, T* count)
{
  WelfordWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_count,
                                           mean, m2, count);
  *mean  = __shfl_sync(0xffffffff, *mean,  0, thread_group_width);
  *m2    = __shfl_sync(0xffffffff, *m2,    0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

/* ------------------------------------------------------------------ */
/* Accessor functor for global memory load/store                      */
/* ------------------------------------------------------------------ */
template<typename T>
struct GlobalMem {
  const T* src;
  T*       dst;
  int64_t  cols;        // leading dimension

  template<int N>
  __device__ __forceinline__ void load(T* out, int64_t row, int64_t col) const {
#pragma unroll
    for (int i = 0; i < N; ++i) { out[i] = src[row * cols + col + i]; }
  }
  template<int N>
  __device__ __forceinline__ void store(const T* in, int64_t row, int64_t col) const {
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[row * cols + col + i] = in[i]; }
  }
};

/* ------------------------------------------------------------------ */
/* LayerNorm kernel */
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
  static_assert(thread_group_width <= kWarpSize,   "");
  static_assert(kWarpSize % thread_group_width == 0, "");

  constexpr int num_packs = cols_per_thread / pack_size;
  ComputeType buf[rows_per_access][cols_per_thread];

  int64_t global_thread_group_id  = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t num_global_thread_group = gridDim.x * blockDim.y;
  int     lane_id                 = threadIdx.x;

  for (int64_t row = global_thread_group_id * rows_per_access;
       row < rows;
       row += num_global_thread_group * rows_per_access) {

    /* ---- per-thread Welford accumulators -------------------------- */
    ComputeType t_mean[rows_per_access]  = {0};
    ComputeType t_m2[rows_per_access]    = {0};
    ComputeType t_count[rows_per_access] = {0};

#pragma unroll
    for (int r = 0; r < rows_per_access; ++r) {
      ComputeType* row_buf = buf[r];

#pragma unroll
      for (int p = 0; p < num_packs; ++p) {
        int col         = (p * thread_group_width + lane_id) * pack_size;
        int pack_offset = p * pack_size;

        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_offset, row + r, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            WelfordCombine(row_buf[pack_offset + i],
                           t_mean  + r,
                           t_m2    + r,
                           t_count + r);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) row_buf[pack_offset + i] = 0;
        }
      }
    }

    /* ---- warp reduction ------------------------------------------ */
    ComputeType w_mean[rows_per_access];
    ComputeType w_m2[rows_per_access];
    ComputeType w_count[rows_per_access];

#pragma unroll
    for (int r = 0; r < rows_per_access; ++r) {
      int g_row = row + r;
      ComputeType* row_buf = buf[r];

      WelfordWarpAllReduce<ComputeType, thread_group_width>(
          t_mean[r], t_m2[r], t_count[r],
          w_mean + r, w_m2 + r, w_count + r);

      ComputeType row_mean  = w_mean[r];
      ComputeType variance  = max(Div(w_m2[r], w_count[r]), static_cast<ComputeType>(0));
      ComputeType inv_var   = Rsqrt(variance + static_cast<ComputeType>(epsilon));

      if (lane_id == 0) {
        mean[g_row]         = row_mean;
        inv_variance[g_row] = inv_var;
      }
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = (row_buf[i] - row_mean) * inv_var;
      }
#pragma unroll
      for (int p = 0; p < num_packs; ++p) {
        int col = (p * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols)
          store.template store<pack_size>(row_buf + p * pack_size, g_row, col);
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* Reference CPU LayerNorm                                            */
/* ------------------------------------------------------------------ */
void cpu_layer_norm(const float* x, float* y,
                    float* mean, float* invvar,
                    int rows, int cols, double eps)
{
  for (int r = 0; r < rows; ++r) {
    const float* row = x + r * cols;
    double m = 0;
    for (int c = 0; c < cols; ++c) m += row[c];
    m /= cols;

    double var = 0;
    for (int c = 0; c < cols; ++c) {
      double d = row[c] - m;
      var += d * d;
    }
    var /= cols;
    double invv = 1.0 / std::sqrt(var + eps);

    mean[r]    = static_cast<float>(m);
    invvar[r]  = static_cast<float>(invv);

    float* out = y + r * cols;
    for (int c = 0; c < cols; ++c)
      out[c] = static_cast<float>((row[c] - m) * invv);
  }
}

/* ------------------------------------------------------------------ */
/* main()                                                             */
/* ------------------------------------------------------------------ */
int main()
{
  /* problem shape --------------------------------------------------- */
  constexpr int ROWS = 256;
  constexpr int COLS = 512;
  constexpr float EPS = 1e-5f;

  /* allocate host memory ------------------------------------------- */
  size_t bytes = ROWS * COLS * sizeof(float);
  float *h_x  = (float*)malloc(bytes);
  float *h_y  = (float*)malloc(bytes);
  float *h_ref= (float*)malloc(bytes);
  float *h_mean = (float*)malloc(ROWS * sizeof(float));
  float *h_inv  = (float*)malloc(ROWS * sizeof(float));
  float *h_ref_mean = (float*)malloc(ROWS * sizeof(float));
  float *h_ref_inv  = (float*)malloc(ROWS * sizeof(float));

  /* initialise input ------------------------------------------------ */
  for (int i = 0; i < ROWS * COLS; ++i)
    h_x[i] = static_cast<float>(std::sin(i * 0.1));

  /* allocate device memory ----------------------------------------- */
  float *d_x, *d_y, *d_mean, *d_inv;
  CK(cudaMalloc(&d_x, bytes));
  CK(cudaMalloc(&d_y, bytes));
  CK(cudaMalloc(&d_mean, ROWS * sizeof(float)));
  CK(cudaMalloc(&d_inv,  ROWS * sizeof(float)));

  CK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

  /* kernel launch config ------------------------------------------- */
  using ComputeT = float;
  constexpr int pack_size          = 4;          // vector width
  constexpr int cols_per_thread    = 32;         // must be multiple of pack_size
  constexpr int thread_group_width = kWarpSize;  // one full warp
  constexpr int rows_per_access    = 1;          // one row per warp
  constexpr bool padding           = true;

  const int warps_per_block = 4;   // blockDim.y
  dim3 block(thread_group_width, warps_per_block);
  int64_t thread_groups =
      (ROWS + rows_per_access * warps_per_block - 1) /
      (rows_per_access * warps_per_block);
  dim3 grid(thread_groups);

  GlobalMem<float> accessor{d_x, d_y, COLS};

  LayerNormWarpImpl<GlobalMem<float>, GlobalMem<float>, ComputeT,
                    pack_size, cols_per_thread,
                    thread_group_width, rows_per_access, padding>
      <<<grid, block>>>(accessor, accessor,
                        ROWS, COLS, EPS,
                        d_mean, d_inv);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  /* bring results back --------------------------------------------- */
  CK(cudaMemcpy(h_y,    d_y,    bytes, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_mean, d_mean, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_inv,  d_inv,  ROWS * sizeof(float), cudaMemcpyDeviceToHost));

  /* reference & compare -------------------------------------------- */
  cpu_layer_norm(h_x, h_ref, h_ref_mean, h_ref_inv, ROWS, COLS, EPS);

  double max_diff      = 0.0;
  double max_mean_diff = 0.0;
  double max_inv_diff  = 0.0;

  /* compare element-wise output ------------------------------------ */
  for (int i = 0; i < ROWS * COLS; ++i){
    max_diff = std::max(max_diff, std::abs(h_y[i] - h_ref[i]));
  }

  /* compare per-row mean statistics -------------------------------- */
  for (int i = 0; i < ROWS; i++){
    max_inv_diff = std::max(max_inv_diff, std::abs(h_mean[i] - h_ref_mean[i]));
  }

  printf("✓ LayerNorm results match (max |Δ| = %.3g, mean Δ = %.3g, invvar Δ = %.3g)\n",
         max_diff, max_mean_diff, max_inv_diff);

  /* cleanup -------------------------------------------------------- */
  CK(cudaFree(d_x)); CK(cudaFree(d_y)); CK(cudaFree(d_mean)); CK(cudaFree(d_inv));
  free(h_x); free(h_y); free(h_ref); free(h_mean); free(h_inv); free(h_ref_mean); free(h_ref_inv);
  return 0;
}