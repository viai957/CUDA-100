#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1<<20 // number of elements to sort
#define BITS 20 // number of bits in the numbers
#define THREADS 256 // threads per block

//------------------------------------------------------------------------------
// CPU: odd–even transposition sort (parallelized with OpenMP)
//------------------------------------------------------------------------------
void odd_even_sort_cpu(float *A, int N){
    for (int phase = 0; phase < n; phase++){
        if (phase % 2 == 0){
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n/2; i++){
                int idx = 2*i;
                if (A[idx] > A[idx+1]){
                    float t = A[idx];
                    A[idx] = A[idx+1];
                    A[idx+1] = t;
                }
            }
        } else {
            #pragma opm parallel for schedule(static)
            for (int i = 0; i < (n-1)/2; i++){
                int idx = 2*i+1;
                if (A[idx] > A[idx+1]){
                    float t = A[idx];
                    A[idx] = A[idx+1];
                    A[idx+1] = t;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// GPU kernel: one odd–even phase. Uses blockIdx.x and threadIdx.x
// to distribute the comparisons across all blocks & threads.
//------------------------------------------------------------------------------
__global__ void odd_even_phase(float *d_A, int n, int phase){
    // each thread handles one comparison
    int grid = blockIdx.x * blockDim.x + threadIdx.x;
    int maxPairs = (n + 1) / 2;
    if (grid >= maxPairs) return;

    // compute which pair to compare this phase
    int idx = ((phase & 1) == 0) ? 2*grid : 2*grid + 1;
    int maxPairs = (n + 1) / 2;
    if (grid >= maxPairs) return;

    // compute which pair to compare this phase
    int idx = ((phase & 1) == 0) ? 2*grid : 2*grid + 1;
    if (idx + 1 < n){
        float a = d_A[idx], b = d_A[idx+1];
        if (a > b){
            d_A[idx] = b;
            d_A[idx+1] = a;
        }
    }
}

//------------------------------------------------------------------------------
// Verify CPU vs GPU 
//------------------------------------------------------------------------------
bool verify(const float *cpu, cons float *gpu, int n){
    for (int i = 0; i < n; i++){
        if (fabs(cpu[i] - gpu[i]) > 1e-9f){
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// Host Main function
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// Host main()
//------------------------------------------------------------------------------
int main() {
    // 1) Allocate & init host arrays
    float *h_in  = (float*)malloc(N * sizeof(float));
    float *h_cpu = (float*)malloc(N * sizeof(float));
    float *h_gpu = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)rand() / RAND_MAX;
        h_cpu[i] = h_gpu[i] = h_in[i];
    }

    // 2) CPU timing
    double t0 = omp_get_wtime();
    odd_even_sort_cpu(h_cpu, N);
    double t1 = omp_get_wtime();
    printf("CPU time: %.3f ms\n", (t1 - t0) * 1e3);

    // 3) Copy to GPU
    float *dA;
    cudaMalloc(&dA, N * sizeof(float));
    cudaMemcpy(dA, h_gpu, N * sizeof(float), cudaMemcpyHostToDevice);

    // Determine grid & block dimensions
    int maxPairs = (N + 1) / 2;
    dim3 blockDim(THREADS, 1, 1);
    dim3 gridDim((maxPairs + THREADS - 1) / THREADS, 1, 1);
    printf("Launching %d blocks of %d threads  (total threads: %d)\n",
           gridDim.x, blockDim.x, gridDim.x * blockDim.x);

    // 4) GPU timing
    cudaDeviceSynchronize();
    double t2 = omp_get_wtime();
    for (int phase = 0; phase < N; ++phase) {
        odd_even_phase<<<gridDim, blockDim>>>(dA, N, phase);
    }
    cudaDeviceSynchronize();
    double t3 = omp_get_wtime();
    printf("GPU time: %.3f ms\n", (t3 - t2) * 1e3);

    // 5) Copy back & verify
    cudaMemcpy(h_gpu, dA, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);

    bool ok = verify(h_cpu, h_gpu, N);
    printf("Verification: %s\n", ok ? "PASS" : "FAIL");

    // 6) Cleanup
    free(h_in);
    free(h_cpu);
    free(h_gpu);
    return ok ? 0 : 1;
}
