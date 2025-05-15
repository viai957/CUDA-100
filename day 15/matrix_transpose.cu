#include <stdio.h>
#include <assert.h>

// Check CUDA runtime API results could be used arond any runtime API call
inline 
cudaError_t checkCuda(cudaError_t results)
{
  if (result != cudaSuccess){
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(results));
    assert(result == cudaSuccess);
  }
  return results;
}

const int TILE_DIM = 32; // Tile size for shared memory
const int BLOCK_ROWS = 8; // Number of rows per block in shared memory
const int N = 1024;

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
  bool passed = true;
  for (int i = 0; i < n; i++){
    if (res[i] != ref[i]){
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
    if(passed){
      printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
    }
  }
  // simple copy kernel 
  // Used as reference case representing best effictive bandwidth
  __global__ void copy(float *odata, const float *idata)
  {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j++){
      odata[(y+j) * width + x] = idata[(y + j) * width + x];
    }
  }

  // 



}