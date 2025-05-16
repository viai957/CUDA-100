#include <stdio.h>
#include <assert.h>

// Check CUDA runtime API results could be used arond any runtime API call
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess){
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

const int TILE_DIM = 32; // Tile size for shared memory
const int BLOCK_ROWS = 8; // Number of rows per block in shared memory
const int NUM_REPS = 100; // Number of repetitions for timing

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
    if (passed){
      printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
    }
  }
}


  // simple copy kernel 
  // Used as reference case representing best effictive bandwidth
  // ┌───────────────────────┐
  // │       Full Matrix     │
  // │  ┌─────┬─────┬─────┐  │
  // │  │Tile │Tile │Tile │  │
  // │  │(0,0)│(1,0)│(2,0)│  │
  // │  ├─────┼─────┼─────┤  │
  // │  │Tile │Tile │Tile │  │
  // │  │(0,1)│(1,1)│(2,1)│  │
  // │  ├─────┼─────┼─────┤  │
  // │  │Tile │Tile │Tile │  │
  // │  │(0,2)│(1,2)│(2,2)│  │
  // │  └─────┴─────┴─────┘  │
  // └───────────────────────┘
  //    Tile (32×32)                Thread Block
  // ┌──────────────┐            (32×8 threads)
  // │              │            ┌──────────────┐
  // │  Thread      │            │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ ← Each thread 
  // │  (tx,ty)     │            │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│   handles elements
  // │     ↓        │            │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│   at (tx,ty),
  // │     *        │            │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│   (tx,ty+8),
  // │     *        │            │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│   (tx,ty+16), 
  // │     *        │            │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│   and (tx,ty+24)
  // │     *        │            │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
  // │              │            │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
  // └──────────────┘            └──────────────┘
  // Example with Values
  // For a 1024×1024 matrix with TILE_DIM=32 and BLOCK_ROWS=8:

  // Grid size: 32×32 blocks (1024/32 = 32 in each dimension)
  // Block size: 32×8 threads (to cover the 32×32 tile efficiently)
  // Each thread processes 4 elements (32/8 = 4)
  // Thread (5,3) in Block (2,1) would:

  // Calculate position: x = 2×32 + 5 = 69, y = 1×32 + 3 = 35
  // ┌─────────────────────────────────── 1024 columns ───────────────────────────────────┐
  // │                                                                                     │
  // │                ┌─────────────────────┐                                              │
  // │                │     Block (2,1)     │                                              │
  // │                │  ┌───────────────┐  │                                              │
  // │          row 35│  │ (5,3)         │  │                                              │
  // │                │  │    ↓          │  │                                              │
  // │                │  │    ●          │  │                                              │
  // │                │  │    │          │  │                                              │
  // │                │  │    │          │  │                                              │
  // │                │  │    │          │  │                                              │
  // │                │  │    │          │  │                                              │
  // │                │  │    │          │  │                                              │
  // │                │  │    │          │  │                                              │
  // │                │  │    │          │  │                                              │
  // │          row 66│  │    ●          │  │                                              │
  // │                │  └───────────────┘  │                                              │
  // │                └─────────────────────┘                                              │
  // │                         col 69                                                      │
  // └─────────────────────────────────────────────────────────────────────────────────────┘
  // Copy elements at:
  // (35, 69) → odata[35×1024 + 69] = idata[35×1024 + 69]
  // (43, 69) → odata[43×1024 + 69] = idata[43×1024 + 69]
  // (51, 69) → odata[51×1024 + 69] = idata[51×1024 + 69]
  // (59, 69) → odata[59×1024 + 69] = idata[59×1024 + 69]

  //   Given:

  // x = 2 × 32 + 5 = 69
  // y = 1 × 32 + 3 = 35
  // width = 1024
  // TILE_DIM = 32
  // BLOCK_ROWS = 8
  // Iteration	j value	Matrix Position	Memory Index	Operation
  // 1	0	(35, 69)	35×1024 + 69 = 35,869	odata[35,869] = idata[35,869]
  // 2	8	(43, 69)	43×1024 + 69 = 44,037	odata[44,037] = idata[44,037]
  // 3	16	(51, 69)	51×1024 + 69 = 52,205	odata[52,205] = idata[52,205]
  // 4	24	(59, 69)	59×1024 + 69 = 60,373	odata[60,373] = idata[60,373]

  __global__ void copy(float *odata, const float *idata)
  {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      odata[(y+j)*width + x] = idata[(y+j)*width + x];
    }
  }

  // copy kernel using shared memory
  __global__ void copySharedMem(float *odata, const float *idata)
  {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
    }
    
    __syncthreads();
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      odata[(y+j)*width + x] = tile[threadIdx.y+j][threadIdx.x];
    }
  }

  // naive transpose kernel 
  // Simple transpose kernel; dosn't use shared memory
  // Global memory access pattern is not coalesced but writes are not
  // x = 2 × 32 + 5 = 69 (column in input)
  // y = 1 × 32 + 3 = 35 (row in input)
  // width = 1024
  // TILE_DIM = 32
  // BLOCK_ROWS = 8
  // Iteration Table
  // Iteration	j value	Input Position	Input Memory	Output Position	Output Memory	Operation
  // 1	0	(35, 69)	(35×1024)+69 = 35,869	(69, 35)	(69×1024)+35 = 70,675	odata[70,675] = idata[35,869]
  // 2	8	(43, 69)	(43×1024)+69 = 44,037	(69, 43)	(69×1024)+43 = 70,683	odata[70,683] = idata[44,037]
  // 3	16	(51, 69)	(51×1024)+69 = 52,205	(69, 51)	(69×1024)+51 = 70,691	odata[70,691] = idata[52,205]
  // 4	24	(59, 69)	(59×1024)+69 = 60,373	(69, 59)	(69×1024)+59 = 70,699	odata[70,699] = idata[60,373]
  // 4	24	(59, 69)	(59×1024)+69 = 60,373	(69, 59)	(69×1024)+59 = 70,699	odata[70,699] = idata[60,373]
  __global__ void transposeNaive(float *odata, const float *idata)
  {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
      odata[x*width + (y+j)] = idata[(y+j)*width + x];
    }
  }

  // coalesced transpose kernel
  // Using shared memory to achive coalesing in both reads and writes
  // Tile width == # blanks causes shared memory blank conflict 
  // This kernel uses shared memory to dramatically improve the performance of matrix transpose by ensuring coalesced memory access patterns. 
  // Let me walk through exactly what happens for Thread (5,3) in Block (2,1).
  // The kernel:
  // 1. Loads a 32×32 tile from input into shared memory with coalesced reads
  // 2. Performs the transpose operation through clever indexing in shared memory
  // 3. Writes back to global memory with coalesced writes
  // Iteration	j value	Input Position	Input Address	Shared Memory Location
  // 1	        0	(35, 69)	 35×1024+69 = 35,869	tile[3][5]
  // 2	        8	(43, 69)	 43×1024+69 = 44,037	tile[11][5]
  // 3	       16	(51, 69)	 51×1024+69 = 52,205	tile[19][5]
  // 4	       24	(59, 69)	 59×1024+69 = 60,373	tile[27][5]
  // Second Phase: Writing Transposed Data
  // Iteration	j value	Shared Memory Read	Output Position	Output Address
  // 1	0	tile[5][3]	(67, 37)	67×1024+37 = 68,645
  // 2	8	tile[5][11]	(75, 37)	75×1024+37 = 76,837
  // 3	16	tile[5][19]	(83, 37)	83×1024+37 = 85,029
  // 4	24	tile[5][27]	(91, 37)	91×1024+37 = 93,221
  //   Input Matrix                          Shared Memory                         Output Matrix
  // ┌───────────────────┐                ┌───────────────────┐                ┌───────────────────┐
  // │                   │                │       [0,5]       │                │                   │
  // │      col 69       │                │        ...        │                │      col 37       │
  // │        ↓          │                │       [3,5]       │                │        ↓          │
  // │ row 35→ ●─────────┼─────Read─────→│        ...        │                │ row 67→ ●─────────┼─────┐
  // │        │          │                │      [11,5]       │                │        │          │     │
  // │        │          │                │        ...        │                │        │          │     │
  // │ row 43→ ●─────────┼─────Read─────→│      [19,5]       │   ┌─Transposed─┼────────┘          │     │
  // │        │          │                │        ...        │   │            │                   │     │
  // │        │          │                │      [27,5]       │   │            │ row 75→ ●─────────┼─────┐
  // │ row 51→ ●─────────┼─────Read─────→│                   │   │            │        │          │     │
  // │        │          │           ┌────┼─────[5,3]────────┼───┘            │        │          │     │
  // │        │          │           │    │     [5,11]       │                │ row 83→ ●─────────┼─────┐
  // │ row 59→ ●─────────┼─────Read──┼───→│     [5,19]       │                │        │          │     │
  // │                   │           │    │     [5,27]       │                │        │          │     │
  // └───────────────────┘           │    └───────────────────┘                │ row 91→ ●─────────┼─────┘
  //                                 └───────────────────────────Write─────────┘                   │

// Overall Strategy
// The kernel:
// Loads a 32×32 tile from input into shared memory with coalesced reads
// Performs the transpose operation through clever indexing in shared memory
// Writes back to global memory with coalesced writes
  __global__ void transposeCoalesced(float *odata, const float *idata)
  {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];  
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y+j];
    }
  }

  // No bank-conflict transpose kernel
  // Same as transposeCoalesced but with first tile dimention is padded
  // to avoid bank conflicts
  // This kernel solves a critical performance problem in transposeCoalesced 
  // by adding padding to the shared memory array. 
  // Let me explain how it works for Thread (5,3) in Block (2,1).
  // Phase 1: Loading Data (Same as coalesced version)
  // Iteration	Input     Position	    Input Address	  Shared Memory
  // 1	        (35, 69)	35×1024+69 =  35,869	        tile[3][5]
  // 2	        (43, 69)	43×1024+69 =  44,037	        tile[11][5]
  // 3	        (51, 69)	51×1024+69 =  52,205	        tile[19][5]
  // 4	        (59, 69)	59×1024+69 =  60,373	        tile[27][5]
  // Phase 2: Writing Transposed Data
  // Iteration	Read from Shared Memory	  Output Position	       Output Address
  // 1	        tile[5][3]	(67, 37)	    67×1024+37           = 68,645
  // 2	        tile[5][11]	(75, 37)	    75×1024+37           = 76,837
  // 3	        tile[5][19]	(83, 37)	    83×1024+37           = 85,029
  // 4	        tile[5][27]	(91, 37)	    91×1024+37           = 93,221
  //        Shared Memory Without Padding           Shared Memory With Padding
  // Why Bank Conflicts Occur Without Padding
  // Shared memory is divided into 32 banks. When multiple threads access the same bank simultaneously, a conflict occurs that serializes access.
  // Without padding:
  // Thread (5,3) reads from addresses: 5×32+3, 5×32+11, 5×32+19, 5×32+27
  // Thread (6,3) reads from addresses: 6×32+3, 6×32+11, 6×32+19, 6×32+27
  // These map to banks: 3, 11, 19, 27 (for both threads)
  // Result: Bank conflicts! Multiple threads hit the same banks
  // How Padding Eliminates Bank Conflicts
  // With TILE_DIM+1 padding (33 instead of 32):

  // Thread (5,3) reads from addresses: 5×33+3, 5×33+11, 5×33+19, 5×33+27
  // These map to banks: 8, 16, 24, 0
  // Thread (6,3) reads from addresses: 6×33+3, 6×33+11, 6×33+19, 6×33+27
  // These map to banks: 9, 17, 25, 1
  // Since 33 and 32 are relatively prime, the +1 padding ensures threads access different banks, eliminating conflicts.
  //        (Bank conflicts occur)                  (No bank conflicts)
  // ┌────────────────────────────────────┐     ┌─────────────────────────────────┐
  // │                                    │     │                                 │
  // │  Thread 5,3 → Bank 3               │     │  Thread 5,3 → Bank 8            │
  // │  Thread 6,3 → Bank 3  <-- CONFLICT │     │  Thread 6,3 → Bank 9            │
  // │  Thread 7,3 → Bank 3  <--          │     │  Thread 7,3 → Bank 10           │
  // │  ...                               │     │  ...                            │
  // │  Thread 5,11 → Bank 11             │     │  Thread 5,11 → Bank 16          │
  // │  Thread 6,11 → Bank 11 <-- CONFLICT│     │  Thread 6,11 → Bank 17          │
  // │  ...                               │     │  ...                            │
  // └────────────────────────────────────┘     └─────────────────────────────────┘
  __global__ void transposeNoBankConflicts(float *odata, const float *idata)
  {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose x and y
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y+j];
    }
  }

  // Main function
  int main(int argc, char **argv)
{
  const int nx = 1024;
  const int ny = 1024;
  const int mem_size = nx*ny*sizeof(float);

  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("\nDevice : %s\n", prop.name);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
  checkCuda( cudaSetDevice(devId) );

  float *h_idata = (float*)malloc(mem_size);
  float *h_cdata = (float*)malloc(mem_size);
  float *h_tdata = (float*)malloc(mem_size);
  float *gold    = (float*)malloc(mem_size);
  
  float *d_idata, *d_cdata, *d_tdata;
  checkCuda( cudaMalloc(&d_idata, mem_size) );
  checkCuda( cudaMalloc(&d_cdata, mem_size) );
  checkCuda( cudaMalloc(&d_tdata, mem_size) );

  // check parameters and calculate execution configuration
  if (nx % TILE_DIM || ny % TILE_DIM) {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    goto error_exit;
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    goto error_exit;
  }
    
  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_idata[j*nx + i] = j*nx + i;

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_idata[i*nx + j];
  
  // device
  checkCuda( cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );
  
  // events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;

  // ------------
  // time kernels
  // ------------
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");
  
  // ----
  // copy 
  // ----
printf("\nStarting copy kernel...\n");
printf("%25s", "copy");
checkCuda( cudaMemset(d_cdata, 0, mem_size) );
// warm up
copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
printf("Launched copy kernel.\n");
checkCuda( cudaEventRecord(startEvent, 0) );
for (int i = 0; i < NUM_REPS; i++)
   copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
printf("Completed %d repetitions of copy kernel.\n", NUM_REPS);
checkCuda( cudaEventRecord(stopEvent, 0) );
checkCuda( cudaEventSynchronize(stopEvent) );
checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
checkCuda( cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost) );
printf("Copy kernel finished, running postprocess...\n");
postprocess(h_idata, h_cdata, nx*ny, ms);

  // -------------
  // copySharedMem 
  // -------------
printf("\nStarting shared memory copy kernel...\n");
printf("%25s", "shared memory copy");
checkCuda( cudaMemset(d_cdata, 0, mem_size) );
// warm up
copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
printf("Launched shared memory copy kernel.\n");
checkCuda( cudaEventRecord(startEvent, 0) );
for (int i = 0; i < NUM_REPS; i++)
   copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
printf("Completed %d repetitions of shared memory copy kernel.\n", NUM_REPS);
checkCuda( cudaEventRecord(stopEvent, 0) );
checkCuda( cudaEventSynchronize(stopEvent) );
checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
checkCuda( cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost) );
printf("Shared memory copy kernel finished, running postprocess...\n");
postprocess(h_idata, h_cdata, nx * ny, ms);

  // --------------
  // transposeNaive 
  // --------------
printf("\nStarting naive transpose kernel...\n");
printf("%25s", "naive transpose");
checkCuda( cudaMemset(d_tdata, 0, mem_size) );
// warmup
transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
printf("Launched naive transpose kernel.\n");
checkCuda( cudaEventRecord(startEvent, 0) );
for (int i = 0; i < NUM_REPS; i++)
   transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
printf("Completed %d repetitions of naive transpose kernel.\n", NUM_REPS);
checkCuda( cudaEventRecord(stopEvent, 0) );
checkCuda( cudaEventSynchronize(stopEvent) );
checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
printf("Naive transpose kernel finished, running postprocess...\n");
postprocess(gold, h_tdata, nx * ny, ms);

  // ------------------
  // transposeCoalesced 
  // ------------------
printf("\nStarting coalesced transpose kernel...\n");
printf("%25s", "coalesced transpose");
checkCuda( cudaMemset(d_tdata, 0, mem_size) );
// warmup
transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
printf("Launched coalesced transpose kernel.\n");
checkCuda( cudaEventRecord(startEvent, 0) );
for (int i = 0; i < NUM_REPS; i++)
   transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
printf("Completed %d repetitions of coalesced transpose kernel.\n", NUM_REPS);
checkCuda( cudaEventRecord(stopEvent, 0) );
checkCuda( cudaEventSynchronize(stopEvent) );
checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
printf("Coalesced transpose kernel finished, running postprocess...\n");
postprocess(gold, h_tdata, nx * ny, ms);

  // ------------------------
  // transposeNoBankConflicts
  // ------------------------
printf("\nStarting conflict-free transpose kernel...\n");
printf("%25s", "conflict-free transpose");
checkCuda( cudaMemset(d_tdata, 0, mem_size) );
// warmup
transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
printf("Launched conflict-free transpose kernel.\n");
checkCuda( cudaEventRecord(startEvent, 0) );
for (int i = 0; i < NUM_REPS; i++)
   transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
printf("Completed %d repetitions of conflict-free transpose kernel.\n", NUM_REPS);
checkCuda( cudaEventRecord(stopEvent, 0) );
checkCuda( cudaEventSynchronize(stopEvent) );
checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
printf("Conflict-free transpose kernel finished, running postprocess...\n");
postprocess(gold, h_tdata, nx * ny, ms);

error_exit:
  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaFree(d_tdata) );
  checkCuda( cudaFree(d_cdata) );
  checkCuda( cudaFree(d_idata) );
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);
}