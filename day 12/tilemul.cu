#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void tileKernel(const float *dM,const float *dN,float *dP,const int Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for(int i = 0 ; i < TILE_WIDTH/Width ; ++i){
        Mds[ty][tx] = dM[row*Width + i*TILE_WIDTH + tx];
        Nds[ty][tx] = dN[(i*TILE_WIDTH + ty)*Width + col];
        __syncthreads();

        for(int k = 0 ;k<TILE_WIDTH;++k){
            Pvalue = Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();    

        }
    dP[row*Width + col] = Pvalue;
}