#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void Attention(float *Q, float *K, float*V, float *O, )