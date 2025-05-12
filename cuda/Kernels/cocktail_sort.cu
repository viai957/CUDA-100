#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdint.h>
#include <chrono>
#include "cuda_common.cuh"

/*
 * Cocktail Shaker Sort Benchmark
 *  Example : [5, 1, 4, 2, 8, 0, 2]
 * 
 * Forrward pass: (left to right):
 * 5, 1, 4, 2, 8, 0, 2] → [1, 5, 4, 2, 8, 0, 2] → [1, 4, 5, 2, 8, 0, 2] → 
   [1, 4, 2, 5, 8, 0, 2] → [1, 4, 2, 5, 0, 8, 2] → [1, 4, 2, 5, 0, 2, 8]
 * 
 * Backward pass: (right to left):
 * [1, 4, 2, 5, 0, 2, 8] → [1, 4, 2, 5, 0, 2, 8] → [1, 4, 2, 0, 5, 2, 8] → 
   [1, 4, 2, 0, 5, 2, 8] → [1, 4, 0, 2, 5, 2, 8] → [1, 4, 0, 2, 5, 2, 8]
 * 
 */

// Cocktail Shaker Sort implementation (host/CPU)
void cocktailShakerSort