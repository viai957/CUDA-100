#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

#define NUM_POINTS 10000
#define K 3
#define DIM 2  
#define MAX_ITER 100

__global__ void assign_clusters(float *points, float *centroids, int *labels, int numPoints, int k, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    float minDist = FLT_MAX;
    int bestCluster = 0;

    // Compute distance from point[i] to each centroid
    for (int j = 0; j < k; j++) {
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = points[i * dim + d] - centroids[j * dim + d];
            dist += diff * diff;
        }
        
        if (dist < minDist) {
            minDist = dist;
            bestCluster = j;
        }
    }

    labels[i] = bestCluster;
}

__global__ void compute_new_centroids(float *points, float *centroids, int *labels, int *clusterSizes, int numPoints, int k, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    int cluster = labels[i];
    
    atomicAdd(&clusterSizes[cluster], 1);
    for (int d = 0; d < dim; d++) {
        atomicAdd(&centroids[cluster * dim + d], points[i * dim + d]);
    }
}

void kmeans(float *h_points, int numPoints, int k, int dim) {
    size_t pointsSize = numPoints * dim * sizeof(float);
    size_t centroidsSize = k * dim * sizeof(float);
    size_t labelsSize = numPoints * sizeof(int);
    size_t clusterSizesSize = k * sizeof(int);

    float *d_points, *d_centroids;
    int *d_labels, *d_clusterSizes;
    cudaMalloc(&d_points, pointsSize);
    cudaMalloc(&d_centroids, centroidsSize);
    cudaMalloc(&d_labels, labelsSize);
    cudaMalloc(&d_clusterSizes, clusterSizesSize);

    cudaMemcpy(d_points, h_points, pointsSize, cudaMemcpyHostToDevice);

    float *h_centroids = (float*)malloc(centroidsSize);
    for (int i = 0; i < k; i++) {
        int idx = rand() % numPoints;
        for (int d = 0; d < dim; d++) {
            h_centroids[i * dim + d] = h_points[idx * dim + d];
        }
    }
    cudaMemcpy(d_centroids, h_centroids, centroidsSize, cudaMemcpyHostToDevice);

    int *h_labels = (int*)malloc(labelsSize);

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        assign_clusters<<<gridSize, blockSize>>>(d_points, d_centroids, d_labels, numPoints, k, dim);
        cudaDeviceSynchronize();

        cudaMemset(d_centroids, 0, centroidsSize);
        cudaMemset(d_clusterSizes, 0, clusterSizesSize);

        compute_new_centroids<<<gridSize, blockSize>>>(d_points, d_centroids, d_labels, d_clusterSizes, numPoints, k, dim);
        cudaDeviceSynchronize();

        int *h_clusterSizes = (int*)malloc(clusterSizesSize);
        cudaMemcpy(h_clusterSizes, d_clusterSizes, clusterSizesSize, cudaMemcpyDeviceToHost);

        cudaMemcpy(h_centroids, d_centroids, centroidsSize, cudaMemcpyDeviceToHost);

        for (int i = 0; i < k; i++) {
            if (h_clusterSizes[i] > 0) {  // Changed from h_labels to h_clusterSizes
                for (int d = 0; d < dim; d++) {
                    h_centroids[i * dim + d] /= h_clusterSizes[i];
                }
            }
        }

        cudaMemcpy(d_centroids, h_centroids, centroidsSize, cudaMemcpyHostToDevice);
    }

    printf("Final Cluster Centers:\n");
    for (int i = 0; i < k; i++) {
        printf("Cluster %d: (%.2f, %.2f)\n", i, h_centroids[i * dim], h_centroids[i * dim + 1]);
    }

    free(h_centroids);
    free(h_labels);
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_clusterSizes);
}

int main() {
    srand(42);

    float *h_points = (float*)malloc(NUM_POINTS * DIM * sizeof(float));
    for (int i = 0; i < NUM_POINTS; i++) {
        h_points[i * DIM] = ((float)rand() / RAND_MAX) * 100.0f;   // X coordinate
        h_points[i * DIM + 1] = ((float)rand() / RAND_MAX) * 100.0f; // Y coordinate
    }

    kmeans(h_points, NUM_POINTS, K, DIM);

    free(h_points);
    return 0;
}