#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void multi_head_attention_kernel(
    const scalar_t* __restrict__ Q,      // [batch_size, seq_len, d_model]
    const scalar_t* __restrict__ K,      // [batch_size, seq_len, d_model]
    const scalar_t* __restrict__ V,      // [batch_size, seq_len, d_model]
    scalar_t* __restrict__ O,            // [batch_size, seq_len, d_model]
    scalar_t* __restrict__ temp,         // [batch_size, h, seq_len, seq_len] - temporary storage for attention scores
    int seq_len,
    int d_model,
    int h,
    int d_k,                             // d_k = d_model / h
    float dropout_prob,
    int* rand_seeds                      // Random seeds for dropout
) {
    // Calculate indices for this thread
    const int batch_idx = blockIdx.z;                            // Batch dimension
    const int head_idx = blockIdx.y;                             // Head dimension
    const int row = threadIdx.y + blockIdx.x * blockDim.y;       // seq_len dimension (query seq position)
    const int col = threadIdx.x + blockDim.x * blockIdx.x;       // seq_len dimension (key seq position)
    
    // Skip out-of-bounds threads
    if (row >= seq_len || col >= seq_len || batch_idx >= gridDim.z || head_idx >= h) {
        return;
    }
    
    // Calculate offsets for each head's part of Q, K, V matrices
    const int batch_offset = batch_idx * seq_len * d_model;
    const int head_offset = head_idx * d_k;
    
    // Step 1: Calculate attention score for this position (QK^T / sqrt(d_k))
    scalar_t score = 0.0f;
    for (int i = 0; i < d_k; i++) {
        // Q[batch, row, head_offset + i] * K[batch, col, head_offset + i]
        const int q_idx = batch_offset + row * d_model + head_offset + i;
        const int k_idx = batch_offset + col * d_model + head_offset + i;
        score += Q[q_idx] * K[k_idx];
    }
    
    // Scale by sqrt(d_k)
    score /= sqrt(static_cast<float>(d_k));
    
    // Store in temporary buffer for softmax
    const int temp_idx = batch_idx * h * seq_len * seq_len + 
                        head_idx * seq_len * seq_len + 
                        row * seq_len + col;
    temp[temp_idx] = score;
    
    // Sync to ensure all scores are computed before softmax
    __syncthreads();
    
    // Only the threads processing the first column of each row compute softmax
    if (col == 0) {
        // Step 2: Find maximum score for numerical stability
        scalar_t max_val = -INFINITY;
        for (int i = 0; i < seq_len; i++) {
            const int idx = batch_idx * h * seq_len * seq_len + 
                          head_idx * seq_len * seq_len + 
                          row * seq_len + i;
            max_val = max(max_val, temp[idx]);
        }
        
        // Step 3: Compute exponentials and sum
        scalar_t sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            const int idx = batch_idx * h * seq_len * seq_len + 
                          head_idx * seq_len * seq_len + 
                          row * seq_len + i;
            temp[idx] = exp(temp[idx] - max_val);
            
            // Apply dropout if needed
            if (dropout_prob > 0.0f) {
                // Simple hash-based random number generator
                unsigned int seed = rand_seeds[batch_idx * h * seq_len + head_idx * seq_len + row];
                unsigned int hash = (seed ^ i) + (seed << 6) + (seed >> 2);
                float rand_val = static_cast<float>(hash % 1000) / 1000.0f;
                
                if (rand_val < dropout_prob) {
                    temp[idx] = 0.0f;
                } else {
                    temp[idx] /= (1.0f - dropout_prob);  // Scale remaining values
                }
            }
            
            sum_exp += temp[idx];
        }
        
        // Step 4: Normalize with softmax
        if (sum_exp > 0.0f) {  // Avoid division by zero
            for (int i = 0; i < seq_len; i++) {
                const int idx = batch_idx * h * seq_len * seq_len + 
                              head_idx * seq_len * seq_len + 
                              row * seq_len + i;
                temp[idx] /= sum_exp;
            }
        }
    }
    
    // Wait for softmax to complete
    __syncthreads();
    
    // Step 5: Multiply by V and store results - each thread handles one value element
    // Only threads with col < d_k compute the final output
    if (col < d_k) {
        scalar_t sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            // Attention score for this position
            const int score_idx = batch_idx * h * seq_len * seq_len + 
                                head_idx * seq_len * seq_len + 
                                row * seq_len + i;
            
            // V value for this position
            const int v_idx = batch_offset + i * d_model + head_offset + col;
            
            sum += temp[score_idx] * V[v_idx];
        }
        
        // Store result in output
        const int out_idx = batch_offset + row * d_model + head_offset + col;
        O[out_idx] = sum;
    }
}

torch::Tensor multi_head_attention_cuda_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    int h,
    float dropout_prob) {
    
    const auto batch_size = Q.size(0);
    const auto seq_len = Q.size(1);
    const auto d_model = Q.size(2);
    const auto d_k = d_model / h;
    
    // Create output tensor of same shape as input
    auto O = torch::zeros_like(Q);
    
    // Allocate temporary storage for attention scores
    auto temp = torch::zeros({batch_size, h, seq_len, seq_len}, 
                            Q.options());
    
    // Random seeds for dropout
    auto rand_seeds = torch::randint(0, 1000000, {batch_size * h * seq_len}, 
                                   torch::TensorOptions().device(Q.device()).dtype(torch::kInt32));
    
    // Define block and grid dimensions
    dim3 block_dim(16, 16);  // 16x16 threads per block
    
    // Each block handles a portion of seq_len x seq_len for QK^T
    // x dimension for seq_len/16 blocks, y dimension for h heads, z dimension for batch_size
    dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x, 
                h, 
                batch_size);
    
    // Get stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel with appropriate tensor types
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "multi_head_attention_kernel", ([&] {
        multi_head_attention_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            O.data_ptr<scalar_t>(),
            temp.data_ptr<scalar_t>(),
            seq_len,
            d_model,
            h,
            d_k,
            dropout_prob,
            rand_seeds.data_ptr<int>()
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    return O;
} 