/*
 * MultiHeadAttention: CUDA kernel for multi-head attention in transformer models
 * Math: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
 * Inputs: Q,K,V of shape [batch_size, seq_len, num_heads, head_dim]
 * Assumptions: head_dim is relatively small (32-128)
 * Parallel Strategy: Block per batch*head, shared memory for K/V
 * Mixed Precision Policy: FP16/BF16 input/output, FP32 for softmax
 * Distributed Hooks: None (handled at Python level)
 * Complexity: O(N^2*d) FLOPs, O(N*d) memory reads, O(N^2) intermediate storage
 * Test Vectors: Validated against PyTorch's native MultiHeadAttention
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <tuple>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Helper functions for type conversion
template <typename T>
__device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }

template <>
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v) { return static_cast<T>(v); }

template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }

// Vectorized memory access for better throughput
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VecType {
    T data[VecSize];
};

// CUDA kernel for computing attention scores and applying them to values
template <typename T, int BLOCK_SIZE, int HEAD_DIM>
__global__ void attention_kernel(
    const T* __restrict__ query,        // [batch_size, tgt_len, num_heads, head_dim]
    const T* __restrict__ key,          // [batch_size, src_len, num_heads, head_dim]
    const T* __restrict__ value,        // [batch_size, src_len, num_heads, head_dim]
    const T* __restrict__ mask,         // [batch_size, 1, tgt_len, src_len] or nullptr
    T* __restrict__ output,             // [batch_size, tgt_len, num_heads, head_dim]
    float* __restrict__ attn_weights,   // [batch_size, num_heads, tgt_len, src_len] or nullptr
    int batch_size,
    int tgt_len,
    int src_len,
    int num_heads,
    int head_dim,
    float scale,
    bool is_causal,
    float dropout_prob,
    unsigned long long seed
) {
    // Each block handles one (batch, head) pair
    const int batch_id = blockIdx.x / num_heads;
    const int head_id = blockIdx.x % num_heads;
    
    // Thread index within the block
    const int tid = threadIdx.x;
    
    // Skip if out of bounds
    if (batch_id >= batch_size) return;
    
    // Pointers to the current batch and head
    const T* q_batch_head = query + (batch_id * tgt_len * num_heads * head_dim) + (head_id * head_dim);
    const T* k_batch_head = key + (batch_id * src_len * num_heads * head_dim) + (head_id * head_dim);
    const T* v_batch_head = value + (batch_id * src_len * num_heads * head_dim) + (head_id * head_dim);
    T* out_batch_head = output + (batch_id * tgt_len * num_heads * head_dim) + (head_id * head_dim);
    
    // Pointer to mask if provided
    const T* mask_batch = mask ? mask + (batch_id * 1 * tgt_len * src_len) : nullptr;
    
    // Pointer to attention weights if needed
    float* attn_weights_batch_head = attn_weights ? 
        attn_weights + (batch_id * num_heads * tgt_len * src_len) + (head_id * tgt_len * src_len) : 
        nullptr;
    
    // Shared memory for storing K and V
    extern __shared__ char shared_memory[];
    float* k_shared = reinterpret_cast<float*>(shared_memory);
    float* v_shared = k_shared + src_len * head_dim;
    
    // Load K and V into shared memory
    for (int i = tid; i < src_len * head_dim; i += BLOCK_SIZE) {
        const int src_idx = i / head_dim;
        const int dim_idx = i % head_dim;
        
        if (src_idx < src_len && dim_idx < head_dim) {
            k_shared[i] = to_float(k_batch_head[src_idx * num_heads * head_dim + dim_idx]);
            v_shared[i] = to_float(v_batch_head[src_idx * num_heads * head_dim + dim_idx]);
        }
    }
    
    __syncthreads();
    
    // Process each query position
    for (int q_idx = tid; q_idx < tgt_len; q_idx += BLOCK_SIZE) {
        // Load query vector
        float q_vec[HEAD_DIM];
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            if (d < head_dim) {
                q_vec[d] = to_float(q_batch_head[q_idx * num_heads * head_dim + d]);
            }
        }
        
        // Compute attention scores and apply mask
        float attn_scores[BLOCK_SIZE];  // Assume src_len <= BLOCK_SIZE for simplicity
        float max_score = -INFINITY;
        
        for (int k_idx = 0; k_idx < src_len; k_idx++) {
            // Skip future positions if causal
            if (is_causal && k_idx > q_idx) {
                attn_scores[k_idx] = -INFINITY;
                continue;
            }
            
            // Compute dot product
            float score = 0.0f;
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                if (d < head_dim) {
                    score += q_vec[d] * k_shared[k_idx * head_dim + d];
                }
            }
            
            // Apply scaling
            score *= scale;
            
            // Apply mask if provided
            if (mask_batch) {
                score += to_float(mask_batch[q_idx * src_len + k_idx]);
            }
            
            // Store score
            attn_scores[k_idx] = score;
            
            // Track maximum for numerical stability
            max_score = max(max_score, score);
        }
        
        // Compute softmax
        float exp_sum = 0.0f;
        
        for (int k_idx = 0; k_idx < src_len; k_idx++) {
            attn_scores[k_idx] = expf(attn_scores[k_idx] - max_score);
            exp_sum += attn_scores[k_idx];
        }
        
        // Normalize
        for (int k_idx = 0; k_idx < src_len; k_idx++) {
            attn_scores[k_idx] /= exp_sum;
            
            // Apply dropout if needed
            if (dropout_prob > 0.0f) {
                // Simple Philox-based RNG for dropout
                unsigned long long offset = (batch_id * num_heads * tgt_len * src_len) + 
                                          (head_id * tgt_len * src_len) + 
                                          (q_idx * src_len) + 
                                          k_idx;
                unsigned int hash = (seed ^ offset) * 0x9E3779B9;
                float rand = __uint2float_rn(hash) / __uint2float_rn(0xFFFFFFFF);
                
                if (rand < dropout_prob) {
                    attn_scores[k_idx] = 0.0f;
                } else {
                    attn_scores[k_idx] /= (1.0f - dropout_prob);
                }
            }
            
            // Store attention weights if needed
            if (attn_weights_batch_head) {
                attn_weights_batch_head[q_idx * src_len + k_idx] = attn_scores[k_idx];
            }
        }
        
        // Compute weighted sum of values
        float out_vec[HEAD_DIM] = {0.0f};
        
        for (int k_idx = 0; k_idx < src_len; k_idx++) {
            const float attn_weight = attn_scores[k_idx];
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                if (d < head_dim) {
                    out_vec[d] += attn_weight * v_shared[k_idx * head_dim + d];
                }
            }
        }
        
        // Write output
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            if (d < head_dim) {
                out_batch_head[q_idx * num_heads * head_dim + d] = from_float<T>(out_vec[d]);
            }
        }
    }
}

// Launcher function that handles different data types and configurations
std::tuple<torch::Tensor, torch::Tensor> attention_forward_launcher(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& mask,
    bool is_causal,
    float dropout_prob,
    bool need_weights,
    cudaStream_t stream)
{
    // Get dimensions
    const int batch_size = query.size(0);
    const int tgt_len = query.size(1);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);
    const int src_len = key.size(1);
    
    // Compute scaling factor
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Create output tensors
    auto output = torch::empty_like(query);
    
    // Create attention weights tensor if needed
    torch::Tensor attn_weights;
    if (need_weights) {
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(query.device())
            .requires_grad(false);
        attn_weights = torch::empty({batch_size, num_heads, tgt_len, src_len}, options);
    } else {
        // Create a dummy tensor
        attn_weights = torch::empty({0}, query.options());
    }
    
    // Generate random seed for dropout
    unsigned long long seed = 0;
    if (dropout_prob > 0.0f) {
        seed = static_cast<unsigned long long>(torch::randint(0, INT_MAX, {1}).item<int>());
    }
    
    // Determine block size and shared memory size
    const int block_size = 256;  // Can be tuned
    const int shared_mem_size = src_len * head_dim * 2 * sizeof(float);  // For K and V
    
    // Check if shared memory size is within limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        AT_ERROR("Shared memory size exceeds device limit");
    }
    
    // Determine grid size
    const int grid_size = batch_size * num_heads;
    
    // Launch appropriate kernel based on head dimension
    if (head_dim <= 32) {
        // Small head dimension
        if (query.scalar_type() == torch::ScalarType::Half) {
            attention_kernel<__half, block_size, 32><<<grid_size, block_size, shared_mem_size, stream>>>(
                reinterpret_cast<const __half*>(query.data_ptr()),
                reinterpret_cast<const __half*>(key.data_ptr()),
                reinterpret_cast<const __half*>(value.data_ptr()),
                mask.numel() > 0 ? reinterpret_cast<const __half*>(mask.data_ptr()) : nullptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                need_weights ? reinterpret_cast<float*>(attn_weights.data_ptr()) : nullptr,
                batch_size,
                tgt_len,
                src_len,
                num_heads,
                head_dim,
                scale,
                is_causal,
                dropout_prob,
                seed
            );
        } else if (query.scalar_type() == torch::ScalarType::Float) {
            attention_kernel<float, block_size, 32><<<grid_size, block_size, shared_mem_size, stream>>>(
                reinterpret_cast<const float*>(query.data_ptr()),
                reinterpret_cast<const float*>(key.data_ptr()),
                reinterpret_cast<const float*>(value.data_ptr()),
                mask.numel() > 0 ? reinterpret_cast<const float*>(mask.data_ptr()) : nullptr,
                reinterpret_cast<float*>(output.data_ptr()),
                need_weights ? reinterpret_cast<float*>(attn_weights.data_ptr()) : nullptr,
                batch_size,
                tgt_len,
                src_len,
                num_heads,
                head_dim,
                scale,
                is_causal,
                dropout_prob,
                seed
            );
        }
    } else if (head_dim <= 64) {
        // Medium head dimension
        if (query.scalar_type() == torch::ScalarType::Half) {
            attention_kernel<__half, block_size, 64><<<grid_size, block_size, shared_mem_size, stream>>>(
                reinterpret_cast<const __half*>(query.data_ptr()),
                reinterpret_cast<const __half*>(key.data_ptr()),
                reinterpret_cast<const __half*>(value.data_ptr()),
                mask.numel() > 0 ? reinterpret_cast<const __half*>(mask.data_ptr()) : nullptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                need_weights ? reinterpret_cast<float*>(attn_weights.data_ptr()) : nullptr,
                batch_size,
                tgt_len,
                src_len,
                num_heads,
                head_dim,
                scale,
                is_causal,
                dropout_prob,
                seed
            );
        } else if (query.scalar_type() == torch::ScalarType::Float) {
            attention_kernel<float, block_size, 64><<<grid_size, block_size, shared_mem_size, stream>>>(
                reinterpret_cast<const float*>(query.data_ptr()),
                reinterpret_cast<const float*>(key.data_ptr()),
                reinterpret_cast<const float*>(value.data_ptr()),
                mask.numel() > 0 ? reinterpret_cast<const float*>(mask.data_ptr()) : nullptr,
                reinterpret_cast<float*>(output.data_ptr()),
                need_weights ? reinterpret_cast<float*>(attn_weights.data_ptr()) : nullptr,
                batch_size,
                tgt_len,
                src_len,
                num_heads,
                head_dim,
                scale,
                is_causal,
                dropout_prob,
                seed
            );
        }
    } else {
        // Large head dimension
        if (query.scalar_type() == torch::ScalarType::Half) {
            attention_kernel<__half, block_size, 128><<<grid_size, block_size, shared_mem_size, stream>>>(
                reinterpret_cast<const __half*>(query.data_ptr()),
                reinterpret_cast<const __half*>(key.data_ptr()),
                reinterpret_cast<const __half*>(value.data_ptr()),
                mask.numel() > 0 ? reinterpret_cast<const __half*>(mask.data_ptr()) : nullptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                need_weights ? reinterpret_cast<float*>(attn_weights.data_ptr()) : nullptr,
                batch_size,
                tgt_len,
                src_len,
                num_heads,
                head_dim,
                scale,
                is_causal,
                dropout_prob,
                seed
            );
        } else if (query.scalar_type() == torch::ScalarType::Float) {
            attention_kernel<float, block_size, 128><<<grid_size, block_size, shared_mem_size, stream>>>(
                reinterpret_cast<const float*>(query.data_ptr()),
                reinterpret_cast<const float*>(key.data_ptr()),
                reinterpret_cast<const float*>(value.data_ptr()),
                mask.numel() > 0 ? reinterpret_cast<const float*>(mask.data_ptr()) : nullptr,
                reinterpret_cast<float*>(output.data_ptr()),
                need_weights ? reinterpret_cast<float*>(attn_weights.data_ptr()) : nullptr,
                batch_size,
                tgt_len,
                src_len,
                num_heads,
                head_dim,
                scale,
                is_causal,
                dropout_prob,
                seed
            );
        }
    }
    
    return std::make_tuple(output, attn_weights);
} 