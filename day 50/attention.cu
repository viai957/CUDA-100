/*
 * Attention: Optimized CUDA kernel for multi-head attention
 * Math: 
 *   Q = X * Wq, K = X * Wk, V = X * Wv  (for self-attention)
 *   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 * Inputs / Outputs:
 *   - Q: Query tensor (batch_size, seq_len, num_heads, head_dim) - half precision
 *   - K: Key tensor (batch_size, seq_len, num_heads, head_dim) - half precision
 *   - V: Value tensor (batch_size, seq_len, num_heads, head_dim) - half precision
 *   - mask: Optional attention mask (batch_size, seq_len, seq_len) - half precision
 *   - output: Output tensor (batch_size, seq_len, num_heads, head_dim) - half precision
 * Assumptions:
 *   - Forward pass only
 *   - Causal masking for decoder self-attention
 *   - Optimized for Tensor Core operations where possible
 * Parallel Strategy:
 *   - 3D grid of thread blocks (batch_size, num_heads, seq_len)
 *   - Each thread block computes one output sequence position for one head
 *   - Shared memory for K and V to reduce global memory access
 * Mixed Precision Policy:
 *   - Input/output/weights in FP16
 *   - Internal accumulation in FP32 for numerical stability
 * Distributed Hooks: N/A (single GPU implementation)
 * Complexity:
 *   - FLOPs: 2 * batch_size * num_heads * seq_len * seq_len * head_dim
 *   - Bytes moved: batch_size * num_heads * seq_len * head_dim * (2 + seq_len) * sizeof(half)
 * Test Vectors:
 *   - batch_size=2, seq_len=4, num_heads=2, head_dim=32
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <nvToolsExt.h>

// Error checking macro
#define CK(call) \
  { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      std::exit(EXIT_FAILURE); \
    } \
  }

// Constants for attention
#define MAX_SEQ_LENGTH 2048
#define MAX_HEAD_DIM 128
#define THREADS_PER_BLOCK 256
#define BLOCK_SIZE_X 16  // For shared memory tiling

/**
 * CUDA kernel for scaled dot-product attention
 * This computes attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 */
template<bool causal_mask>
__global__ void ScaledDotProductAttentionKernel(
    const half* __restrict__ query,     // [batch_size, seq_len, num_heads, head_dim]
    const half* __restrict__ key,       // [batch_size, seq_len, num_heads, head_dim]
    const half* __restrict__ value,     // [batch_size, seq_len, num_heads, head_dim]
    const half* __restrict__ mask,      // [batch_size, seq_len, seq_len] or nullptr
    half* __restrict__ output,          // [batch_size, seq_len, num_heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim)
{
    // Shared memory for K and V tiles
    __shared__ float s_key[BLOCK_SIZE_X][MAX_HEAD_DIM];
    __shared__ float s_value[BLOCK_SIZE_X][MAX_HEAD_DIM];
    
    // Get indices
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int query_idx = blockIdx.z;
    
    int tid = threadIdx.x;
    
    // Load query vector for this position
    float q_vector[MAX_HEAD_DIM];
    const int q_offset = ((batch_idx * seq_len + query_idx) * num_heads + head_idx) * head_dim;
    
    for (int i = tid; i < head_dim; i += blockDim.x) {
        q_vector[i] = __half2float(query[q_offset + i]);
    }
    
    // Scaling factor for attention
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Initialize accumulator for output
    float output_vector[MAX_HEAD_DIM] = {0.0f};
    
    // Process key and value vectors in tiles
    float attention_weights[BLOCK_SIZE_X];
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    // First pass: compute max for numerical stability in softmax
    for (int key_tile = 0; key_tile < seq_len; key_tile += BLOCK_SIZE_X) {
        int key_limit = min(BLOCK_SIZE_X, seq_len - key_tile);
        
        // Load key tile into shared memory
        for (int i = tid; i < key_limit * head_dim; i += blockDim.x) {
            int k_idx = i / head_dim;
            int h_idx = i % head_dim;
            int k_offset = ((batch_idx * seq_len + (key_tile + k_idx)) * num_heads + head_idx) * head_dim + h_idx;
            s_key[k_idx][h_idx] = __half2float(key[k_offset]);
        }
        
        __syncthreads();
        
        // Compute attention scores for this tile
        for (int k_idx = 0; k_idx < key_limit; k_idx++) {
            // Skip if using causal mask and this is a future token
            if (causal_mask && (key_tile + k_idx) > query_idx) {
                attention_weights[k_idx] = -INFINITY;
                continue;
            }
            
            // Compute dot product
            float dot_product = 0.0f;
            for (int h_idx = 0; h_idx < head_dim; h_idx++) {
                dot_product += q_vector[h_idx] * s_key[k_idx][h_idx];
            }
            
            // Scale dot product
            float score = dot_product * scale;
            
            // Apply mask if provided
            if (mask != nullptr) {
                int mask_offset = batch_idx * seq_len * seq_len + query_idx * seq_len + (key_tile + k_idx);
                score += __half2float(mask[mask_offset]);
            }
            
            attention_weights[k_idx] = score;
            max_val = max(max_val, score);
        }
        
        __syncthreads();
    }
    
    // Second pass: compute softmax and weighted sum
    for (int key_tile = 0; key_tile < seq_len; key_tile += BLOCK_SIZE_X) {
        int key_limit = min(BLOCK_SIZE_X, seq_len - key_tile);
        
        // Load key and value tiles into shared memory
        for (int i = tid; i < key_limit * head_dim; i += blockDim.x) {
            int k_idx = i / head_dim;
            int h_idx = i % head_dim;
            int v_offset = ((batch_idx * seq_len + (key_tile + k_idx)) * num_heads + head_idx) * head_dim + h_idx;
            s_value[k_idx][h_idx] = __half2float(value[v_offset]);
        }
        
        __syncthreads();
        
        // Compute softmax and weighted sum
        for (int k_idx = 0; k_idx < key_limit; k_idx++) {
            // Skip if using causal mask and this is a future token
            if (causal_mask && (key_tile + k_idx) > query_idx) {
                continue;
            }
            
            // Get attention weight from first pass
            float score = attention_weights[k_idx];
            
            // Apply exp(score - max_val) for numerical stability
            float exp_score = expf(score - max_val);
            sum_exp += exp_score;
            
            // Weighted sum with value vectors
            for (int h_idx = 0; h_idx < head_dim; h_idx++) {
                output_vector[h_idx] += exp_score * s_value[k_idx][h_idx];
            }
        }
        
        __syncthreads();
    }
    
    // Normalize by sum of exponentials
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int h_idx = 0; h_idx < head_dim; h_idx++) {
            output_vector[h_idx] *= inv_sum;
        }
    }
    
    // Write output
    int out_offset = ((batch_idx * seq_len + query_idx) * num_heads + head_idx) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        output[out_offset + i] = __float2half(output_vector[i]);
    }
}

/**
 * Host-side launcher for the scaled dot-product attention
 */
void attention_forward_launcher(
    const half* query,
    const half* key,
    const half* value,
    const half* mask,
    half* output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool causal_mask,
    cudaStream_t stream = nullptr)
{
    nvtxRangePush("attention_forward");
    
    // Grid dimensions: [batch_size, num_heads, seq_len]
    dim3 grid(batch_size, num_heads, seq_len);
    dim3 block(THREADS_PER_BLOCK);
    
    // Launch appropriate kernel based on masking
    if (causal_mask) {
        ScaledDotProductAttentionKernel<true><<<grid, block, 0, stream>>>(
            query, key, value, mask, output,
            batch_size, seq_len, num_heads, head_dim
        );
    } else {
        ScaledDotProductAttentionKernel<false><<<grid, block, 0, stream>>>(
            query, key, value, mask, output,
            batch_size, seq_len, num_heads, head_dim
        );
    }
    
    // Check for errors
    CK(cudaGetLastError());
    
    nvtxRangePop();
}

/**
 * CUDA kernel for QKV projection
 * This computes Q = X * Wq, K = X * Wk, V = X * Wv in a single kernel
 */
__global__ void QKVProjectionKernel(
    const half* __restrict__ input,      // [batch_size, seq_len, embed_dim]
    const half* __restrict__ weight_q,   // [embed_dim, num_heads * head_dim]
    const half* __restrict__ weight_k,   // [embed_dim, num_heads * head_dim]
    const half* __restrict__ weight_v,   // [embed_dim, num_heads * head_dim]
    const half* __restrict__ bias_q,     // [num_heads * head_dim]
    const half* __restrict__ bias_k,     // [num_heads * head_dim]
    const half* __restrict__ bias_v,     // [num_heads * head_dim]
    half* __restrict__ query,            // [batch_size, seq_len, num_heads, head_dim]
    half* __restrict__ key,              // [batch_size, seq_len, num_heads, head_dim]
    half* __restrict__ value,            // [batch_size, seq_len, num_heads, head_dim]
    int batch_size,
    int seq_len,
    int embed_dim,
    int num_heads,
    int head_dim)
{
    // Get indices
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.z / head_dim;
    int dim_idx = blockIdx.z % head_dim;
    int tid = threadIdx.x;
    
    // Each thread computes one output element for Q, K, and V
    if (head_idx < num_heads) {
        // Input offset
        int input_offset = (batch_idx * seq_len + seq_idx) * embed_dim;
        
        // Output offsets
        int output_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim + dim_idx;
        
        // Weight offsets
        int weight_offset_q = head_idx * head_dim + dim_idx;
        int weight_offset_k = head_idx * head_dim + dim_idx;
        int weight_offset_v = head_idx * head_dim + dim_idx;
        
        // Compute Q, K, V projections
        float q_val = 0.0f;
        float k_val = 0.0f;
        float v_val = 0.0f;
        
        for (int i = 0; i < embed_dim; i++) {
            float input_val = __half2float(input[input_offset + i]);
            q_val += input_val * __half2float(weight_q[i * num_heads * head_dim + weight_offset_q]);
            k_val += input_val * __half2float(weight_k[i * num_heads * head_dim + weight_offset_k]);
            v_val += input_val * __half2float(weight_v[i * num_heads * head_dim + weight_offset_v]);
        }
        
        // Add bias if provided
        if (bias_q != nullptr) {
            q_val += __half2float(bias_q[weight_offset_q]);
        }
        if (bias_k != nullptr) {
            k_val += __half2float(bias_k[weight_offset_k]);
        }
        if (bias_v != nullptr) {
            v_val += __half2float(bias_v[weight_offset_v]);
        }
        
        // Write outputs
        query[output_offset] = __float2half(q_val);
        key[output_offset] = __float2half(k_val);
        value[output_offset] = __float2half(v_val);
    }
}

/**
 * Host-side launcher for the QKV projection
 */
void qkv_projection_launcher(
    const half* input,
    const half* weight_q,
    const half* weight_k,
    const half* weight_v,
    const half* bias_q,
    const half* bias_k,
    const half* bias_v,
    half* query,
    half* key,
    half* value,
    int batch_size,
    int seq_len,
    int embed_dim,
    int num_heads,
    int head_dim,
    cudaStream_t stream = nullptr)
{
    nvtxRangePush("qkv_projection");
    
    // Grid dimensions: [batch_size, seq_len, num_heads * head_dim]
    dim3 grid(batch_size, seq_len, num_heads * head_dim);
    dim3 block(THREADS_PER_BLOCK);
    
    // Launch kernel
    QKVProjectionKernel<<<grid, block, 0, stream>>>(
        input, weight_q, weight_k, weight_v, bias_q, bias_k, bias_v,
        query, key, value,
        batch_size, seq_len, embed_dim, num_heads, head_dim
    );
    
    // Check for errors
    CK(cudaGetLastError());
    
    nvtxRangePop();
}

/**
 * CUDA kernel for output projection
 * This computes output = attention_output * Wo + bo
 */
__global__ void OutputProjectionKernel(
    const half* __restrict__ attention_output,  // [batch_size, seq_len, num_heads, head_dim]
    const half* __restrict__ weight_output,     // [num_heads * head_dim, embed_dim]
    const half* __restrict__ bias_output,       // [embed_dim]
    half* __restrict__ output,                  // [batch_size, seq_len, embed_dim]
    int batch_size,
    int seq_len,
    int embed_dim,
    int num_heads,
    int head_dim)
{
    // Get indices
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int embed_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (embed_idx < embed_dim) {
        // Output offset
        int output_offset = (batch_idx * seq_len + seq_idx) * embed_dim + embed_idx;
        
        // Compute output projection
        float out_val = 0.0f;
        
        for (int head_idx = 0; head_idx < num_heads; head_idx++) {
            for (int dim_idx = 0; dim_idx < head_dim; dim_idx++) {
                int attn_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim + dim_idx;
                int weight_offset = (head_idx * head_dim + dim_idx) * embed_dim + embed_idx;
                
                out_val += __half2float(attention_output[attn_offset]) * 
                           __half2float(weight_output[weight_offset]);
            }
        }
        
        // Add bias if provided
        if (bias_output != nullptr) {
            out_val += __half2float(bias_output[embed_idx]);
        }
        
        // Write output
        output[output_offset] = __float2half(out_val);
    }
}

/**
 * Host-side launcher for the output projection
 */
void output_projection_launcher(
    const half* attention_output,
    const half* weight_output,
    const half* bias_output,
    half* output,
    int batch_size,
    int seq_len,
    int embed_dim,
    int num_heads,
    int head_dim,
    cudaStream_t stream = nullptr)
{
    nvtxRangePush("output_projection");
    
    // Grid dimensions
    int threads_per_block = 256;
    int blocks_per_dim = (embed_dim + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch_size, seq_len, blocks_per_dim);
    dim3 block(threads_per_block);
    
    // Launch kernel
    OutputProjectionKernel<<<grid, block, 0, stream>>>(
        attention_output, weight_output, bias_output, output,
        batch_size, seq_len, embed_dim, num_heads, head_dim
    );
    
    // Check for errors
    CK(cudaGetLastError());
    
    nvtxRangePop();
}

#ifdef UNIT_TEST
int main() {
    // Test parameters
    const int batch_size = 2;
    const int seq_len = 4;
    const int embed_dim = 64;
    const int num_heads = 2;
    const int head_dim = 32;
    
    // Allocate host memory
    size_t input_size = batch_size * seq_len * embed_dim;
    size_t qkv_size = batch_size * seq_len * num_heads * head_dim;
    size_t weight_size = embed_dim * num_heads * head_dim;
    size_t bias_size = num_heads * head_dim;
    size_t output_size = batch_size * seq_len * embed_dim;
    
    half *h_input = new half[input_size];
    half *h_weight_q = new half[weight_size];
    half *h_weight_k = new half[weight_size];
    half *h_weight_v = new half[weight_size];
    half *h_bias_q = new half[bias_size];
    half *h_bias_k = new half[bias_size];
    half *h_bias_v = new half[bias_size];
    half *h_weight_output = new half[num_heads * head_dim * embed_dim];
    half *h_bias_output = new half[embed_dim];
    half *h_query = new half[qkv_size];
    half *h_key = new half[qkv_size];
    half *h_value = new half[qkv_size];
    half *h_attention_output = new half[qkv_size];
    half *h_output = new half[output_size];
    
    // Initialize with some values
    for (int i = 0; i < input_size; i++) {
        h_input[i] = __float2half(0.01f * i);
    }
    for (int i = 0; i < weight_size; i++) {
        h_weight_q[i] = __float2half(0.01f);
        h_weight_k[i] = __float2half(0.01f);
        h_weight_v[i] = __float2half(0.01f);
    }
    for (int i = 0; i < bias_size; i++) {
        h_bias_q[i] = __float2half(0.1f);
        h_bias_k[i] = __float2half(0.1f);
        h_bias_v[i] = __float2half(0.1f);
    }
    for (int i = 0; i < num_heads * head_dim * embed_dim; i++) {
        h_weight_output[i] = __float2half(0.01f);
    }
    for (int i = 0; i < embed_dim; i++) {
        h_bias_output[i] = __float2half(0.1f);
    }
    
    // Allocate device memory
    half *d_input, *d_weight_q, *d_weight_k, *d_weight_v;
    half *d_bias_q, *d_bias_k, *d_bias_v, *d_weight_output, *d_bias_output;
    half *d_query, *d_key, *d_value, *d_attention_output, *d_output;
    
    CK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CK(cudaMalloc(&d_weight_q, weight_size * sizeof(half)));
    CK(cudaMalloc(&d_weight_k, weight_size * sizeof(half)));
    CK(cudaMalloc(&d_weight_v, weight_size * sizeof(half)));
    CK(cudaMalloc(&d_bias_q, bias_size * sizeof(half)));
    CK(cudaMalloc(&d_bias_k, bias_size * sizeof(half)));
    CK(cudaMalloc(&d_bias_v, bias_size * sizeof(half)));
    CK(cudaMalloc(&d_weight_output, num_heads * head_dim * embed_dim * sizeof(half)));
    CK(cudaMalloc(&d_bias_output, embed_dim * sizeof(half)));
    CK(cudaMalloc(&d_query, qkv_size * sizeof(half)));
    CK(cudaMalloc(&d_key, qkv_size * sizeof(half)));
    CK(cudaMalloc(&d_value, qkv_size * sizeof(half)));
    CK(cudaMalloc(&d_attention_output, qkv_size * sizeof(half)));
    CK(cudaMalloc(&d_output, output_size * sizeof(half)));
    
    // Copy data to device
    CK(cudaMemcpy(d_input, h_input, input_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_weight_q, h_weight_q, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_weight_k, h_weight_k, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_weight_v, h_weight_v, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bias_q, h_bias_q, bias_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bias_k, h_bias_k, bias_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bias_v, h_bias_v, bias_size * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_weight_output, h_weight_output, num_heads * head_dim * embed_dim * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bias_output, h_bias_output, embed_dim * sizeof(half), cudaMemcpyHostToDevice));
    
    // Step 1: QKV Projection
    qkv_projection_launcher(
        d_input, d_weight_q, d_weight_k, d_weight_v,
        d_bias_q, d_bias_k, d_bias_v,
        d_query, d_key, d_value,
        batch_size, seq_len, embed_dim, num_heads, head_dim
    );
    
    // Step 2: Attention
    attention_forward_launcher(
        d_query, d_key, d_value, nullptr, d_attention_output,
        batch_size, seq_len, num_heads, head_dim, true
    );
    
    // Step 3: Output Projection
    output_projection_launcher(
        d_attention_output, d_weight_output, d_bias_output, d_output,
        batch_size, seq_len, embed_dim, num_heads, head_dim
    );
    
    // Copy results back
    CK(cudaMemcpy(h_query, d_query, qkv_size * sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_key, d_key, qkv_size * sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_value, d_value, qkv_size * sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_attention_output, d_attention_output, qkv_size * sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_output, d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Print some results for verification
    printf("✓ MultiHeadAttention test completed!\n");
    printf("Sample output values:\n");
    for (int i = 0; i < 5; i++) {
        printf("  output[%d] = %f\n", i, __half2float(h_output[i]));
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_weight_q;
    delete[] h_weight_k;
    delete[] h_weight_v;
    delete[] h_bias_q;
    delete[] h_bias_k;
    delete[] h_bias_v;
    delete[] h_weight_output;
    delete[] h_bias_output;
    delete[] h_query;
    delete[] h_key;
    delete[] h_value;
    delete[] h_attention_output;
    delete[] h_output;
    
    CK(cudaFree(d_input));
    CK(cudaFree(d_weight_q));
    CK(cudaFree(d_weight_k));
    CK(cudaFree(d_weight_v));
    CK(cudaFree(d_bias_q));
    CK(cudaFree(d_bias_k));
    CK(cudaFree(d_bias_v));
    CK(cudaFree(d_weight_output));
    CK(cudaFree(d_bias_output));
    CK(cudaFree(d_query));
    CK(cudaFree(d_key));
    CK(cudaFree(d_value));
    CK(cudaFree(d_attention_output));
    CK(cudaFree(d_output));
    
    return 0;
}
#endif

/* Profiling example & performance tips 
 * 
 * To profile this kernel:
 * 1. nsys profile --stats=true ./attention_test
 * 
 * Performance tips:
 * - For larger batch sizes or sequence lengths, consider using tensor cores via WMMA API
 * - For small batch sizes or sequence lengths, reduce shared memory usage
 * - Consider using FlashAttention algorithm for better memory efficiency
 * - For inference-only workloads, consider using KV cache
 */ 