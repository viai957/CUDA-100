#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <cfloat>
#include <cuda_runtime.h>

// Multi-Head Attention Kernel
__gloabl__ void multi_head_attention_kernel(
    float *Q, // [batch_size, seq_len, d_model]
    float *K, // [batch_size, seq_len, d_model]
    float *V, // [batch_size, seq_len, d_model]
    float *O, // [batch_size, seq_len, d_model]
    float *temp // [batch_size, h, seq_len, seq_len] - temporary storage for attention scores
    int seq_len, 
    int d_model,
    int h,
    int d_k, // d_k = d_model / h
    float dropout_prob // Dropout probability
) {
    // Calculate indices for this thread
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int row = threadIdx.y + blockIdx.x * blockDim.y;
    const int col = threadIdx.x + blockDim.x * blockIdx.x;

    // Skip out-of scope threads
    if (row >= seq_len || col >= seq_len || batch_idx >= gridDim.z || head_idx >= h){
        return;
    }

    // Calculate offset for each head's part of Q, K, V matrices
    const int batch_offset = batch_idx * seq_len * d_model; // 2 * 14 * 128 = 3584
    // Each head has its own part of the d_model dimension
    // d_k is the dimension of each head, so we calculate the offset for this head
    // For example, if d_model = 128 and h = 8, then d_k = 16
    // Each head has d_k dimensions, so the offset for this head is head_idx * d_k
    const int head_offset = head_idx * d_k;

    // Step 1: Calculate attention score for this position (QK^T / sqrt(d_k))
    float score = 0.0f;
    for (int i = 0; i < d_k; i++){
        // Q[batch, row, head_offset + i] * K[batch, col, head_offset + i]
        const int q_idx = batch_offset + row * d_model + head_offset + i;
        const int k_idx = batch_offset + col * d_model + head_offset + i;
        score += Q[q_idx] * K[k_idx];
    }

    // Scale by sqrt(d_k)
    score /= sqrtf(static_cast<float>(d_k));

    // Step 2: Store the score in the temporary storage
    const int temp_idx = batch_size * h * seq_len * se_len + 
                            head_idx * seq_len * seq_len +
                            row * seq_len + col;
    temp[temp_idx] = score;

    // Sync to ensure all the score for numerical stability
    __syncthreads();

    // Only the threads processing the first column of each row compute softmax
    if (col == 0){
        // Find the maximum score in this row for numerical stability
        float max_score = -FLT_MAX;
        for (int i = 0; i < seq_len; i++){
            const int idx = batch_idx * h * seq_len * seq_len + 
                            head_idx * seq_len * seq_len +
                            row * seq_len + i;
            max_value = fmaxf(max_val, temp[idx]);
        }

        // Step 3: Compute exponential of scores and sum them
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++){
            const int idx = batch_idx * h * seq_len * seq_len + 
                            head_idx * seq_len * seq_len +
                            row * seq_len + i;
            temp[idx] = expf(temp[idx] - max_score);

            // Apply dropout if needed
            if (dropout_prob > 0.0f){
                // Note: In a real implemnentation, we'd use cuRAND for true random dropout
                // This is just a placeholder that drops values based on their position
                if ((i % row) % 100 < dropout_prob * 100){
                    temp [idx] = 0.0f; // Drop this value
                } else {
                    temp[idx] / = (1.0f - dropout_prob); // Scale remaining values
                }
            }
            sum_exp += temp[idx];
        }

        // Step 4: Normilize with softmax
        if (sum_exp > 0.0f){
            for (int i = 0; i < seq_len; i++){
                const int idx = batch_idx * h * seq_len * seq_len +
                                head_idx * seq_len * seq_len +
                                row * seq_len + i;
            temp[idx] /= sum_exp;
            }
        }
    }

    // Wait for softmax to complete
    __syncthreads();

    // Step 5: Compute the output for this position (softmax(QK^T) * V) - each thread handles one value element
    // Only threads with col < d_k compute the final output
    if (col < d_k){
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++){
            // Attention score for this position
            const int score_idx = batch_idx * h * seq_len + 
                                    head_idx * seq_len * seq_len +
                                    row * seq_len + i;

            // V value for this position
            const int v_idx = batch_offset + i * d_model = head_offset + col;

            sum += temp[score_idx] * V[v_idx];
        }

        // Store result in output matrix O
        const int o_idx = batch_offset + row * d_model + head_offset + col;
        O[o_idx] = sum;
    }
}

// CPU implementation of multi-head attention for verification
void cpu_multi_head_attention(
    float *Q,      // [batch_size, seq_len, d_model]
    float *K,      // [batch_size, seq_len, d_model]
    float *V,      // [batch_size, seq_len, d_model]
    float *O,      // [batch_size, seq_len, d_model]
    int batch_size,
    int seq_len,
    int d_model,
    int h,
    float dropout_prob = 0.0f
) {
    const int d_k = d_model / h;
    
    // Temporary storage for attention scores
    std::vector<float> attention_scores(batch_size * h * seq_len * seq_len, 0.0f);
    
    // For each batch and head
    for (int b = 0; b < batch_size; b++) {
        for (int head = 0; head < h; head++) {
            
            // Calculate attention scores (Q * K^T / sqrt(d_k))
            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                    float score = 0.0f;
                    
                    // Dot product of query and key
                    for (int i = 0; i < d_k; i++) {
                        int q_idx = b * seq_len * d_model + q_pos * d_model + head * d_k + i;
                        int k_idx = b * seq_len * d_model + k_pos * d_model + head * d_k + i;
                        score += Q[q_idx] * K[k_idx];
                    }
                    
                    // Scale
                    score /= sqrtf(static_cast<float>(d_k));
                    
                    // Store score
                    int score_idx = b * h * seq_len * seq_len + head * seq_len * seq_len + q_pos * seq_len + k_pos;
                    attention_scores[score_idx] = score;
                }
                
                // Apply softmax for this row
                // 1. Find max for numerical stability
                float max_val = -FLT_MAX;
                for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                    int score_idx = b * h * seq_len * seq_len + head * seq_len * seq_len + q_pos * seq_len + k_pos;
                    max_val = std::max(max_val, attention_scores[score_idx]);
                }
                
                // 2. Compute exp and sum
                float sum_exp = 0.0f;
                for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                    int score_idx = b * h * seq_len * seq_len + head * seq_len * seq_len + q_pos * seq_len + k_pos;
                    attention_scores[score_idx] = expf(attention_scores[score_idx] - max_val);
                    
                    // Apply dropout if needed
                    if (dropout_prob > 0.0f) {
                        // Simple deterministic dropout for CPU version
                        if ((q_pos * k_pos) % 100 < dropout_prob * 100) {
                            attention_scores[score_idx] = 0.0f;
                        } else {
                            attention_scores[score_idx] /= (1.0f - dropout_prob);
                        }
                    }
                    
                    sum_exp += attention_scores[score_idx];
                }
                
                // 3. Normalize
                if (sum_exp > 0.0f) {
                    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                        int score_idx = b * h * seq_len * seq_len + head * seq_len * seq_len + q_pos * seq_len + k_pos;
                        attention_scores[score_idx] /= sum_exp;
                    }
                }
            }
            
            // Multiply attention scores by values
            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                for (int i = 0; i < d_k; i++) {
                    float sum = 0.0f;
                    
                    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                        int score_idx = b * h * seq_len * seq_len + head * seq_len * seq_len + q_pos * seq_len + k_pos;
                        int v_idx = b * seq_len * d_model + k_pos * d_model + head * d_k + i;
                        sum += attention_scores[score_idx] * V[v_idx];
                    }
                    
                    // Store in output
                    int out_idx = b * seq_len * d_model + q_pos * d_model + head * d_k + i;
                    O[out_idx] = sum;
                }
            }
        }
    }
}

int main() {
    // Model configuration
    int batch_size = 2;
    int seq_len = 128;
    int d_model = 512;
    int h = 8;        // Number of attention heads
    int d_k = d_model / h;  // Dimension per head
    float dropout_prob = 0.1f;
    
    // Validate configuration
    if (d_model % h != 0) {
        std::cerr << "Error: d_model must be divisible by h" << std::endl;
        return 1;
    }

    // Allocate memory for input and output
    size_t total_size = batch_size * seq_len * d_model;
    float *h_Q = (float *)malloc(total_size * sizeof(float));
    float *h_K = (float *)malloc(total_size * sizeof(float));
    float *h_V = (float *)malloc(total_size * sizeof(float));
    float *h_O_cuda = (float *)malloc(total_size * sizeof(float));
    float *h_O_cpu = (float *)malloc(total_size * sizeof(float));
    
    // Initialize input data with simple pattern for verification
    for (size_t i = 0; i < total_size; i++) {
        h_Q[i] = 0.1f * (i % 10);
        h_K[i] = 0.1f * (i % 10); 
        h_V[i] = 0.1f * (i % 10);
    }

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_temp;
    cudaMalloc(&d_Q, total_size * sizeof(float));
    cudaMalloc(&d_K, total_size * sizeof(float));
    cudaMalloc(&d_V, total_size * sizeof(float));
    cudaMalloc(&d_O, total_size * sizeof(float));
    
    // Allocate temp memory for attention scores
    size_t temp_size = batch_size * h * seq_len * seq_len;
    cudaMalloc(&d_temp, temp_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_Q, h_Q, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    
    // Each block handles a portion of seq_len x seq_len for QK^T
    // x dimension for seq_len/16 blocks, y dimension for h heads, z dimension for batch_size
    dim3 gridDim((seq_len + blockDim.x - 1) / blockDim.x, 
                h, 
                batch_size);

    std::cout << "Launching kernel with grid: (" << gridDim.x << ", " 
              << gridDim.y << ", " << gridDim.z << ")" << std::endl;
    std::cout << "Block size: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
              
    // CUDA execution timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    multi_head_attention_kernel<<<gridDim, blockDim>>>(
        d_Q, d_K, d_V, d_O, d_temp, 
        seq_len, d_model, h, d_k, dropout_prob);
    
    // Wait for kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "CUDA execution time: " << duration.count() << " ms" << std::endl;

    // Copy results back to host
    cudaMemcpy(h_O_cuda, d_O, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU execution for verification
    start = std::chrono::high_resolution_clock::now();
    cpu_multi_head_attention(h_Q, h_K, h_V, h_O_cpu, batch_size, seq_len, d_model, h, dropout_prob);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "CPU execution time: " << duration.count() << " ms" << std::endl;

    // Verify results (compare a subset of outputs)
    float max_diff = 0.0f;
    int diff_count = 0;
    
    for (int i = 0; i < 100 && i < total_size; i++) {
        float diff = std::abs(h_O_cuda[i] - h_O_cpu[i]);
        if (diff > 1e-3f) {
            diff_count++;
            max_diff = std::max(max_diff, diff);
            
            if (diff_count <= 5) {  // Print only first 5 differences
                std::cout << "Difference at index " << i << ": CUDA=" 
                      << h_O_cuda[i] << ", CPU=" << h_O_cpu[i] 
                      << ", diff=" << diff << std::endl;
            }
        }
    }
    
    if (diff_count == 0) {
        std::cout << "Results match within tolerance!" << std::endl;
    } else {
        std::cout << "Found " << diff_count << " differences. Max diff: " << max_diff << std::endl;
        std::cout << "Note: Some difference is expected due to different computation orders." << std::endl;
    }

    // Free memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_temp);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O_cuda);
    free(h_O_cpu);

    return 0;
}
