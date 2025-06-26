#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>

// Forward declarations of the CUDA functions
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
    cudaStream_t stream = nullptr);

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
    cudaStream_t stream = nullptr);

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
    cudaStream_t stream = nullptr);

// PyTorch wrapper for the scaled dot-product attention
torch::Tensor attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask,
    bool causal_mask)
{
    // Input validation
    TORCH_CHECK(query.is_cuda(), "Query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "Key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "Value must be a CUDA tensor");
    TORCH_CHECK(mask.numel() == 0 || mask.is_cuda(), "Mask (if provided) must be a CUDA tensor");
    
    TORCH_CHECK(query.scalar_type() == torch::kHalf, "Query must be half precision");
    TORCH_CHECK(key.scalar_type() == torch::kHalf, "Key must be half precision");
    TORCH_CHECK(value.scalar_type() == torch::kHalf, "Value must be half precision");
    TORCH_CHECK(mask.numel() == 0 || mask.scalar_type() == torch::kHalf, "Mask must be half precision");
    
    // Get dimensions
    int batch_size = query.size(0);
    int seq_len = query.size(1);
    int num_heads = query.size(2);
    int head_dim = query.size(3);
    
    // Check dimensions
    TORCH_CHECK(key.size(0) == batch_size && key.size(1) == seq_len && 
                key.size(2) == num_heads && key.size(3) == head_dim,
                "Key has incompatible dimensions");
    
    TORCH_CHECK(value.size(0) == batch_size && value.size(1) == seq_len && 
                value.size(2) == num_heads && value.size(3) == head_dim,
                "Value has incompatible dimensions");
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kHalf)
        .device(query.device())
        .requires_grad(false);
    
    torch::Tensor output = torch::empty({batch_size, seq_len, num_heads, head_dim}, options);
    
    // Get raw pointers
    const half* query_ptr = reinterpret_cast<const half*>(query.data_ptr());
    const half* key_ptr = reinterpret_cast<const half*>(key.data_ptr());
    const half* value_ptr = reinterpret_cast<const half*>(value.data_ptr());
    const half* mask_ptr = mask.numel() > 0 ? reinterpret_cast<const half*>(mask.data_ptr()) : nullptr;
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    attention_forward_launcher(
        query_ptr,
        key_ptr,
        value_ptr,
        mask_ptr,
        output_ptr,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        causal_mask,
        stream
    );
    
    return output;
}

// PyTorch wrapper for the QKV projection
std::vector<torch::Tensor> qkv_projection(
    torch::Tensor input,
    torch::Tensor weight_q,
    torch::Tensor weight_k,
    torch::Tensor weight_v,
    torch::Tensor bias_q,
    torch::Tensor bias_k,
    torch::Tensor bias_v)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight_q.is_cuda(), "Weight_q must be a CUDA tensor");
    TORCH_CHECK(weight_k.is_cuda(), "Weight_k must be a CUDA tensor");
    TORCH_CHECK(weight_v.is_cuda(), "Weight_v must be a CUDA tensor");
    TORCH_CHECK(bias_q.numel() == 0 || bias_q.is_cuda(), "Bias_q (if provided) must be a CUDA tensor");
    TORCH_CHECK(bias_k.numel() == 0 || bias_k.is_cuda(), "Bias_k (if provided) must be a CUDA tensor");
    TORCH_CHECK(bias_v.numel() == 0 || bias_v.is_cuda(), "Bias_v (if provided) must be a CUDA tensor");
    
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "Input must be half precision");
    TORCH_CHECK(weight_q.scalar_type() == torch::kHalf, "Weight_q must be half precision");
    TORCH_CHECK(weight_k.scalar_type() == torch::kHalf, "Weight_k must be half precision");
    TORCH_CHECK(weight_v.scalar_type() == torch::kHalf, "Weight_v must be half precision");
    TORCH_CHECK(bias_q.numel() == 0 || bias_q.scalar_type() == torch::kHalf, "Bias_q must be half precision");
    TORCH_CHECK(bias_k.numel() == 0 || bias_k.scalar_type() == torch::kHalf, "Bias_k must be half precision");
    TORCH_CHECK(bias_v.numel() == 0 || bias_v.scalar_type() == torch::kHalf, "Bias_v must be half precision");
    
    // Get dimensions
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int embed_dim = input.size(2);
    int num_heads = weight_q.size(1) / 64;  // Assuming head_dim = 64
    int head_dim = 64;
    
    // Create output tensors
    auto options = torch::TensorOptions()
        .dtype(torch::kHalf)
        .device(input.device())
        .requires_grad(false);
    
    torch::Tensor query = torch::empty({batch_size, seq_len, num_heads, head_dim}, options);
    torch::Tensor key = torch::empty({batch_size, seq_len, num_heads, head_dim}, options);
    torch::Tensor value = torch::empty({batch_size, seq_len, num_heads, head_dim}, options);
    
    // Get raw pointers
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr());
    const half* weight_q_ptr = reinterpret_cast<const half*>(weight_q.data_ptr());
    const half* weight_k_ptr = reinterpret_cast<const half*>(weight_k.data_ptr());
    const half* weight_v_ptr = reinterpret_cast<const half*>(weight_v.data_ptr());
    const half* bias_q_ptr = bias_q.numel() > 0 ? reinterpret_cast<const half*>(bias_q.data_ptr()) : nullptr;
    const half* bias_k_ptr = bias_k.numel() > 0 ? reinterpret_cast<const half*>(bias_k.data_ptr()) : nullptr;
    const half* bias_v_ptr = bias_v.numel() > 0 ? reinterpret_cast<const half*>(bias_v.data_ptr()) : nullptr;
    half* query_ptr = reinterpret_cast<half*>(query.data_ptr());
    half* key_ptr = reinterpret_cast<half*>(key.data_ptr());
    half* value_ptr = reinterpret_cast<half*>(value.data_ptr());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    qkv_projection_launcher(
        input_ptr,
        weight_q_ptr,
        weight_k_ptr,
        weight_v_ptr,
        bias_q_ptr,
        bias_k_ptr,
        bias_v_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        batch_size,
        seq_len,
        embed_dim,
        num_heads,
        head_dim,
        stream
    );
    
    return {query, key, value};
}

// PyTorch wrapper for the output projection
torch::Tensor output_projection(
    torch::Tensor attention_output,
    torch::Tensor weight_output,
    torch::Tensor bias_output)
{
    // Input validation
    TORCH_CHECK(attention_output.is_cuda(), "Attention output must be a CUDA tensor");
    TORCH_CHECK(weight_output.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias_output.numel() == 0 || bias_output.is_cuda(), "Bias (if provided) must be a CUDA tensor");
    
    TORCH_CHECK(attention_output.scalar_type() == torch::kHalf, "Attention output must be half precision");
    TORCH_CHECK(weight_output.scalar_type() == torch::kHalf, "Weight must be half precision");
    TORCH_CHECK(bias_output.numel() == 0 || bias_output.scalar_type() == torch::kHalf, "Bias must be half precision");
    
    // Get dimensions
    int batch_size = attention_output.size(0);
    int seq_len = attention_output.size(1);
    int num_heads = attention_output.size(2);
    int head_dim = attention_output.size(3);
    int embed_dim = weight_output.size(1);
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kHalf)
        .device(attention_output.device())
        .requires_grad(false);
    
    torch::Tensor output = torch::empty({batch_size, seq_len, embed_dim}, options);
    
    // Get raw pointers
    const half* attention_output_ptr = reinterpret_cast<const half*>(attention_output.data_ptr());
    const half* weight_output_ptr = reinterpret_cast<const half*>(weight_output.data_ptr());
    const half* bias_output_ptr = bias_output.numel() > 0 ? reinterpret_cast<const half*>(bias_output.data_ptr()) : nullptr;
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    output_projection_launcher(
        attention_output_ptr,
        weight_output_ptr,
        bias_output_ptr,
        output_ptr,
        batch_size,
        seq_len,
        embed_dim,
        num_heads,
        head_dim,
        stream
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "Custom CUDA Attention forward (CUDA)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("mask") = torch::Tensor(),
          py::arg("causal_mask") = false);
    
    m.def("qkv_projection", &qkv_projection, "Custom CUDA QKV projection (CUDA)",
          py::arg("input"),
          py::arg("weight_q"),
          py::arg("weight_k"),
          py::arg("weight_v"),
          py::arg("bias_q") = torch::Tensor(),
          py::arg("bias_k") = torch::Tensor(),
          py::arg("bias_v") = torch::Tensor());
    
    m.def("output_projection", &output_projection, "Custom CUDA output projection (CUDA)",
          py::arg("attention_output"),
          py::arg("weight_output"),
          py::arg("bias_output") = torch::Tensor());
} 