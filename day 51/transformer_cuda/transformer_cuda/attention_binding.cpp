#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA function
std::tuple<torch::Tensor, torch::Tensor> attention_forward_launcher(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& mask,
    bool is_causal,
    float dropout_prob,
    bool need_weights,
    cudaStream_t stream);

// PyTorch wrapper for the attention forward function
std::tuple<torch::Tensor, torch::Tensor> attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask,
    bool is_causal,
    float dropout_prob = 0.0,
    bool need_weights = false)
{
    // Input validation
    TORCH_CHECK(query.is_cuda(), "Query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "Key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "Value must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda() || mask.numel() == 0, "Mask (if provided) must be a CUDA tensor");
    
    TORCH_CHECK(query.dim() == 4, "Query must be a 4D tensor [batch_size, tgt_len, num_heads, head_dim]");
    TORCH_CHECK(key.dim() == 4, "Key must be a 4D tensor [batch_size, src_len, num_heads, head_dim]");
    TORCH_CHECK(value.dim() == 4, "Value must be a 4D tensor [batch_size, src_len, num_heads, head_dim]");
    
    // Get dimensions
    int batch_size = query.size(0);
    int tgt_len = query.size(1);
    int num_heads = query.size(2);
    int head_dim = query.size(3);
    int src_len = key.size(1);
    
    // Check dimensions
    TORCH_CHECK(key.size(0) == batch_size, 
                "Key batch size mismatch: expected ", 
                batch_size, " but got ", key.size(0));
    TORCH_CHECK(value.size(0) == batch_size, 
                "Value batch size mismatch: expected ", 
                batch_size, " but got ", value.size(0));
    TORCH_CHECK(key.size(1) == value.size(1), 
                "Key and value sequence length mismatch: expected ", 
                key.size(1), " but got ", value.size(1));
    TORCH_CHECK(key.size(2) == num_heads, 
                "Key heads mismatch: expected ", 
                num_heads, " but got ", key.size(2));
    TORCH_CHECK(value.size(2) == num_heads, 
                "Value heads mismatch: expected ", 
                num_heads, " but got ", value.size(2));
    TORCH_CHECK(key.size(3) == head_dim, 
                "Key head dimension mismatch: expected ", 
                head_dim, " but got ", key.size(3));
    TORCH_CHECK(value.size(3) == head_dim, 
                "Value head dimension mismatch: expected ", 
                head_dim, " but got ", value.size(3));
    
    // Check mask dimensions if provided
    if (mask.numel() > 0) {
        if (mask.dim() == 4) {
            TORCH_CHECK(mask.size(0) == batch_size, "Mask batch size mismatch");
            TORCH_CHECK(mask.size(2) == tgt_len || mask.size(2) == 1, "Mask target length mismatch");
            TORCH_CHECK(mask.size(3) == src_len || mask.size(3) == 1, "Mask source length mismatch");
        } else {
            TORCH_CHECK(false, "Mask must be a 4D tensor [batch_size, 1, tgt_len, src_len]");
        }
    }
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    return attention_forward_launcher(
        query,
        key,
        value,
        mask,
        is_causal,
        dropout_prob,
        need_weights,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "MultiHeadAttention forward (CUDA)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("mask"),
          py::arg("is_causal"),
          py::arg("dropout_prob") = 0.0,
          py::arg("need_weights") = false);
} 