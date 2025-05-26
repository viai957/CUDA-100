#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor multi_head_attention_cuda_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    int h,
    float dropout_prob);

// C++ interface
torch::Tensor multi_head_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    int h,
    float dropout_prob) {
    
    // Input validation
    TORCH_CHECK(Q.dim() == 3, "Query tensor must be 3D (batch_size, seq_len, d_model)");
    TORCH_CHECK(K.dim() == 3, "Key tensor must be 3D (batch_size, seq_len, d_model)");
    TORCH_CHECK(V.dim() == 3, "Value tensor must be 3D (batch_size, seq_len, d_model)");
    
    TORCH_CHECK(Q.size(0) == K.size(0) && K.size(0) == V.size(0), 
                "Batch sizes of Query, Key, and Value tensors must match");
    TORCH_CHECK(Q.size(1) == K.size(1) && K.size(1) == V.size(1),
                "Sequence lengths of Query, Key, and Value tensors must match");
    TORCH_CHECK(Q.size(2) == K.size(2) && K.size(2) == V.size(2),
                "Feature dimensions of Query, Key, and Value tensors must match");
    
    TORCH_CHECK(Q.size(2) % h == 0, "Feature dimension must be divisible by number of heads");
    
    TORCH_CHECK(Q.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(K.is_cuda(), "Key tensor must be on CUDA device");
    TORCH_CHECK(V.is_cuda(), "Value tensor must be on CUDA device");
    
    return multi_head_attention_cuda_forward(Q, K, V, h, dropout_prob);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_head_attention", &multi_head_attention, "Multi-head attention forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("h"), py::arg("dropout_prob")=0.0f);
} 