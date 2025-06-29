#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA function
torch::Tensor feed_forward_forward_launcher(
    const torch::Tensor& input,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_bias,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc2_bias,
    const std::string& activation,
    float dropout_prob,
    cudaStream_t stream);

// PyTorch wrapper for the feed_forward forward function
torch::Tensor feed_forward_forward(
    torch::Tensor input,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias,
    std::string activation = "gelu",
    float dropout_prob = 0.0)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(fc1_weight.is_cuda(), "FC1 weight must be a CUDA tensor");
    TORCH_CHECK(fc1_bias.is_cuda() || fc1_bias.numel() == 0, "FC1 bias (if provided) must be a CUDA tensor");
    TORCH_CHECK(fc2_weight.is_cuda(), "FC2 weight must be a CUDA tensor");
    TORCH_CHECK(fc2_bias.is_cuda() || fc2_bias.numel() == 0, "FC2 bias (if provided) must be a CUDA tensor");
    
    // Get dimensions
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int d_model = input.size(2);
    int d_ff = fc1_weight.size(0);
    
    // Check dimensions
    TORCH_CHECK(fc1_weight.size(1) == d_model, 
                "FC1 weight input dimension mismatch: expected ", 
                d_model, " but got ", fc1_weight.size(1));
    TORCH_CHECK(fc1_bias.numel() == 0 || fc1_bias.numel() == d_ff,
                "FC1 bias has incompatible size: expected ",
                d_ff, " but got ", fc1_bias.numel());
    TORCH_CHECK(fc2_weight.size(0) == d_model, 
                "FC2 weight output dimension mismatch: expected ", 
                d_model, " but got ", fc2_weight.size(0));
    TORCH_CHECK(fc2_weight.size(1) == d_ff, 
                "FC2 weight input dimension mismatch: expected ", 
                d_ff, " but got ", fc2_weight.size(1));
    TORCH_CHECK(fc2_bias.numel() == 0 || fc2_bias.numel() == d_model,
                "FC2 bias has incompatible size: expected ",
                d_model, " but got ", fc2_bias.numel());
    
    // Check activation type
    TORCH_CHECK(activation == "gelu" || activation == "relu",
                "Activation must be 'gelu' or 'relu', but got '", activation, "'");
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    return feed_forward_forward_launcher(
        input,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        activation,
        dropout_prob,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("feed_forward_forward", &feed_forward_forward, "FeedForward forward (CUDA)",
          py::arg("input"),
          py::arg("fc1_weight"),
          py::arg("fc1_bias"),
          py::arg("fc2_weight"),
          py::arg("fc2_bias"),
          py::arg("activation") = "gelu",
          py::arg("dropout_prob") = 0.0);
} 