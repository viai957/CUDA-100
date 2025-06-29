#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA function
void layernorm_forward_launcher(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& mean,
    torch::Tensor& inv_variance,
    double epsilon,
    cudaStream_t stream);

// PyTorch wrapper for the layernorm forward function
std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double epsilon = 1e-5)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda() || bias.numel() == 0, "Bias (if provided) must be a CUDA tensor");
    
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [batch_size*seq_len, hidden_size]");
    
    // Get dimensions
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    
    // Check dimensions
    TORCH_CHECK(weight.numel() == hidden_size, 
                "Weight has incompatible size: expected ", 
                hidden_size, " but got ", weight.numel());
    
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == hidden_size,
                "Bias has incompatible size: expected ",
                hidden_size, " but got ", bias.numel());
    
    // Create output tensors
    auto output = torch::empty_like(input);
    auto mean = torch::empty({batch_size}, input.options());
    auto inv_variance = torch::empty({batch_size}, input.options());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    layernorm_forward_launcher(
        input,
        weight,
        bias,
        output,
        mean,
        inv_variance,
        epsilon,
        stream
    );
    
    return {output, mean, inv_variance};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("epsilon") = 1e-5);
} 