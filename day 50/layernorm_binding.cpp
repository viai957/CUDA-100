#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>

// Forward declaration of the CUDA function
void layer_norm_forward_launcher(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    half* mean,
    half* inv_variance,
    int batch_size,
    int hidden_size,
    double epsilon,
    cudaStream_t stream = nullptr);

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
    
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "Input must be half precision");
    TORCH_CHECK(weight.scalar_type() == torch::kHalf, "Weight must be half precision");
    TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == torch::kHalf, "Bias must be half precision");
    
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
    auto options = torch::TensorOptions()
        .dtype(torch::kHalf)
        .device(input.device())
        .requires_grad(false);
    
    torch::Tensor output = torch::empty_like(input, options);
    torch::Tensor mean = torch::empty({batch_size}, options);
    torch::Tensor inv_variance = torch::empty({batch_size}, options);
    
    // Get raw pointers
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr());
    const half* weight_ptr = reinterpret_cast<const half*>(weight.data_ptr());
    const half* bias_ptr = bias.numel() > 0 ? reinterpret_cast<const half*>(bias.data_ptr()) : nullptr;
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr());
    half* mean_ptr = reinterpret_cast<half*>(mean.data_ptr());
    half* inv_variance_ptr = reinterpret_cast<half*>(inv_variance.data_ptr());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    layer_norm_forward_launcher(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        mean_ptr,
        inv_variance_ptr,
        batch_size,
        hidden_size,
        epsilon,
        stream
    );
    
    return {output, mean, inv_variance};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "Custom CUDA LayerNorm forward (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor(),
          py::arg("epsilon") = 1e-5);
} 