#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>

// Forward declaration of the CUDA function
void linear_forward_launcher(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream = nullptr);

// PyTorch wrapper for the linear forward function
torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias)
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
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    // Check dimensions
    TORCH_CHECK(weight.size(1) == in_features, 
                "Weight matrix has incompatible dimensions: expected ", 
                in_features, " input features but got ", weight.size(1));
    
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == out_features,
                "Bias must have size equal to out_features: expected ",
                out_features, " but got ", bias.numel());
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kHalf)
        .device(input.device())
        .requires_grad(false);
    
    torch::Tensor output = torch::empty({batch_size, out_features}, options);
    
    // Get raw pointers
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr());
    const half* weight_ptr = reinterpret_cast<const half*>(weight.data_ptr());
    const half* bias_ptr = bias.numel() > 0 ? reinterpret_cast<const half*>(bias.data_ptr()) : nullptr;
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    linear_forward_launcher(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_features,
        out_features,
        stream
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Custom CUDA Linear forward (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor());
} 