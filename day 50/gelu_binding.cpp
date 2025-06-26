#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>

// Forward declaration of the CUDA function
void gelu_forward_launcher(
    const half* input,
    half* output,
    int n,
    cudaStream_t stream = nullptr);

// PyTorch wrapper for the GELU forward function
torch::Tensor gelu_forward(torch::Tensor input)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "Input must be half precision");
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kHalf)
        .device(input.device())
        .requires_grad(false);
    
    torch::Tensor output = torch::empty_like(input, options);
    
    // Get raw pointers
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr());
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr());
    
    // Get total number of elements
    int n = input.numel();
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    gelu_forward_launcher(input_ptr, output_ptr, n, stream);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_forward", &gelu_forward, "Custom CUDA GELU forward (CUDA)",
          py::arg("input"));
} 