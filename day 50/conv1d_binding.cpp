#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>

// Forward declaration of CUDA function
void conv1d_forward_launcher(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int batch_size,
    int in_channels,
    int in_width,
    int out_channels,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream = nullptr);

// Helper to calculate output width
int calculate_out_width(int in_width, int kernel_size, int stride, int padding);

// PyTorch wrapper for the Conv1d forward pass
torch::Tensor conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding) {
    // Ensure inputs are contiguous and in half precision (FP16)
    input = input.contiguous();
    weight = weight.contiguous();
    
    // Check inputs
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be float16 (half precision)");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "Weight must be float16 (half precision)");
    
    // Check if bias is provided
    bool has_bias = bias.defined();
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat16, "Bias must be float16 (half precision)");
        bias = bias.contiguous();
    }
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_width = input.size(2);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Verify weight dimensions
    TORCH_CHECK(weight.size(1) == in_channels, 
                "Weight in_channels dimension must match input channels");
    
    // Calculate output dimensions
    int out_width = calculate_out_width(in_width, kernel_size, stride, padding);
    
    // Create output tensor
    auto output = torch::empty({batch_size, out_channels, out_width}, 
                               torch::dtype(torch::kFloat16).device(input.device()));
    
    // Get raw pointers for CUDA kernel
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr());
    const half* weight_ptr = reinterpret_cast<const half*>(weight.data_ptr());
    const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr()) : nullptr;
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    conv1d_forward_launcher(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        in_width,
        out_channels,
        out_width,
        kernel_size,
        stride,
        padding,
        stream
    );
    
    return output;
}

// Binding function for PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv1d_forward, "Conv1d forward (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor(),
          py::arg("stride") = 1,
          py::arg("padding") = 0);
} 