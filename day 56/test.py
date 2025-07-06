import torch
from torch.utils.cpp_extension import load
import time
from liger_kernel.ops import rms_norm

def rms_norm(tensor):
    return tensor / torch.sqrt(torch.mean(tensor ** 2))

sources = ["binding.cpp", "RMS.cu", "RMSBetter.cu"]
RMS = load("RMS", sources=sources, verbose=True)
print("Custom CUDA extension loaded.")

tensor_sizes = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]

for tensor_size in tensor_sizes:
    print("=" * 50)
    print("Input Size: ", tensor_size)
    print("=" * 50)
    input_tensor = torch.randn(tensor_size, device='cuda')

    # PyTorch RMS time and result
    pytorch_time = 0
    result_pytorch = None
    for _ in range(5):
        start_time = time.time()
        result_pytorch = rms_norm(input_tensor)
        pytorch_time += time.time() - start_time
    print(f"PyTorch RMS time: {pytorch_time / 6:.6f} seconds")

    # Custom kernel time and result
    custom_time = 0
    result_custom = None
    for _ in range(5):
        start_time = time.time()
        result_custom = RMS.RMSV2(input_tensor)
        custom_time += time.time() - start_time
    print(f"Custom kernel time: {custom_time / 6:.6f} seconds")

    # Liger kernel time and result
    liger_time = 0
    result_liger = None
    for _ in range(5):
        start_time = time.time()
        result_liger = rms_norm(input_tensor)
        liger_time += time.time() - start_time
    print(f"Liger kernel time: {liger_time / 6:.6f} seconds")

    # Checking if the results are the same
    pytorch_custom_diff = torch.max(torch.abs(result_pytorch - result_custom))
    pytorch_liger_diff = torch.max(torch.abs(result_pytorch - result_liger))

    print(f"Max difference between PyTorch and Custom kernel: {pytorch_custom_diff.item():.6f}")
    print(f"Max difference between PyTorch and Liger kernel: {pytorch_liger_diff.item():.6f}")

    # Check if they are numerically close (within tolerance)
    are_pytorch_custom_close = torch.allclose(result_pytorch, result_custom, atol=1)  # You can adjust the tolerance
    are_pytorch_liger_close = torch.allclose(result_pytorch, result_liger, atol=1)  # You can adjust the tolerance

    if are_pytorch_custom_close:
        print("PyTorch and Custom kernel results are the same!")
    else:
        print("PyTorch and Custom kernel results are different.")

    if are_pytorch_liger_close:
        print("PyTorch and Liger kernel results are the same!")
    else:
        print("PyTorch and Liger kernel results are different.")

    print("=" * 50 + "\n")
