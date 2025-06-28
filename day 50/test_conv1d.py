import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from whisper_cuda.conv1d import Conv1d

def test_conv1d_correctness():
    """Test the correctness of our CUDA Conv1d implementation against PyTorch's."""
    print("Testing Conv1d correctness...")
    
    # Set up test parameters
    batch_size = 4
    in_channels = 384
    in_width = 1500
    out_channels = 384
    kernel_size = 3
    stride = 1
    padding = 1
    
    # Create input tensor in half precision
    x = torch.randn(batch_size, in_channels, in_width, 
                    dtype=torch.float16, device='cuda', requires_grad=False)
    
    # Create our optimized Conv1d module
    cuda_conv1d = Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    ).cuda().half()
    
    # Create standard PyTorch Conv1d module with same parameters
    torch_conv1d = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    ).cuda().half()
    
    # Copy weights from PyTorch module to our module to ensure identical parameters
    with torch.no_grad():
        cuda_conv1d.weight.copy_(torch_conv1d.weight)
        cuda_conv1d.bias.copy_(torch_conv1d.bias)
    
    # Forward pass through both modules
    with torch.no_grad():
        cuda_output = cuda_conv1d(x)
        torch_output = torch_conv1d(x)
    
    # Check that outputs match
    rel_diff = torch.abs(cuda_output - torch_output) / (torch.abs(torch_output) + 1e-7)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"Max relative difference: {max_rel_diff:.6f}")
    print(f"Mean relative difference: {mean_rel_diff:.6f}")
    
    # Test passes if maximum relative difference is less than 0.1%
    passed = max_rel_diff < 1e-3
    print(f"Test {'PASSED' if passed else 'FAILED'}")
    
    return passed

def benchmark_conv1d(batch_sizes, widths, repeats=10):
    """Benchmark CUDA Conv1d against PyTorch's implementation."""
    print("\nBenchmarking Conv1d performance...")
    
    # Fixed parameters
    in_channels = 384
    out_channels = 384
    kernel_size = 3
    stride = 1
    padding = 1
    
    results = {
        'pytorch': [],
        'cuda': [],
        'speedup': [],
        'batch_sizes': batch_sizes,
        'widths': widths
    }
    
    for batch_size in batch_sizes:
        for width in widths:
            print(f"\nBenchmarking with batch_size={batch_size}, width={width}")
            
            # Create input tensor
            x = torch.randn(batch_size, in_channels, width, 
                           dtype=torch.float16, device='cuda')
            
            # Create modules
            cuda_conv1d = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ).cuda().half()
            
            torch_conv1d = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ).cuda().half()
            
            # Warm-up runs
            for _ in range(5):
                _ = cuda_conv1d(x)
                _ = torch_conv1d(x)
            
            # Time PyTorch implementation
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                _ = torch_conv1d(x)
                torch.cuda.synchronize()
            torch_time = (time.time() - start) / repeats
            
            # Time our CUDA implementation
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                _ = cuda_conv1d(x)
                torch.cuda.synchronize()
            cuda_time = (time.time() - start) / repeats
            
            # Calculate speedup
            speedup = torch_time / cuda_time
            
            # Store results
            results['pytorch'].append(torch_time * 1000)  # ms
            results['cuda'].append(cuda_time * 1000)      # ms
            results['speedup'].append(speedup)
            
            print(f"PyTorch: {torch_time*1000:.3f} ms")
            print(f"CUDA:    {cuda_time*1000:.3f} ms")
            print(f"Speedup: {speedup:.2f}x")
    
    return results

def plot_results(results):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 6))
    
    # Format labels for x-axis
    x_labels = [f"B={b}, W={w}" for b, w in zip(results['batch_sizes'], results['widths'])]
    x = np.arange(len(x_labels))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, results['pytorch'], width, label='PyTorch')
    plt.bar(x + width/2, results['cuda'], width, label='Our CUDA')
    
    # Add text for speedup
    for i, speedup in enumerate(results['speedup']):
        plt.annotate(f"{speedup:.2f}x", 
                    xy=(i, max(results['pytorch'][i], results['cuda'][i])),
                    ha='center', va='bottom')
    
    # Customize plot
    plt.xlabel('Batch Size & Input Width')
    plt.ylabel('Time (ms)')
    plt.title('Conv1d Performance Comparison')
    plt.xticks(x, x_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig('conv1d_benchmark.png')
    plt.close()
    
    print("\nBenchmark plot saved as 'conv1d_benchmark.png'")

def test_different_kernel_sizes():
    """Test Conv1d with different kernel sizes."""
    print("\nTesting different kernel sizes...")
    
    batch_size = 4
    in_channels = 128
    in_width = 1000
    out_channels = 128
    
    # Create input tensor
    x = torch.randn(batch_size, in_channels, in_width,
                   dtype=torch.float16, device='cuda')
    
    for kernel_size in [3, 5, 7]:
        print(f"\nTesting kernel_size={kernel_size}")
        padding = kernel_size // 2  # Same padding
        
        # Create modules
        cuda_conv1d = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ).cuda().half()
        
        torch_conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ).cuda().half()
        
        # Copy weights
        with torch.no_grad():
            cuda_conv1d.weight.copy_(torch_conv1d.weight)
            cuda_conv1d.bias.copy_(torch_conv1d.bias)
        
        # Forward pass
        with torch.no_grad():
            cuda_output = cuda_conv1d(x)
            torch_output = torch_conv1d(x)
        
        # Check outputs
        rel_diff = torch.abs(cuda_output - torch_output) / (torch.abs(torch_output) + 1e-7)
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        print(f"Max relative difference: {max_rel_diff:.6f}")
        print(f"Mean relative difference: {mean_rel_diff:.6f}")
        
        # Test passes if maximum relative difference is less than 0.1%
        passed = max_rel_diff < 1e-3
        print(f"Test {'PASSED' if passed else 'FAILED'}")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping tests.")
        exit()
    
    # Run correctness tests
    test_conv1d_correctness()
    test_different_kernel_sizes()
    
    # Run benchmarks for different sizes
    batch_sizes = [1, 2, 4, 8, 8, 8]
    widths = [1500, 1500, 1500, 1500, 3000, 6000]
    
    results = benchmark_conv1d(batch_sizes, widths)
    plot_results(results) 