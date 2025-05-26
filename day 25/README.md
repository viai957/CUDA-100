# Multi-Head Attention Benchmark

This project compares two implementations of the Multi-Head Attention mechanism:

1. PyTorch's native implementation (`attention.py`)
2. A custom CUDA implementation (`Attention.cu`) wrapped in a PyTorch extension

## Files

- `attention.py` - PyTorch implementation of Multi-Head Attention
- `Attention.cu` - Original CUDA implementation of Multi-Head Attention
- `attention_cuda.cpp` - C++ interface for the PyTorch extension
- `attention_cuda_kernel.cu` - CUDA kernel for the PyTorch extension
- `setup.py` - Setup script for building the extension
- `attention_test.py` - Simple test comparing both implementations
- `attention_benchmark.py` - Comprehensive benchmark of both implementations

## Installation

Make sure you have PyTorch installed with CUDA support. Then, you can either:

1. Build the extension using setup.py:
   ```
   python setup.py install
   ```

2. Let the scripts JIT-compile the extension (slower first run but no installation required)

## Running the Benchmarks

```bash
# Run the simple test
python attention_test.py

# Run the comprehensive benchmark
python attention_benchmark.py
```

The comprehensive benchmark will:
1. Verify that both implementations produce similar results
2. Measure performance with varying sequence lengths
3. Measure performance with varying numbers of attention heads
4. Generate plots with the results (saved as PNG files)

## Implementation Details

The CUDA implementation follows the standard multi-head attention algorithm:

1. Project queries, keys, and values using linear layers
2. Compute attention scores: Q * K^T / sqrt(d_k)
3. Apply softmax to get attention weights
4. Apply optional dropout
5. Compute output: attention_weights * V
6. Project output with a final linear layer

Both implementations follow the same algorithm, but the CUDA version is optimized for GPU execution with custom CUDA kernels. 