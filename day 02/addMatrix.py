import triton 
import torch
import triton.language as tl 
import time

@triton.jit
def addMatrix_efficient(Matrix_A, Matrix_B, Matrix_C, numel, BLOCK_SIZE: tl.constexpr):
    # Use simple 1D indexing for element-wise operations
    pid = tl.program_id(0)
    
    # Calculate the offset for the current thread block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load the values from the matrices
    a = tl.load(Matrix_A + offsets, mask=mask)
    b = tl.load(Matrix_B + offsets, mask=mask)
    c = a + b
    
    # Store the result
    tl.store(Matrix_C + offsets, c, mask=mask)

def test_addMatrix():
    # Define the size of the matrices
    N = 12400000
    M = 12400000
    BLOCK_SIZE = 128  # Larger block size for better performance
    numel = N * M
    
    # Create random matrices A and B
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    B = torch.randn(M, N, device="cuda", dtype=torch.float32)
    C = torch.zeros_like(A, device="cuda", dtype=torch.float32)
    
    A_flat = A.contiguous().view(-1)  # Ensure contiguous memory
    B_flat = B.contiguous().view(-1)
    C_flat = C.contiguous().view(-1)
    
    # Warm-up run to compile the kernel (not timed)
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    addMatrix_efficient[grid](A_flat, B_flat, C_flat, numel, BLOCK_SIZE=BLOCK_SIZE)
    
    # Triton timing (multiple runs for better measurement)
    n_runs = 10
    torch.cuda.synchronize()
    start_triton = time.time()
    for _ in range(n_runs):
        addMatrix_efficient[grid](A_flat, B_flat, C_flat, numel, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_triton) * 1000 / n_runs  # Average ms per run
    C = C_flat.view(M, N)
    
    # PyTorch timing (multiple runs)
    torch.cuda.synchronize()
    start_torch = time.time()
    for _ in range(n_runs):
        expected_C = A + B
    torch.cuda.synchronize()
    torch_time = (time.time() - start_torch) * 1000 / n_runs  # Average ms per run
    
    # Verify the result
    assert torch.allclose(C, expected_C), "Result does not match expected output!"
    print(f"Triton kernel time: {triton_time:.2f} ms")
    print(f"PyTorch time: {torch_time:.2f} ms")
    print(f"Triton - PyTorch time difference: {triton_time - torch_time:.2f} ms")
    print("Test passed!")

test_addMatrix()