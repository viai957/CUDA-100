import triton 
import triton.language as tl 
import torch
import time 
import numpy as np

@triton.jit
def vector_add(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    a = tl.load(A_ptr + offset, mask=mask)
    b = tl.load(B_ptr + offset, mask=mask)
    c = a + b
    tl.store(C_ptr + offset, c, mask=mask)

def test_vector_add():
    N = 1024
    A = torch.arange(0, N, dtype=torch.float32).cuda()
    B = torch.arange(0, N, dtype=torch.float32).cuda()
    C = torch.empty_like(A)

    # Allocate device tensors
    d_A = torch.tensor(A, device="cuda")
    d_B = torch.tensor(B, device="cuda")
    d_C = torch.empty_like(d_A)

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # Time the kernel
    start_time = time.time()
    vector_add[grid](d_A, d_B, d_C, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    elapsed_kernel_time = (time.time() - start_time) * 1000 # Convert to milliseconds
    print(f"Kernel execution time: {elapsed_kernel_time:.2f} ms")

    # Copy result back to host
    C = d_C.cpu()
    print("Result C:", C)

    # Verify the result
    start_check = time.time()
    expected_C = (A + B).cpu()
    assert torch.allclose(C, expected_C), "Result does not match expected output!"
    torch.cuda.synchronize()
    elapsed_check_time = (time.time() - start_check) * 1000 # Convert to milliseconds
    print(f"Verification time: {elapsed_check_time:.2f} ms")
    print("Test passed!")

if __name__ == "__main__":
    test_vector_add()
# This code implements a simple vector addition using Triton, a programming language for GPU computing.
