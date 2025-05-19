import torch
import triton 
import triton.language as tl

@triton.jit
def layernorm_kernel(
    A_ptr, B_ptr,
    rows, cols, stride, epsilon,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Base pointers for the current row
    row_ptr = A_ptr + row_idx * stride + col_offsets
    mask = col_offsets < cols

    # Lead data
    a_vals = tl.load(row_ptr, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(a_vals, axis=0) / cols

    # Compute varience
    diff = a_vals - mean
    var = tl.sum(diff * diff, axis=0) / cols
    std = tl.sqrt(var + epsilon)

    # Normilize and store
    norm_vals = (a_vals - mean) / std
    out_ptrs = B_ptr + row_idx * stride + col_offsets
    tl.store(out_ptrs, norm_vals, mask=mask)

def layernorm():
    rows, cols = 1024, 1024
    BLOCK_SIZE = 128
    epsilon = 1e-7
    A = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    B = torch.empty_like(A)

    # Triton kernel for LayerNorm
    grid = (rows, )
    layernorm_kernel[grid](
        A, B,
        rows=rows,
        cols=cols,
        stride=A.stride(0),
        epsilon=1e-5,
        BLOCK_SIZE=BLOCK_SIZE
    )

    print("Input Matrix A:")
    print(A.cpu().numpy())
    print("LayerNorm Output Matrix B:")
    print(B.cpu().numpy())

if __name__ == "__main__":
    layernorm()