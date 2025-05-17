import triton
import triton.language as tl
import torch
import time

# Constants
TILE_DIM = 32
BLOCK_ROWS = 8
NUM_REPS = 100

# Simple copy kernel
@triton.jit
def copy_kernel(
    output_ptr, input_ptr,
    stride_output_row, stride_output_col,
    stride_input_row, stride_input_col,
    n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Get block indices (2D grid)
    block_idx_x = pid % (n_cols // TILE_DIM)
    block_idx_y = pid // (n_cols // TILE_DIM)
    
    # Get thread indices
    thread_idx_x = tl.arange(0, BLOCK_SIZE) % TILE_DIM
    thread_idx_y = tl.arange(0, BLOCK_SIZE) // TILE_DIM
    
    # Calculate global indices
    row = block_idx_y * TILE_DIM + thread_idx_y
    col = block_idx_x * TILE_DIM + thread_idx_x
    
    # Bounds checking
    mask = (row < n_rows) & (col < n_cols)
    
    # Calculate memory offsets
    in_offset = row * stride_input_row + col * stride_input_col
    out_offset = row * stride_output_row + col * stride_output_col
    
    # Load and store
    x = tl.load(input_ptr + in_offset, mask=mask)
    tl.store(output_ptr + out_offset, x, mask=mask)

# Naive transpose kernel
@triton.jit
def transpose_naive_kernel(
    output_ptr, input_ptr,
    stride_output_row, stride_output_col,
    stride_input_row, stride_input_col,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Get block indices (2D grid)
    block_idx_x = pid % (n_cols // TILE_DIM)
    block_idx_y = pid // (n_cols // TILE_DIM)
    
    # Get thread indices
    thread_idx_x = tl.arange(0, BLOCK_SIZE) % TILE_DIM
    thread_idx_y = tl.arange(0, BLOCK_SIZE) // TILE_DIM
    
    # Calculate global indices
    row_in = block_idx_y * TILE_DIM + thread_idx_y
    col_in = block_idx_x * TILE_DIM + thread_idx_x
    
    # Transpose indices
    row_out = block_idx_x * TILE_DIM + thread_idx_x
    col_out = block_idx_y * TILE_DIM + thread_idx_y
    
    # Bounds checking
    mask_in = (row_in < n_rows) & (col_in < n_cols)
    mask_out = (row_out < n_cols) & (col_out < n_rows)
    
    # Calculate memory offsets
    in_offset = row_in * stride_input_row + col_in * stride_input_col
    out_offset = row_out * stride_output_row + col_out * stride_output_col
    
    # Load and store with transpose
    x = tl.load(input_ptr + in_offset, mask=mask_in)
    tl.store(output_ptr + out_offset, x, mask=mask_out)

# Coalesced transpose kernel using shared memory
@triton.jit
def transpose_coalesced_kernel(
    output_ptr, input_ptr,
    stride_output_row, stride_output_col,
    stride_input_row, stride_input_col,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Get block indices (2D grid)
    block_idx_x = pid % (n_cols // TILE_DIM)
    block_idx_y = pid // (n_cols // TILE_DIM)
    
    # Declare shared memory
    tile = tl.zeros([TILE_DIM, TILE_DIM], dtype=tl.float32)
    
    # Get thread indices
    thread_idx_x = tl.arange(0, BLOCK_SIZE) % TILE_DIM
    thread_idx_y = tl.arange(0, BLOCK_SIZE) // TILE_DIM
    
    # Calculate global indices for loading
    row_in = block_idx_y * TILE_DIM + thread_idx_y
    col_in = block_idx_x * TILE_DIM + thread_idx_x
    
    # Bounds checking for loads
    mask_in = (row_in < n_rows) & (col_in < n_cols)
    
    # Load into shared memory with coalesced read
    for j in range(0, TILE_DIM, BLOCK_ROWS):
        in_row = row_in + j
        in_mask = (in_row < n_rows) & (col_in < n_cols)
        in_offset = in_row * stride_input_row + col_in * stride_input_col
        x = tl.load(input_ptr + in_offset, mask=in_mask)
        tile[thread_idx_y + j, thread_idx_x] = x
    
    tl.debug_barrier()  # Synchronize threads
    
    # Calculate global indices for storing (transposed)
    row_out = block_idx_x * TILE_DIM + thread_idx_y
    col_out = block_idx_y * TILE_DIM + thread_idx_x
    
    # Bounds checking for stores
    mask_out = (row_out < n_cols) & (col_out < n_rows)
    
    # Store from shared memory with transposed indices
    for j in range(0, TILE_DIM, BLOCK_ROWS):
        out_row = row_out + j
        out_mask = (out_row < n_cols) & (col_out < n_rows)
        out_offset = out_row * stride_output_row + col_out * stride_output_col
        x = tile[thread_idx_x, thread_idx_y + j]
        tl.store(output_ptr + out_offset, x, mask=out_mask)

# No bank conflicts transpose kernel with shared memory padding
@triton.jit
def transpose_no_bank_conflicts_kernel(
    output_ptr, input_ptr,
    stride_output_row, stride_output_col,
    stride_input_row, stride_input_col,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Get block indices (2D grid)
    block_idx_x = pid % (n_cols // TILE_DIM)
    block_idx_y = pid // (n_cols // TILE_DIM)
    
    # Declare shared memory with padding to avoid bank conflicts
    # Using TILE_DIM+1 as the column dimension to avoid bank conflicts
    tile = tl.zeros([TILE_DIM, TILE_DIM+1], dtype=tl.float32)
    
    # Get thread indices
    thread_idx_x = tl.arange(0, BLOCK_SIZE) % TILE_DIM
    thread_idx_y = tl.arange(0, BLOCK_SIZE) // TILE_DIM
    
    # Calculate global indices for loading
    row_in = block_idx_y * TILE_DIM + thread_idx_y
    col_in = block_idx_x * TILE_DIM + thread_idx_x
    
    # Bounds checking for loads
    mask_in = (row_in < n_rows) & (col_in < n_cols)
    
    # Load into shared memory with coalesced read
    for j in range(0, TILE_DIM, BLOCK_ROWS):
        in_row = row_in + j
        in_mask = (in_row < n_rows) & (col_in < n_cols)
        in_offset = in_row * stride_input_row + col_in * stride_input_col
        x = tl.load(input_ptr + in_offset, mask=in_mask)
        tile[thread_idx_y + j, thread_idx_x] = x
    
    tl.debug_barrier()  # Synchronize threads
    
    # Calculate global indices for storing (transposed)
    row_out = block_idx_x * TILE_DIM + thread_idx_y
    col_out = block_idx_y * TILE_DIM + thread_idx_x
    
    # Bounds checking for stores
    mask_out = (row_out < n_cols) & (col_out < n_rows)
    
    # Store from shared memory with transposed indices
    for j in range(0, TILE_DIM, BLOCK_ROWS):
        out_row = row_out + j
        out_mask = (out_row < n_cols) & (col_out < n_rows)
        out_offset = out_row * stride_output_row + col_out * stride_output_col
        x = tile[thread_idx_x, thread_idx_y + j]
        tl.store(output_ptr + out_offset, x, mask=out_mask)

# Wrapper functions to launch the kernels
def copy(input_tensor, output_tensor):
    M, N = input_tensor.shape
    grid = (M * N) // (TILE_DIM * BLOCK_ROWS)
    copy_kernel[grid](
        output_tensor, input_tensor,
        output_tensor.stride(0), output_tensor.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        M, N,
        BLOCK_SIZE=TILE_DIM * BLOCK_ROWS
    )
    return output_tensor

def transpose_naive(input_tensor, output_tensor):
    M, N = input_tensor.shape
    grid = (M * N) // (TILE_DIM * BLOCK_ROWS)
    transpose_naive_kernel[grid](
        output_tensor, input_tensor,
        output_tensor.stride(0), output_tensor.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        M, N,
        BLOCK_SIZE=TILE_DIM * BLOCK_ROWS
    )
    return output_tensor

def transpose_coalesced(input_tensor, output_tensor):
    M, N = input_tensor.shape
    grid = (M * N) // (TILE_DIM * BLOCK_ROWS)
    transpose_coalesced_kernel[grid](
        output_tensor, input_tensor,
        output_tensor.stride(0), output_tensor.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        M, N,
        BLOCK_SIZE=TILE_DIM * BLOCK_ROWS
    )
    return output_tensor

def transpose_no_bank_conflicts(input_tensor, output_tensor):
    M, N = input_tensor.shape
    grid = (M * N) // (TILE_DIM * BLOCK_ROWS)
    transpose_no_bank_conflicts_kernel[grid](
        output_tensor, input_tensor,
        output_tensor.stride(0), output_tensor.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        M, N,
        BLOCK_SIZE=TILE_DIM * BLOCK_ROWS
    )
    return output_tensor

def benchmark():
    # Create input matrices
    nx, ny = 1024, 1024
    input_tensor = torch.arange(nx * ny, dtype=torch.float32).reshape(nx, ny).cuda()
    output_tensor = torch.zeros((nx, ny), dtype=torch.float32).cuda()
    gold = input_tensor.t().contiguous()
    
    # Format for output printing
    print(f"{'Kernel':<25}{'Time (ms)':<25}{'Bandwidth (GB/s)':<25}{'Error':<25}")
    print(f"{'------':<25}{'--------':<25}{'----------------':<25}{'-----':<25}")
    
    # Helper function to benchmark and verify
    def benchmark_kernel(kernel_fn, name):
        # Warmup
        kernel_fn(input_tensor, output_tensor)
        
        # Time multiple runs
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(NUM_REPS):
            kernel_fn(input_tensor, output_tensor)
        torch.cuda.synchronize()
        end = time.time()
        
        # Calculate performance
        elapsed_ms = (end - start) * 1000 / NUM_REPS
        bytes_processed = 2 * nx * ny * 4 * NUM_REPS  # 2 accesses per element, 4 bytes per float
        bandwidth_gb_s = bytes_processed / (elapsed_ms * 1e6)
        
        # Verify correctness
        max_error = torch.max(torch.abs(output_tensor - gold)).item() if name != "copy" else 0
        
        print(f"{name:<25}{elapsed_ms:.2f}{bandwidth_gb_s:.2f}{max_error:.6f}")
    
    # Run benchmarks
    benchmark_kernel(copy, "copy")
    benchmark_kernel(transpose_naive, "transpose naive")
    benchmark_kernel(transpose_coalesced, "transpose coalesced")
    benchmark_kernel(transpose_no_bank_conflicts, "transpose no bank conflicts")

if __name__ == "__main__":
    benchmark()