import torch
import triton
import triton.language as tl


# Triton kernel for matrix-vector multiplication
@triton.jit
def matrix_vector_product_kernel(
    # Pointers to matrices
    matrix_ptr, vector_ptr, result_ptr,
    # Matrix dimensions
    M, N,
    # Matrix strides
    stride_matrix_m, stride_matrix_n,
    # Vector stride
    stride_vector,
    # Block size parameter
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Number of rows processed by each program instance
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    row_mask = row_idx < M
    
    # Create vector indices for accessing elements
    vector_idx = tl.arange(0, N)
    vector_mask = vector_idx < N
    
    # Load vector into shared memory
    vector_offsets = vector_idx * stride_vector
    vector = tl.load(vector_ptr + vector_offsets, mask=vector_mask)
    
    # Compute dot product for each row
    row_offsets = row_idx[:, None] * stride_matrix_m + vector_idx[None, :] * stride_matrix_n
    
    # Initialize result
    dot_results = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over the columns of the matrix
    for col_idx in range(0, N):
        # Load matrix elements
        matrix_elements = tl.load(
            matrix_ptr + row_idx[:, None] * stride_matrix_m + col_idx * stride_matrix_n, 
            mask=row_mask[:, None]
        )
        # Load corresponding vector element
        vector_element = tl.load(vector_ptr + col_idx * stride_vector)
        # Update dot product
        dot_results += matrix_elements[:, 0] * vector_element
    
    # Store the results
    tl.store(result_ptr + row_idx * 4, dot_results, mask=row_mask)


class MatrixVectorMultiplier:
    def __init__(self, n):
        self.dimension = n
        self.BLOCK_SIZE = 32  # Similar to THREADS_PER_BLOCK in CUDA
    
    def initialize(self):
        # Create matrix and vector with initial values
        matrix = torch.ones((self.dimension, self.dimension), dtype=torch.float32, device='cuda')
        vector = torch.full((self.dimension,), 2.0, dtype=torch.float32, device='cuda')
        return matrix, vector
    
    def compute(self, matrix, vector):
        # Ensure inputs are on the GPU
        if not matrix.is_cuda:
            matrix = matrix.cuda()
        if not vector.is_cuda:
            vector = vector.cuda()
            
        # Create output tensor
        result = torch.zeros(self.dimension, dtype=torch.float32, device='cuda')
        
        # Calculate grid dimensions
        grid = (self.dimension + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        
        # Launch kernel
        matrix_vector_product_kernel[grid](
            matrix, vector, result,
            self.dimension, self.dimension,
            matrix.stride(0), matrix.stride(1),
            vector.stride(0),
            BLOCK_SIZE=self.BLOCK_SIZE,
        )
        
        return result
    
    def display_results(self, matrix, vector, result):
        print("\nMatrix A:")
        print(matrix.cpu().numpy())
        
        print("\nVector B:")
        print(vector.cpu().numpy())
        
        print("\nResult C:")
        print(result.cpu().numpy())


# Optimized version with better memory access patterns
@triton.jit
def enhanced_matrix_vector_product_kernel(
    matrix_ptr, vector_ptr, result_ptr,
    M, N,
    stride_matrix_m, stride_matrix_n,
    stride_vector,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Row indices
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < M
    
    # Load entire vector into shared memory for faster access
    offsets = tl.arange(0, N) * stride_vector
    vector = tl.load(vector_ptr + offsets, mask=tl.arange(0, N) < N)
    
    # Initialize results
    dots = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Each row computes a dot product
    for i in range(0, N):
        # Load matrix elements for the current column
        matrix_elements = tl.load(
            matrix_ptr + row_idx * stride_matrix_m + i * stride_matrix_n, 
            mask=row_mask
        )
        # Multiply and accumulate
        dots += matrix_elements * vector[i]
    
    # Store results
    tl.store(result_ptr + row_idx, dots, mask=row_mask)


def main():
    # Set matrix size
    MATRIX_SIZE = 10
    
    # Create multiplier
    multiplier = MatrixVectorMultiplier(MATRIX_SIZE)
    
    # Initialize data
    matrix, vector = multiplier.initialize()
    
    # Perform computation
    result = multiplier.compute(matrix, vector)
    
    # Display results
    multiplier.display_results(matrix, vector, result)


if __name__ == "__main__":
    main()