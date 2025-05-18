import torch
import triton
import triton.language as tl

@triton.jit
def matrix_vector_product_kernel(
    # Pointers to the matrices
    matrix_ptr, vector_ptr, result_ptr,
    # Matrix dimentions
    M, N,
    # Matrix Strides
    stride_matrix_m, stride_matrix_n,
    # Vector strides
    stride_vector,
    # Block size parameter
    BLOCK_SIZE: tl.constexpr,
): 
    # Program ID 
    pid = tl.program_id(axis=0)

    # Number of rows processed by each program instance
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Calculate the number of rows to process
    

class MatrixVectorMultiplier:
    def __init__(self, n):
        self.dimention = n
        self.BLOCK_SIZE = 32 

    def initialize(self):
        # Create matrix and vector with initial values
        matrix = torch.ones((self.dimention, self.dimention), device="cuda", dtype=torch.float32)
        vector = torch.ones((self.dimention,), device="cuda", dtype=torch.float32)
        return matrix, vector
    
    def compute(self, matrix, vector):
        # Ensure inputs are on the GPU
        if not matrix.is_cuda:
            matrix = matrix.cuda()
        if not vector.is_cuda:
            vector = vector.cuda()

        # Create output vector
        result = torch.zeros((self.dimention,), device="cuda", dtype=torch.float32)

        # Calculate grid dimention
        grid = (self.dimention + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        # Launch the kernel
        matrix_vector_product_kernel[grid](
            matrix, vector, result,
            self.dimention, self.dimention,
            matrix.stride(0), matrix.stride(1), 
            vector.stride(0), result.stride(0),
            BLOCK_SIZE=self.BLOCK_SIZE,
        )

        return result