import torch
import triton 
import triton.language as tl 

@triton.jit
def __kernel_function__(input_pointer, output_pointer, N, BLOCK_SIZE: tl.constexpr):
    # Get the program ID
    program_id = tl.program_id(0) 
    # Offset for the program ID
    offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    input_data = tl.load(input_pointer + offset, mask=mask)
    output_data = tl.sqrt(input_data)
    tl.store(output_pointer + offset, output_data, mask=mask)

def main():
    N = 1024
    
    input_data = torch.arange(0, N, dtype=torch.float32).cuda()
    print("Input Data:", input_data)

    output_data = torch.empty_like(input_data)

    input_ptr = input_data.to("cuda")
    output_ptr = output_data.to("cuda")

    BLOCK_SIZE = 256

    GRID = (triton.cdiv(N, BLOCK_SIZE),)
    __kernel_function__[GRID](input_ptr, output_ptr, N, BLOCK_SIZE)
    
    output_data = output_ptr.cpu()
    print("Output Data:", output_data)

if __name__ == "__main__":
    main()
