# Day 1 - 100 Days of CUDA

#### File: vectadd.cu
#### Summary:
Implemented vector addition by writing a simple CUDA program. Explored how to launch a kernel to perform a parallelized addition of two arrays, where each thread computes the sum of a pair of values.

- **File:** `vectadd.cu`
- **Summary:**  
  - Implemented vector addition using a simple CUDA program.
  - Launched a kernel to perform parallelized addition of two arrays.
  - Each thread computes the sum of a pair of values.

- **Learned:**
  - Basics of writing a CUDA kernel.
  - Understanding of grid, block, and thread hierarchy in CUDA.
  - How to allocate and manage device (GPU) memory using:
    - `cudaMalloc`
    - `cudaMemcpy`
    - `cudaFree`

## Reading:
### 📖 Reflections on "Programming Massively Parallel Processors" (2007)

Today I dove deep into the first 22 pages of the Programming Massively Parallel Processors book, and I have to say reading this in 2025 feels surreal, almost like the authors time-traveled from the future into 2007 to leave behind a prophetic roadmap for parallel computing.

The opening chapter sets the stage by tracing the evolution from traditional, single-core CPU computing to the explosion of multicore and many-core architectures. It outlines how, for decades, Moore's Law kept single-thread performance improving — allowing software developers to blissfully ride the hardware improvement curve without worrying about parallelism. But that ride hit a wall around 2003, when power consumption and heat dissipation issues capped CPU clock speeds.

This concurrency revolution — a term echoed from Herb Sutter's famous writings — marks the inflection point. Suddenly, developers could no longer count on hardware to make their single-threaded code run faster. They needed to think differently. Parallelism was no longer optional; it was survival.

### 🔥 Surprising Technical Elements (for 2007)

Several things struck me hard, even by 2025 standards:
	•	GPUs already had 240 cores (NVIDIA GTX 280) when CPUs were just timidly moving from 2 to 4 cores.
	•	Memory Bandwidth: GPUs were achieving 85–150 GB/s while CPUs struggled around 10–20 GB/s — a full 10x bandwidth gap. Even today, we still wrestle with CPU-GPU bandwidth mismatches, but back then, it was already glaring.
	•	Architectural Simplicity for Parallelism: GPUs sacrificed sophisticated caches and speculative execution (hallmarks of CPU design) to favour raw floating-point throughput by packing simple ALUs. This "sacrifice" is now considered a feature in modern accelerators like TPUs and AI-specialised chips.
	•	CUDA in 2007: It's wild to think CUDA's debut came alongside this architectural shift, unifying CPU and GPU programming models. It set the stage for today's deep learning revolution. Without CUDA, frameworks like TensorFlow, PyTorch, and large-scale foundation model training would have been inconceivable.

📈 Conceptual Foundations
	•	CPUs prioritise low-latency, sequential execution. They are optimised for tasks requiring complex control logic and quick, unpredictable memory accesses.
	•	GPUs prioritise high-throughput, massively parallel execution. They rely on the ability to hide memory latencies by simply scheduling another ready thread — a concept we now canonise as thread-level parallelism (TLP).

This different philosophy between CPU and GPU design reminds me of the Tortoise and the Hare — CPUs are smart, cautious tortoises optimising every move, while GPUs are dumb but fast hares racing through sheer numbers.

🧠 Forward-Thinking Reflection

Reading this in 2025, it's clear the authors understood the direction of compute evolution at a granular level long before it became obvious to the mainstream. When you consider that transformers, LLMs, and AI hardware accelerators today all depend on these ideas, this book feels not just prescient but visionary.

If you had internalised and acted on these ideas in 2007, you would have been 5–10 years ahead of the curve.


# Day 2 - Matrix Addition in CUDA

#### File: matrix_add.cu
#### Summary:
Implemented matrix addition using CUDA, exploring different approaches to parallelize 2D data structures. Developed multiple kernel implementations to understand various parallelization strategies.

- **File:** `matrix_add.cu`
- **Summary:**
  - Implemented three different approaches to matrix addition:
    - Row-based parallelization (MatrixAdd_C)
    - Block-based 2D parallelization (MatrixAdd_B)
    - Column-based parallelization (MatrixAdd_D)
  - Utilized 2D grid and block configurations for efficient matrix operations

- **Learned:**
  - 2D thread block and grid organization
  - Different strategies for parallelizing matrix operations
  - Memory access patterns in 2D data structures
  - Using `dim3` for multi-dimensional kernel launches
  - Importance of proper indexing in 2D matrix operations

## Reading:
### 📖 Reflections on "Programming Massively Parallel Processors" (2007)



# Day 3 - Matrix-Vector Multiplication in CUDA

#### File: matrix_mul.cu
#### Summary:
Implemented an optimized matrix-vector multiplication using CUDA, exploring advanced memory management patterns and object-oriented CUDA programming concepts.

- **File:** `matrix_mul.cu`
- **Summary:**
  - Implemented matrix-vector multiplication with shared memory optimization
  - Developed a structured approach using modern C++ and CUDA best practices
  - Created reusable components for CUDA memory management and computation
  - Utilized template metaprogramming for generic CUDA operations

- **Learned:**
  - Advanced CUDA memory patterns:
    - Shared memory utilization for vector caching
    - Coalesced memory access patterns
    - CUDA memory management using RAII principles
  - Modern CUDA programming techniques:
    - Template-based device memory management
    - Error handling and validation
    - Object-oriented CUDA design patterns
  - Performance optimization strategies:
    - Loop unrolling with #pragma unroll
    - Efficient thread block configuration
    - Shared memory usage for frequently accessed data

- **Key Implementations:**
  - Custom `DeviceMemoryManager` template for CUDA memory handling
  - Enhanced kernel with shared memory optimization
  - Structured error handling and dimension validation
  - Object-oriented approach to CUDA operations

This implementation demonstrates a more sophisticated approach to CUDA programming, combining modern C++ features with efficient GPU computing patterns. The focus was on creating maintainable, reusable, and efficient CUDA code while ensuring optimal performance through various optimization techniques.

# Day 4 - Understanding CUDA Thread Indexing

#### File: indexing_simple.cu
#### Summary:
I know this is a simple one but i had to write this to visualize what was going inside the Thearead allocation.
Created a visual and intuitive demonstration of CUDA's thread indexing system using an apartment building analogy to understand 3D grid and block organization.

- **File:** `indexing_simple.cu`
- **Summary:**
  - Implemented a visualization kernel to understand CUDA's thread hierarchy
  - Used an apartment complex analogy for better understanding:
    - Buildings = Grid in Z dimension (blockIdx.z)
    - Floors = Grid in Y dimension (blockIdx.y)
    - Apartments per floor = Grid in X dimension (blockIdx.x)
    - People per apartment = Threads in a block (threadIdx.x/y/z)

- **Learned:**
  - Thread Indexing Fundamentals:
    - How to calculate global thread IDs in 3D space
    - Understanding block and thread hierarchies
    - Relationship between grid dimensions and block dimensions
  
  - Key CUDA Index Components:
    ```
    Block Position:
    - blockIdx.x: Position along width (horizontal)
    - blockIdx.y: Position along height (vertical)
    - blockIdx.z: Position along depth
    
    Thread Position within Block:
    - threadIdx.x: Local x position in block
    - threadIdx.y: Local y position in block
    - threadIdx.z: Local z position in block
    ```

  - Practical Understanding:
    - How to organize threads in 3D space
    - Calculating unique global thread IDs
    - Visualizing thread distribution across blocks

- **Key Concepts Visualized:**
  - Block ID calculation:
    ```cuda
    block_id = blockIdx.x + 
               blockIdx.y * gridDim.x + 
               blockIdx.z * gridDim.x * gridDim.y
    ```
  - Thread offset within block:
    ```cuda
    thread_offset = threadIdx.x + 
                   threadIdx.y * blockDim.x + 
                   threadIdx.z * blockDim.x * blockDim.y
    ```
  - Global thread ID:
    ```cuda
    global_id = block_offset + thread_offset
    ```

- **Practical Example:**
  - Created a grid of 2×3×4 blocks (24 total blocks)
  - Each block contains 4×4×4 threads (64 threads per block)
  - Total threads: 24 blocks × 64 threads = 1,536 threads
  - Each thread prints its unique position and ID for visualization

This implementation serves as a fundamental building block for understanding more complex CUDA programs, as proper thread indexing is crucial for correct parallel algorithm implementation.

# Day-18 but modified file name day 15, 16 (vector add Trick & Matrix Transpose)
I did learn about writing effienct algo capitalizing on the Hardware, Shared Memory allows us to efifecnly enable how to manage mutiple threads , warps mapped to blocks which are assigned to multiprocessors on the device. During execution there is a finer grouping of threads into warps. Multiprocessors on the GPU execute instructions for each warp in SIMD (Single Instruction Multiple Data) fashion. Got very little time today as i had to work on my day job. Hope to catchup from tomorrow as its a week end. 


