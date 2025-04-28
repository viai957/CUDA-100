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
### 📖 Reflections on “Programming Massively Parallel Processors” (2007)

Today I dove deep into the first 22 pages of the Programming Massively Parallel Processors book, and I have to say reading this in 2025 feels surreal, almost like the authors time-traveled from the future into 2007 to leave behind a prophetic roadmap for parallel computing.

The opening chapter sets the stage by tracing the evolution from traditional, single-core CPU computing to the explosion of multicore and many-core architectures. It outlines how, for decades, Moore’s Law kept single-thread performance improving — allowing software developers to blissfully ride the hardware improvement curve without worrying about parallelism. But that ride hit a wall around 2003, when power consumption and heat dissipation issues capped CPU clock speeds.

This concurrency revolution — a term echoed from Herb Sutter’s famous writings — marks the inflection point. Suddenly, developers could no longer count on hardware to make their single-threaded code run faster. They needed to think differently. Parallelism was no longer optional; it was survival.

### 🔥 Surprising Technical Elements (for 2007)

Several things struck me hard, even by 2025 standards:
	•	GPUs already had 240 cores (NVIDIA GTX 280) when CPUs were just timidly moving from 2 to 4 cores.
	•	Memory Bandwidth: GPUs were achieving 85–150 GB/s while CPUs struggled around 10–20 GB/s — a full 10x bandwidth gap. Even today, we still wrestle with CPU-GPU bandwidth mismatches, but back then, it was already glaring.
	•	Architectural Simplicity for Parallelism: GPUs sacrificed sophisticated caches and speculative execution (hallmarks of CPU design) to favour raw floating-point throughput by packing simple ALUs. This “sacrifice” is now considered a feature in modern accelerators like TPUs and AI-specialised chips.
	•	CUDA in 2007: It’s wild to think CUDA’s debut came alongside this architectural shift, unifying CPU and GPU programming models. It set the stage for today’s deep learning revolution. Without CUDA, frameworks like TensorFlow, PyTorch, and large-scale foundation model training would have been inconceivable.

📈 Conceptual Foundations
	•	CPUs prioritise low-latency, sequential execution. They are optimised for tasks requiring complex control logic and quick, unpredictable memory accesses.
	•	GPUs prioritise high-throughput, massively parallel execution. They rely on the ability to hide memory latencies by simply scheduling another ready thread — a concept we now canonise as thread-level parallelism (TLP).

This different philosophy between CPU and GPU design reminds me of the Tortoise and the Hare — CPUs are smart, cautious tortoises optimising every move, while GPUs are dumb but fast hares racing through sheer numbers.

🧠 Forward-Thinking Reflection

Reading this in 2025, it’s clear the authors understood the direction of compute evolution at a granular level long before it became obvious to the mainstream. When you consider that transformers, LLMs, and AI hardware accelerators today all depend on these ideas, this book feels not just prescient but visionary.

If you had internalised and acted on these ideas in 2007, you would have been 5–10 years ahead of the curve.


