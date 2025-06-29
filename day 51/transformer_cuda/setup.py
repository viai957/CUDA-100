from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="transformer_cuda",
    version="0.1.0",
    description="CUDA-accelerated Transformer components",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="layernorm_cuda",
            sources=[
                "transformer_cuda/layernorm_binding.cpp",
                "transformer_cuda/layernorm.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",  # V100
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            },
        ),
        CUDAExtension(
            name="attention_cuda",
            sources=[
                "transformer_cuda/attention_binding.cpp",
                "transformer_cuda/attention.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",  # V100
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            },
        ),
        CUDAExtension(
            name="feed_forward_cuda",
            sources=[
                "transformer_cuda/feed_forward_binding.cpp",
                "transformer_cuda/feed_forward.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",  # V100
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            },
        ),
        CUDAExtension(
            name="embedding_cuda",
            sources=[
                "transformer_cuda/embedding_binding.cpp",
                "transformer_cuda/embedding.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",  # V100
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
    ],
) 