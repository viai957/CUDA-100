from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="whisper_cuda",
    version="0.1",
    description="CUDA kernels for Whisper model components",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="layernorm_cuda",
            sources=[
                "layernorm_binding.cpp",
                "layernorm.cu",
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
            name="linear_cuda",
            sources=[
                "linear_binding.cpp",
                "linear.cu",
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
            name="gelu_cuda",
            sources=[
                "gelu_binding.cpp",
                "gelu.cu",
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
                "attention_binding.cpp",
                "attention.cu",
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
        # Add other components as they are implemented
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
) 