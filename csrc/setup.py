"""
Build script for LLMOps custom CUDA kernels.

Usage:
    # Build in-place (development)
    python csrc/setup.py build_ext --inplace

    # Install as a package
    pip install -e csrc/

    # Build with specific CUDA architectures
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" python csrc/setup.py build_ext --inplace

Supported architectures:
    7.0  — V100
    7.5  — T4
    8.0  — A100
    8.6  — A10, A30, RTX 3090
    8.9  — L4, RTX 4090
    9.0  — H100

Requirements:
    - CUDA toolkit >= 11.8
    - PyTorch >= 2.0 (matching CUDA version)
    - C++17 compiler (gcc >= 9 on Linux, MSVC 2019+ on Windows)
"""
import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect CUDA architectures from environment or use sensible defaults
cuda_arch_list = os.environ.get(
    "TORCH_CUDA_ARCH_LIST",
    "7.0;7.5;8.0;8.6;8.9;9.0"   # V100, T4, A100, A10, L4, H100
)

# Convert "8.0;8.6" → ["-gencode=arch=compute_80,code=sm_80", ...]
def arch_flags(arch_list: str):
    flags = []
    for arch in arch_list.split(";"):
        arch = arch.strip().replace(".", "")
        if arch:
            flags += [
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
            ]
    return flags


extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler", "-fPIC",
    ] + arch_flags(cuda_arch_list),
}

# On Windows, replace -fPIC with /O2
if sys.platform == "win32":
    extra_compile_args["cxx"] = ["/O2", "/std:c++17"]
    extra_compile_args["nvcc"] = [
        "-O3", "--use_fast_math", "-std=c++17",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
    ] + arch_flags(cuda_arch_list)

csrc_dir = Path(__file__).parent

setup(
    name="llmops_kernels",
    version="0.1.0",
    description="Custom CUDA kernels for LLMOps inference pipeline",
    author="LLMOps Research Assistant",
    ext_modules=[
        CUDAExtension(
            name="llmops_kernels",
            sources=[str(csrc_dir / "llmops_kernels.cu")],
            extra_compile_args=extra_compile_args,
            libraries=["cublas", "curand"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.11",
    install_requires=["torch>=2.0"],
)
