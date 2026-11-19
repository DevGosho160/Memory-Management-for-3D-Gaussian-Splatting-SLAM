#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            include_dirs=[
                "C:\\Users\\dgosh\\miniconda3\\envs\\MonoGS\\Library\\include\\targets\\x64",
                "C:\\Users\\dgosh\\miniconda3\\envs\\MonoGS\\Library\\include"
            ],
            extra_compile_args={
                "nvcc": ["-allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH", "-D_DISABLE_EXTENDED_ALIGNED_STORAGE", "-D_HAS_DEPRECATED_RESULT_OF=1", "-std=c++20", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],
                "cxx": ["-D__ALLOW_UNSUPPORTED_COMPILER__", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH", "-D_DISABLE_EXTENDED_ALIGNED_STORAGE", "-D_HAS_DEPRECATED_RESULT_OF=1", "/std:c++20"]
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
