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

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

cxx_compiler_flags.append("-D__ALLOW_UNSUPPORTED_COMPILER__")
cxx_compiler_flags.append("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
cxx_compiler_flags.append("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")
cxx_compiler_flags.append("-D_HAS_DEPRECATED_RESULT_OF=1")
setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            include_dirs=[
                "C:\\Users\\dgosh\\miniconda3\\envs\\MonoGS\\Library\\include\\targets\\x64",
                "C:\\Users\\dgosh\\miniconda3\\envs\\MonoGS\\Library\\include"
            ],
            extra_compile_args={"nvcc": ["-allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH", "-D_DISABLE_EXTENDED_ALIGNED_STORAGE", "-D_HAS_DEPRECATED_RESULT_OF=1", "-std=c++20"], "cxx": cxx_compiler_flags + ["/std:c++20"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
