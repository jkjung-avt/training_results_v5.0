# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fast_dropout_lib',
    ext_modules=[
        CUDAExtension(
            name='fast_dropout_lib',
            sources=['fast_dropout_funcs.cu'],
            extra_compile_args={
                               'cxx': ['-O3',],
                                'nvcc':['-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', "--expt-relaxed-constexpr", "-ftemplate-depth=1024", '-gencode=arch=compute_90,code=sm_90', '-gencode=arch=compute_100,code=sm_100']
                               }
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
