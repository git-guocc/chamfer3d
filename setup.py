from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        # '-gencode=arch=compute_75,code=sm_75',  # Turing架构(RTX 2000系列)
        # '-gencode=arch=compute_80,code=sm_80',  # Ampere架构(RTX 3000系列一部分)
        # '-gencode=arch=compute_86,code=sm_86',  # Ampere架构(RTX 3000系列其他)
        '-gencode=arch=compute_89,code=sm_89'   # 您当前的GPU架构
    ]
}

setup(
    name='chamfer3d',
    author='guocc',
    description='Chamfer Distance CUDA implementation for PyTorch',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/chamfer3d',
    packages=['chamfer3d'],
    ext_modules=[
        CUDAExtension(
            name='chamfer3d._C',
            sources=[
                'src/chamfer_ops.cu',
                'src/chamfer_kernel.cu',
            ],
            include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include')],
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)