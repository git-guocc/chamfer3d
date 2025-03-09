from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

def get_cuda_arch_flags():
    try:
        if not torch.cuda.is_available():
            return ['-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_89,code=sm_89']

        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("No CUDA devices available")
            
        arch_flags = []
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            if major < 7:
                continue
            compute_capability = f"compute_{major}{minor}"
            sm_code = f"sm_{major}{minor}"
            arch_flag = f"-gencode=arch={compute_capability},code={sm_code}"
            if arch_flag not in arch_flags:
                arch_flags.append(arch_flag)
        
        return arch_flags if arch_flags else [
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_89,code=sm_89'
        ]
    except Exception as e:
        return ['-gencode=arch=compute_75,code=sm_75',
                '-gencode=arch=compute_80,code=sm_80',
                '-gencode=arch=compute_86,code=sm_86',
                '-gencode=arch=compute_89,code=sm_89']

arch_flags = get_cuda_arch_flags()

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        *arch_flags,
        '--use_fast_math',
        '--ptxas-options=-v',
        '-lineinfo',
        '-std=c++17',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '--expt-relaxed-constexpr',
        '--threads=4'
    ]
}

extension = CUDAExtension(
    name='chamfer3d._C',
    sources=[
        'src/chamfer_ops.cu',
        'src/chamfer_kernel.cu',
    ],
    include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include')],
    extra_compile_args=extra_compile_args
)

setup(
    name='chamfer3d',
    version='0.1.0',
    author='guocc',
    description='Efficient CUDA-accelerated Chamfer Distance for PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/git-guocc/chamfer3d',
    packages=['chamfer3d'],
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.7',
    install_requires=['torch>=1.9.0'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)