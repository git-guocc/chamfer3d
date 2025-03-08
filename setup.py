from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 尝试自动检测GPU架构
def get_cuda_arch_flags():
    try:
        import torch
        import subprocess
        import re
        
        # 获取CUDA设备属性
        if not torch.cuda.is_available():
            return ['-gencode=arch=compute_89,code=sm_89']  # 默认使用4090架构
        
        # 获取设备名称
        device_name = torch.cuda.get_device_name(0).lower()
        
        # 基于设备名称选择架构
        arch_flags = []
        
        # 40系列 (Ada Lovelace)
        if any(x in device_name for x in ['4090', '4080', '4070', '4060', 'rtx 40']):
            arch_flags.append('-gencode=arch=compute_89,code=sm_89')
        
        # 30系列 (Ampere)
        if any(x in device_name for x in ['3090', '3080', '3070', '3060', 'rtx 30', 'a100', 'a10']):
            arch_flags.append('-gencode=arch=compute_86,code=sm_86')
            # 某些A100和A10使用80架构
            if any(x in device_name for x in ['a100', 'a10']):
                arch_flags.append('-gencode=arch=compute_80,code=sm_80')
        
        # 20系列 (Turing)
        if any(x in device_name for x in ['2080', '2070', '2060', 'rtx 20', 'tesla t4']):
            arch_flags.append('-gencode=arch=compute_75,code=sm_75')
            
        # 如果无法识别特定架构，添加默认支持
        if not arch_flags:
            arch_flags = [
                '-gencode=arch=compute_75,code=sm_75',  # Turing
                '-gencode=arch=compute_80,code=sm_80',  # Ampere (A100)
                '-gencode=arch=compute_86,code=sm_86',  # Ampere (GA102)
                '-gencode=arch=compute_89,code=sm_89',  # Ada Lovelace
            ]
            
        return arch_flags
    except Exception as e:
        print(f"无法自动检测GPU架构: {e}")
        # 返回默认支持的架构
        return ['-gencode=arch=compute_89,code=sm_89']  # 默认使用RTX 4090架构

# 获取架构标志
arch_flags = get_cuda_arch_flags()

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        # 支持按需添加的架构
        *arch_flags,  # 展开架构标志
        '--use_fast_math',  # 使用快速数学函数
        '--extra-device-vectorization',  # 额外设备向量化
        '--ptxas-options=-v',  # 详细的编译信息
        '--extended-lambda',  # 支持扩展lambda
        '-lineinfo',  # 行信息，用于性能分析
        '-std=c++17'  # 支持C++17标准
    ]
}

# 检测CUDA版本并添加特定选项
try:
    import torch
    cuda_version = torch.version.cuda
    if cuda_version and float(cuda_version) >= 11.8:
        extra_compile_args['nvcc'].extend([
            '--threads=0',  # 使用所有CPU线程编译
            '--use_fast_math',  # 使用快速数学函数
            '--Werror=cross-execution-space-call',  # 警告主机/设备调用错误
        ])
except:
    pass

setup(
    name='chamfer3d',
    author='guocc',
    description='Chamfer Distance CUDA implementation for PyTorch',
    long_description_content_type='text/markdown',
    url='https://github.com/git-guocc/chamfer3d',
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
    python_requires='>=3.7',
    install_requires=[
        'torch>=2.0.0',
    ],
)