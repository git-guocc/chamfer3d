import torch
from . import _C
from torch import Tensor

class ChamferFunction(torch.autograd.Function):
    """Chamfer距离的自动求导实现"""
    
    @staticmethod
    def forward(ctx, xyz1, xyz2, use_half_precision=False):
        """前向传播计算"""
        # 检查设备
        if not xyz1.is_cuda or not xyz2.is_cuda:
            raise ValueError("输入点云必须在CUDA设备上")
            
        batch_size, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        
        # 检查是否支持半精度
        if use_half_precision and hasattr(_C, "is_mixed_precision_supported"):
            use_half_precision = _C.is_mixed_precision_supported()
        
        # 转换数据类型
        if use_half_precision:
            xyz1 = xyz1.half() if xyz1.dtype != torch.half else xyz1
            xyz2 = xyz2.half() if xyz2.dtype != torch.half else xyz2
        else:
            xyz1 = xyz1.float() if xyz1.dtype != torch.float else xyz1
            xyz2 = xyz2.float() if xyz2.dtype != torch.float else xyz2
        
        # 确保输入是连续的
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        
        # 创建输出tensor
        dtype = torch.half if use_half_precision else torch.float
        dist1 = torch.zeros(batch_size, n, device=xyz1.device, dtype=dtype)
        dist2 = torch.zeros(batch_size, m, device=xyz1.device, dtype=dtype)
        idx1 = torch.zeros(batch_size, n, device=xyz1.device, dtype=torch.int)
        idx2 = torch.zeros(batch_size, m, device=xyz1.device, dtype=torch.int)
        
        # 根据是否使用半精度选择对应的前向计算函数
        if use_half_precision:
            # 检查函数是否存在
            if not hasattr(_C, "chamfer_forward_mixed_precision"):
                raise RuntimeError("半精度前向函数未找到，请确认是否正确编译")
            
            ret = _C.chamfer_forward_mixed_precision(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            # 检查函数是否存在
            if not hasattr(_C, "chamfer_forward"):
                raise RuntimeError("前向函数未找到，请确认是否正确编译")
                
            ret = _C.chamfer_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        
        if ret == 0:
            raise RuntimeError("前向计算失败")
        
        # 保存上下文信息用于反向传播
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        ctx.use_half_precision = use_half_precision
        ctx.batch_size = batch_size
        ctx.n = n
        ctx.m = m
        
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        """反向传播计算"""
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        use_half_precision = ctx.use_half_precision
        batch_size = ctx.batch_size
        n = ctx.n
        m = ctx.m
        
        # 确保梯度类型正确
        if use_half_precision:
            grad_dist1 = grad_dist1.half() if grad_dist1.dtype != torch.half else grad_dist1
            grad_dist2 = grad_dist2.half() if grad_dist2.dtype != torch.half else grad_dist2
        else:
            grad_dist1 = grad_dist1.float() if grad_dist1.dtype != torch.float else grad_dist1
            grad_dist2 = grad_dist2.float() if grad_dist2.dtype != torch.float else grad_dist2
        
        # 确保所有梯度张量是连续的
        grad_dist1 = grad_dist1.contiguous()
        grad_dist2 = grad_dist2.contiguous()
        
        # 创建梯度输出tensor
        dtype = torch.half if use_half_precision else torch.float
        gradxyz1 = torch.zeros_like(xyz1, dtype=dtype)
        gradxyz2 = torch.zeros_like(xyz2, dtype=dtype)
        
        # 根据是否使用半精度选择对应的反向计算函数
        if use_half_precision:
            # 检查函数是否存在
            if not hasattr(_C, "chamfer_backward_mixed_precision"):
                raise RuntimeError("半精度反向函数未找到，请确认是否正确编译")
                
            ret = _C.chamfer_backward_mixed_precision(
                xyz1, xyz2, gradxyz1, gradxyz2, grad_dist1, grad_dist2, idx1, idx2
            )
        else:
            # 检查函数是否存在
            if not hasattr(_C, "chamfer_backward"):
                raise RuntimeError("反向函数未找到，请确认是否正确编译")
                
            ret = _C.chamfer_backward(
                xyz1, xyz2, gradxyz1, gradxyz2, grad_dist1, grad_dist2, idx1, idx2
            )
        
        if ret == 0:
            raise RuntimeError("反向计算失败")
        
        # 返回梯度值 - 与forward中的参数顺序对应
        return gradxyz1, gradxyz2, None

class ChamferDistance(torch.nn.Module):
    """Chamfer距离计算模块
    
    针对RTX 40系列显卡优化的Chamfer距离实现
    支持半精度计算和torch.compile
    """
    def __init__(self, use_half_precision=False):
        """
        初始化Chamfer距离计算模块
        
        Args:
            use_half_precision: 是否使用半精度计算 (仅在支持的GPU上可用)
        """
        super(ChamferDistance, self).__init__()
        # 检查是否支持半精度
        if use_half_precision and hasattr(_C, "is_mixed_precision_supported"):
            self.use_half_precision = _C.is_mixed_precision_supported()
        else:
            self.use_half_precision = False
    
    def forward(self, xyz1: Tensor, xyz2: Tensor) -> tuple:
        """计算两个点云之间的Chamfer距离
        
        Args:
            xyz1 (Tensor): 第一个点云，形状为(B, N, 3)
            xyz2 (Tensor): 第二个点云，形状为(B, M, 3)
            
        Returns:
            tuple: (dist1, dist2, idx1, idx2)，其中:
                - dist1: 从xyz1到xyz2的距离，形状为(B, N)
                - dist2: 从xyz2到xyz1的距离，形状为(B, M)
                - idx1: 从xyz1到xyz2的最近点索引，形状为(B, N)
                - idx2: 从xyz2到xyz1的最近点索引，形状为(B, M)
        """
        # 检查输入
        if xyz1.dim() != 3 or xyz2.dim() != 3:
            raise ValueError(f"输入点云维度错误，期望3维，得到xyz1:{xyz1.dim()}维, xyz2:{xyz2.dim()}维")
        if xyz1.size(2) != 3 or xyz2.size(2) != 3:
            raise ValueError(f"输入点云最后一维必须为3，得到xyz1:{xyz1.size(2)}, xyz2:{xyz2.size(2)}")
        
        # 转换数据类型并使用自动求导函数计算
        return ChamferFunction.apply(xyz1, xyz2, self.use_half_precision)

# 支持torch.compile
def forward(xyz1: Tensor, xyz2: Tensor, use_half_precision: bool = False) -> tuple:
    """函数式API，计算两个点云之间的Chamfer距离
    
    Args:
        xyz1: 第一个点云，形状为(B, N, 3)
        xyz2: 第二个点云，形状为(B, M, 3)
        use_half_precision: 是否使用半精度计算
    
    Returns:
        tuple: (dist1, dist2, idx1, idx2)
    """
    module = ChamferDistance(use_half_precision=use_half_precision)
    return module(xyz1, xyz2)

# 设置可导出符号
__all__ = ['ChamferDistance', 'forward']