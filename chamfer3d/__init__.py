import torch
from torch import nn
from torch.autograd import Function
from chamfer3d._C import chamfer_forward, chamfer_backward

class ChamferDistanceFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        # 确保输入是连续的
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        
        batch_size, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        # 为输出分配内存
        dist1 = torch.zeros(batch_size, n, device=xyz1.device)
        dist2 = torch.zeros(batch_size, m, device=xyz1.device)
        idx1 = torch.zeros(batch_size, n, dtype=torch.int, device=xyz1.device)
        idx2 = torch.zeros(batch_size, m, dtype=torch.int, device=xyz1.device)
        # 调用CUDA前向函数
        chamfer_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        
        # 保存反向传播需要的上下文
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        
        return dist1, dist2, idx1, idx2
    
    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        device = grad_dist1.device
        # 确保梯度输入是连续的
        grad_dist1 = grad_dist1.contiguous()
        grad_dist2 = grad_dist2.contiguous()
        
        # 为梯度输出分配内存
        grad_xyz1 = torch.zeros_like(xyz1)
        grad_xyz2 = torch.zeros_like(xyz2)
        
        grad_xyz1 = grad_xyz1.to(device)
        grad_xyz2 = grad_xyz2.to(device)
        # 调用CUDA反向函数
        chamfer_backward(xyz1, xyz2, grad_xyz1, grad_xyz2, grad_dist1, grad_dist2, idx1, idx2)
        
        # 标量索引没有梯度
        return grad_xyz1, grad_xyz2

class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()
        
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)

def chamfer_distance(xyz1, xyz2):
    """
    计算两组点云之间的 Chamfer 距离
    
    参数:
        xyz1: (batch_size, n_points, 3) 第一组点云
        xyz2: (batch_size, m_points, 3) 第二组点云
        
    返回:
        dist1: (batch_size, n_points) 第一组点到第二组点的最小距离  
        dist2: (batch_size, m_points) 第二组点到第一组点的最小距离
        idx1: (batch_size, n_points) 最近点的索引
        idx2: (batch_size, m_points) 最近点的索引
    """
    return ChamferDistanceFunction.apply(xyz1, xyz2)

__all__ = ['chamfer_forward', 'chamfer_backward', 'chamfer_distance', 'ChamferDistance']