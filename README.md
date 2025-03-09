# Chamfer3D

一个高效的CUDA加速的Chamfer距离计算实现，专为PyTorch设计。支持混合精度计算，并针对NVIDIA 40系列GPU进行了优化。

## 特性

- CUDA加速实现
- 支持混合精度计算（FP16/FP32）
- 自动GPU架构检测
- 针对最新NVIDIA GPU架构优化

## 安装

### 从PyPI安装

```bash
pip install chamfer3d
```

### 从源码安装

```bash
git clone https://github.com/git-guocc/chamfer3d
cd chamfer3d
pip install -e .
```

## 使用方法

```python
import torch
import chamfer3d

# 创建两个点云
points1 = torch.randn(32, 1000, 3).cuda()  # batch_size=32, 1000个点, 3D
points2 = torch.randn(32, 2000, 3).cuda()  # batch_size=32, 2000个点, 3D

# 计算Chamfer距离
dist1, dist2, idx1, idx2 = chamfer3d.forward(points1, points2)

# 使用半精度计算（可选）
points1_half = points1.half()
points2_half = points2.half()
dist1_half, dist2_half, idx1_half, idx2_half = chamfer3d.forward(points1_half, points2_half)
```

## 要求

- Python >= 3.7
- PyTorch >= 1.9.0
- CUDA >= 11.0

## 许可证

MIT License