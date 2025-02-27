# Chamfer3D

一个高效的 CUDA 加速 Chamfer 距离计算库，用于 PyTorch。

## 安装

```bash
pip install chamfer3d
```

## 使用示例

```python
import torch
from chamfer3d import chamfer_distance

# 创建两组点云
xyz1 = torch.rand(32, 1024, 3).cuda()  # 批次大小 32，每组 1024 个点
xyz2 = torch.rand(32, 1024, 3).cuda()

# 计算 Chamfer 距离
dist1, dist2, idx1, idx2 = chamfer_distance(xyz1, xyz2)
# dist1: 从第一组到第二组的最小距离
# dist2: 从第二组到第一组的最小距离
# idx1, idx2: 对应的最近点索引
```

#### 2. 创建 `LICENSE` 文件（例如使用 MIT 许可证）

#### 3. 创建 `MANIFEST.in` 文件