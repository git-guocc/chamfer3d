#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace chamfer3d {

// CUDA内核声明
__global__ void chamferDistanceKernel(
    int batch_size,
    int n,
    const float* xyz1,
    int m,
    const float* xyz2,
    float* dist,
    int* idx
);

__global__ void chamferDistanceGradKernel(
    int batch_size,
    int n,
    const float* xyz1,
    int m,
    const float* xyz2,
    const float* grad_dist,
    const int* idx,
    float* grad_xyz1,
    float* grad_xyz2
);

// 接口函数声明
int chamfer_forward(
    at::Tensor xyz1,
    at::Tensor xyz2, 
    at::Tensor dist1,
    at::Tensor dist2,
    at::Tensor idx1,
    at::Tensor idx2
);

int chamfer_backward(
    at::Tensor xyz1,
    at::Tensor xyz2,
    at::Tensor gradxyz1,
    at::Tensor gradxyz2,
    at::Tensor graddist1,
    at::Tensor graddist2,
    at::Tensor idx1,
    at::Tensor idx2
);

/**
 * 检查张量的有效性
 * 
 * 对于名称中包含"idx"的张量，检查它是否为整数类型(Int或Long)
 * 对于其他张量，检查它是否为浮点类型(Float)
 * 所有张量都必须是连续的并且在CUDA设备上
 */
bool check_tensor(const at::Tensor& tensor, const std::string& name);

} // namespace chamfer3d