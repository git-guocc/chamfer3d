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

// 工具函数声明
bool check_tensor(const at::Tensor& tensor, const std::string& name);

} // namespace chamfer3d