#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "chamfer3d/chamfer_kernel.cuh"

namespace chamfer3d {

__global__ void chamferDistanceKernel(
    int batch_size,
    int n,
    const float* xyz1,
    int m,
    const float* xyz2,
    float* dist,
    int* idx
) {
    const int batch = 512; 
    __shared__ float buf[batch * 3];
    
    // 使用跨步循环处理多批次
    for (int b = blockIdx.x; b < batch_size; b += gridDim.x) {
        // 分批次加载xyz2到共享内存
        for (int k2 = 0; k2 < m; k2 += batch) {
            int end_k = min(m, k2 + batch) - k2;
            
            // 协作加载数据到共享内存
            for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
                buf[j] = xyz2[(b * m + k2) * 3 + j];
            }
            __syncthreads();
            
            // 处理每个点
            for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
                float x1 = xyz1[(b * n + j) * 3 + 0];
                float y1 = xyz1[(b * n + j) * 3 + 1];
                float z1 = xyz1[(b * n + j) * 3 + 2];
                
                float best = 0;
                int best_i = 0;
                
                // 向量化处理：每次计算4个点的距离
                int end_ka = end_k - (end_k & 3); // 向下取整到4的倍数
                
                // 每次处理4个点
                if (end_ka==batch){
                    for (int k = 0; k < end_ka; k += 4) {
                        // 点1
                        {
                            float x2 = buf[k*3+0];
                            float y2 = buf[k*3+1];
                            float z2 = buf[k*3+2];
                            float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                            if (k == 0 || d < best) {
                                best = d;
                                best_i = k + k2;
                            }
                        }     
                        // 点2
                        {
                            float x2 = buf[(k+1)*3+0];
                            float y2 = buf[(k+1)*3+1];
                            float z2 = buf[(k+1)*3+2];
                            float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                            if (d < best) {
                                best = d;
                                best_i = k + k2 + 1;
                            }
                        }
                        
                        // 点3
                        {
                            float x2 = buf[(k+2)*3+0];
                            float y2 = buf[(k+2)*3+1];
                            float z2 = buf[(k+2)*3+2];
                            float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                            if (d < best) {
                                best = d;
                                best_i = k + k2 + 2;
                            }
                        }
                        
                        // 点4
                        {
                            float x2 = buf[(k+3)*3+0];
                            float y2 = buf[(k+3)*3+1];
                            float z2 = buf[(k+3)*3+2];
                            float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                            if (d < best) {
                                best = d;
                                best_i = k + k2 + 3;
                            }
                        }
                    }
                }else{
                    for (int k = 0; k < end_ka; k += 4) {
                        // 点1
                        {
                            float x2 = buf[k*3+0];
                            float y2 = buf[k*3+1];
                            float z2 = buf[k*3+2];
                            float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                            if (k == 0 || d < best) {
                                best = d;
                                best_i = k + k2;
                            }
                        }
                        
                        // 点2
                        {
                            float x2 = buf[(k+1)*3+0];
                            float y2 = buf[(k+1)*3+1];
                            float z2 = buf[(k+1)*3+2];
                            float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                            if (d < best) {
                                best = d;
                                best_i = k + k2 + 1;
                            }
                        }
                        
                        // 点3
                        {
                            float x2 = buf[(k+2)*3+0];
                            float y2 = buf[(k+2)*3+1];
                            float z2 = buf[(k+2)*3+2];
                            float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                            if (d < best) {
                                best = d;
                                best_i = k + k2 + 2;
                            }
                        }
                        
                        // 点4
                        {
                            float x2 = buf[(k+3)*3+0];
                            float y2 = buf[(k+3)*3+1];
                            float z2 = buf[(k+3)*3+2];
                            float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                            if (d < best) {
                                best = d;
                                best_i = k + k2 + 3;
                            }
                        }
                    }
                }
                // 处理剩余的点（不足4个的部分）
                for (int k = end_ka; k < end_k; k++) {
                    float x2 = buf[k*3+0];
                    float y2 = buf[k*3+1];
                    float z2 = buf[k*3+2];
                    float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
                    if (k==0 || d < best) {
                        best = d;
                        best_i = k + k2;
                    }
                }
                
                // 存储这一批次计算的结果
                if (k2==0 || dist[(b*n+j)]>best){
                    dist[b * n + j] = best;
                    idx[b * n + j] = best_i;
                }
            }
            __syncthreads();
        }
    }
}

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
) {
    // 使用跨步循环处理多批次
    for (int b = blockIdx.x; b < batch_size; b += gridDim.x) {
        // 使用跨步循环处理每批内的多点
        for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
            int best_i = idx[b * n + j];
            
            float x1 = xyz1[(b * n + j) * 3 + 0];
            float y1 = xyz1[(b * n + j) * 3 + 1];
            float z1 = xyz1[(b * n + j) * 3 + 2];
            
            float x2 = xyz2[(b * m + best_i) * 3 + 0];
            float y2 = xyz2[(b * m + best_i) * 3 + 1];
            float z2 = xyz2[(b * m + best_i) * 3 + 2];
            
            float g = grad_dist[b * n + j] * 2;
            
            atomicAdd(&(grad_xyz1[(b * n + j) * 3 + 0]), g * (x1 - x2));
            atomicAdd(&(grad_xyz1[(b * n + j) * 3 + 1]), g * (y1 - y2));
            atomicAdd(&(grad_xyz1[(b * n + j) * 3 + 2]), g * (z1 - z2));
            
            atomicAdd(&(grad_xyz2[(b * m + best_i) * 3 + 0]), -g * (x1 - x2));
            atomicAdd(&(grad_xyz2[(b * m + best_i) * 3 + 1]), -g * (y1 - y2));
            atomicAdd(&(grad_xyz2[(b * m + best_i) * 3 + 2]), -g * (z1 - z2));
        }
    }
}

// 保留工具函数
bool check_tensor(const at::Tensor& tensor, const std::string& name) {
    if (!tensor.is_contiguous()) {
        printf("%s tensor is not contiguous!\n", name.c_str());
        return false;
    }
    if (!tensor.is_cuda()) {
        printf("%s tensor is not on CUDA device!\n", name.c_str());
        return false;
    }
    if (tensor.scalar_type() != at::kFloat) {
        printf("%s tensor is not of type float!\n", name.c_str());
        return false;
    }
    return true;
}

} // namespace chamfer3d