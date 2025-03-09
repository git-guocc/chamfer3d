#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include "chamfer3d/chamfer_kernel.cuh"

namespace chamfer3d {

// 使用协作组和向量加载/存储优化的前向计算
__global__ void chamferDistanceKernel(
    int batch_size,
    int n,
    const float* xyz1,
    int m,
    const float* xyz2,
    float* dist,
    int* idx
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    
    const int batch = 512; 
    __shared__ float buf[batch * 3];
    
    // 使用协作组和更高效的内存访问模式
    for (int b = blockIdx.x; b < batch_size; b += gridDim.x) {
        // 分批次加载xyz2到共享内存
        for (int k2 = 0; k2 < m; k2 += batch) {
            int end_k = min(m, k2 + batch) - k2;
            
            // 使用向量加载优化共享内存加载
            // 每个线程加载一个或多个float4值(优化内存事务)
            for (int j = threadIdx.x; j < (end_k * 3 + 3) / 4; j += blockDim.x) {
                int base_idx = j * 4;
                if (base_idx < end_k * 3) {
                    // 尝试向量加载，如果剩余数据足够
                    if (base_idx + 3 < end_k * 3) {
                        float4 data;
                        float4* src_ptr = (float4*)&xyz2[(b * m + k2) * 3 + base_idx];
                        float4* dst_ptr = (float4*)&buf[base_idx];
                        *dst_ptr = *src_ptr;
                    } else {
                        // 处理边界情况
                        for (int k = 0; k < 4 && base_idx + k < end_k * 3; k++) {
                            buf[base_idx + k] = xyz2[(b * m + k2) * 3 + base_idx + k];
                        }
                    }
                }
            }
            block.sync();
            
            // 处理每个点
            for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
                // 向量加载点云1数据
                float3 p1;
                p1.x = xyz1[(b * n + j) * 3 + 0];
                p1.y = xyz1[(b * n + j) * 3 + 1];
                p1.z = xyz1[(b * n + j) * 3 + 2];
                
                float best = 1e20f; // 初始化为一个大值
                int best_i = 0;
                
                // 向量化处理：批量比较距离
                int k = 0;
                
                // 使用float4处理多个点，利用40系列显卡的向量单元
                for (; k + 4 <= end_k; k += 4) {
                    // 点1
                    float3 p2_1 = {buf[k*3], buf[k*3+1], buf[k*3+2]};
                    float d1 = (p1.x-p2_1.x)*(p1.x-p2_1.x) + 
                               (p1.y-p2_1.y)*(p1.y-p2_1.y) + 
                               (p1.z-p2_1.z)*(p1.z-p2_1.z);
                    
                    // 点2
                    float3 p2_2 = {buf[(k+1)*3], buf[(k+1)*3+1], buf[(k+1)*3+2]};
                    float d2 = (p1.x-p2_2.x)*(p1.x-p2_2.x) + 
                               (p1.y-p2_2.y)*(p1.y-p2_2.y) + 
                               (p1.z-p2_2.z)*(p1.z-p2_2.z);
                    
                    // 点3
                    float3 p2_3 = {buf[(k+2)*3], buf[(k+2)*3+1], buf[(k+2)*3+2]};
                    float d3 = (p1.x-p2_3.x)*(p1.x-p2_3.x) + 
                               (p1.y-p2_3.y)*(p1.y-p2_3.y) + 
                               (p1.z-p2_3.z)*(p1.z-p2_3.z);
                    
                    // 点4
                    float3 p2_4 = {buf[(k+3)*3], buf[(k+3)*3+1], buf[(k+3)*3+2]};
                    float d4 = (p1.x-p2_4.x)*(p1.x-p2_4.x) + 
                               (p1.y-p2_4.y)*(p1.y-p2_4.y) + 
                               (p1.z-p2_4.z)*(p1.z-p2_4.z);
                    
                    // 使用三元运算符优化分支预测
                    if (k == 0) {
                        best = d1;
                        best_i = k + k2;
                    } else {
                        if (d1 < best) { best = d1; best_i = k + k2; }
                    }
                    
                    if (d2 < best) { best = d2; best_i = k + k2 + 1; }
                    if (d3 < best) { best = d3; best_i = k + k2 + 2; }
                    if (d4 < best) { best = d4; best_i = k + k2 + 3; }
                }
                
                // 处理剩余的点
                for (; k < end_k; k++) {
                    float3 p2 = {buf[k*3], buf[k*3+1], buf[k*3+2]};
                    float d = (p1.x-p2.x)*(p1.x-p2.x) + 
                              (p1.y-p2.y)*(p1.y-p2.y) + 
                              (p1.z-p2.z)*(p1.z-p2.z);
                    
                    if (k == 0 && k2 == 0) {
                        best = d;
                        best_i = k + k2;
                    } else if (d < best) {
                        best = d;
                        best_i = k + k2;
                    }
                }
                
                // 存储这一批次计算的结果
                if (k2 == 0 || dist[(b*n+j)] > best) {
                    dist[b * n + j] = best;
                    idx[b * n + j] = best_i;
                }
            }
            block.sync();
        }
    }
}

// 优化的反向传播计算
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
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    
    // 使用共享内存优化访问模式
    extern __shared__ float shared_mem[];
    float* shared_grad = shared_mem;
    int* shared_idx = (int*)&shared_grad[blockDim.x];
    
    // 获取全局和块内索引
    const int tid = threadIdx.x;
    const int gid = blockIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.y;
    
    // 每批处理
    for (int b = blockIdx.x; b < batch_size; b += gridDim.x) {
        // 分块处理点云
        for (int j_base = 0; j_base < n; j_base += blockDim.x) {
            int j = j_base + tid;
            
            // 预加载数据到共享内存
            if (j < n) {
                shared_idx[tid] = idx[b * n + j];
                shared_grad[tid] = grad_dist[b * n + j] * 2.0f;
            }
            block.sync();
            
            // 保证仅处理有效范围内的点
            if (j < n) {
                int best_i = shared_idx[tid];
                float g = shared_grad[tid];
                
                // 向量加载点坐标
                float3 p1, p2;
                
                // 使用非合并访问，但对40系显卡更优化的加载模式
                reinterpret_cast<float3*>(&p1.x)[0] = reinterpret_cast<const float3*>(&xyz1[(b * n + j) * 3])[0];
                reinterpret_cast<float3*>(&p2.x)[0] = reinterpret_cast<const float3*>(&xyz2[(b * m + best_i) * 3])[0];
                
                // 计算梯度向量
                float3 grad_vec;
                grad_vec.x = g * (p1.x - p2.x);
                grad_vec.y = g * (p1.y - p2.y);
                grad_vec.z = g * (p1.z - p2.z);
                
                // 使用原子操作更新梯度 - 40系GPU对原子操作有更好的优化
                atomicAdd(&(grad_xyz1[(b * n + j) * 3 + 0]), grad_vec.x);
                atomicAdd(&(grad_xyz1[(b * n + j) * 3 + 1]), grad_vec.y);
                atomicAdd(&(grad_xyz1[(b * n + j) * 3 + 2]), grad_vec.z);
                
                atomicAdd(&(grad_xyz2[(b * m + best_i) * 3 + 0]), -grad_vec.x);
                atomicAdd(&(grad_xyz2[(b * m + best_i) * 3 + 1]), -grad_vec.y);
                atomicAdd(&(grad_xyz2[(b * m + best_i) * 3 + 2]), -grad_vec.z);
            }
            block.sync();
        }
    }
}

// 半精度版本的前向计算内核 - 重写为兼容禁用半精度操作符的环境
__global__ void chamferDistanceKernelHalf(
    int batch_size,
    int n,
    const __half* xyz1,
    int m,
    const __half* xyz2,
    __half* dist,
    int* idx
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    
    const int batch = 512; 
    __shared__ float buf_float[batch * 3];  // 使用float共享内存而不是half
    
    for (int b = blockIdx.x; b < batch_size; b += gridDim.x) {
        // 分批次加载xyz2到共享内存
        for (int k2 = 0; k2 < m; k2 += batch) {
            int end_k = min(m, k2 + batch) - k2;
            
            // 加载数据到共享内存，但先转换为float类型
            for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
                if (j < end_k * 3) {
                    buf_float[j] = __half2float(xyz2[(b * m + k2) * 3 + j]);
                }
            }
            block.sync();
            
            // 处理每个点
            for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
                // 加载点云1数据并转换为float
                float fx1 = __half2float(xyz1[(b * n + j) * 3 + 0]);
                float fy1 = __half2float(xyz1[(b * n + j) * 3 + 1]);
                float fz1 = __half2float(xyz1[(b * n + j) * 3 + 2]);
                
                // 存储最佳距离和索引
                float best = 1e20f;
                int best_i = 0;
                
                // 批量处理点
                int k = 0;
                for (; k + 4 <= end_k; k += 4) {
                    // 使用浮点计算以获得更高精度
                    // 第1个点
                    float fx2_1 = buf_float[k*3+0];
                    float fy2_1 = buf_float[k*3+1];
                    float fz2_1 = buf_float[k*3+2];
                    float d1 = (fx1-fx2_1)*(fx1-fx2_1) + (fy1-fy2_1)*(fy1-fy2_1) + (fz1-fz2_1)*(fz1-fz2_1);
                    
                    // 第2个点
                    float fx2_2 = buf_float[(k+1)*3+0];
                    float fy2_2 = buf_float[(k+1)*3+1];
                    float fz2_2 = buf_float[(k+1)*3+2];
                    float d2 = (fx1-fx2_2)*(fx1-fx2_2) + (fy1-fy2_2)*(fy1-fy2_2) + (fz1-fz2_2)*(fz1-fz2_2);
                    
                    // 第3个点
                    float fx2_3 = buf_float[(k+2)*3+0];
                    float fy2_3 = buf_float[(k+2)*3+1];
                    float fz2_3 = buf_float[(k+2)*3+2];
                    float d3 = (fx1-fx2_3)*(fx1-fx2_3) + (fy1-fy2_3)*(fy1-fy2_3) + (fz1-fz2_3)*(fz1-fz2_3);
                    
                    // 第4个点
                    float fx2_4 = buf_float[(k+3)*3+0];
                    float fy2_4 = buf_float[(k+3)*3+1];
                    float fz2_4 = buf_float[(k+3)*3+2];
                    float d4 = (fx1-fx2_4)*(fx1-fx2_4) + (fy1-fy2_4)*(fy1-fy2_4) + (fz1-fz2_4)*(fz1-fz2_4);
                    
                    // 找出最小距离
                    if (k == 0 && k2 == 0) {
                        best = d1;
                        best_i = k + k2;
                    } else {
                        if (d1 < best) { 
                            best = d1; 
                            best_i = k + k2; 
                        }
                    }
                    
                    if (d2 < best) { best = d2; best_i = k + k2 + 1; }
                    if (d3 < best) { best = d3; best_i = k + k2 + 2; }
                    if (d4 < best) { best = d4; best_i = k + k2 + 3; }
                }
                
                // 处理剩余的点
                for (; k < end_k; k++) {
                    float fx2 = buf_float[k*3+0];
                    float fy2 = buf_float[k*3+1];
                    float fz2 = buf_float[k*3+2];
                    float d = (fx1-fx2)*(fx1-fx2) + (fy1-fy2)*(fy1-fy2) + (fz1-fz2)*(fz1-fz2);
                    
                    if (k == 0 && k2 == 0) {
                        best = d;
                        best_i = k + k2;
                    } else if (d < best) {
                        best = d;
                        best_i = k + k2;
                    }
                }
                
                // 存储结果，需要先获取当前值
                float curr_dist = 0.0f;
                if (k2 > 0) {
                    curr_dist = __half2float(dist[b * n + j]);
                }
                
                if (k2 == 0 || curr_dist > best) {
                    dist[b * n + j] = __float2half(best);
                    idx[b * n + j] = best_i;
                }
            }
            block.sync();
        }
    }
}

// 重写半精度反向传播内核，以兼容禁用半精度运算符的环境
__global__ void chamferDistanceGradKernelHalf(
    int batch_size,
    int n,
    const __half* xyz1,
    int m,
    const __half* xyz2,
    const __half* grad_dist,
    const int* idx,
    __half* grad_xyz1,
    __half* grad_xyz2
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    
    // 使用共享内存
    extern __shared__ float shared_mem[];
    float* shared_grad = shared_mem;
    int* shared_idx = (int*)&shared_grad[blockDim.x];
    
    const int tid = threadIdx.x;
    
    for (int b = blockIdx.x; b < batch_size; b += gridDim.x) {
        for (int j_base = 0; j_base < n; j_base += blockDim.x) {
            int j = j_base + tid;
            
            // 预加载数据到共享内存
            if (j < n) {
                shared_idx[tid] = idx[b * n + j];
                shared_grad[tid] = __half2float(grad_dist[b * n + j]) * 2.0f;
            }
            block.sync();
            
            if (j < n) {
                int best_i = shared_idx[tid];
                float g = shared_grad[tid];
                
                // 加载坐标并转换为float
                float p1[3], p2[3];
                
                p1[0] = __half2float(xyz1[(b * n + j) * 3 + 0]);
                p1[1] = __half2float(xyz1[(b * n + j) * 3 + 1]);
                p1[2] = __half2float(xyz1[(b * n + j) * 3 + 2]);
                
                p2[0] = __half2float(xyz2[(b * m + best_i) * 3 + 0]);
                p2[1] = __half2float(xyz2[(b * m + best_i) * 3 + 1]);
                p2[2] = __half2float(xyz2[(b * m + best_i) * 3 + 2]);
                
                // 计算梯度向量
                float grad_vec[3];
                grad_vec[0] = g * (p1[0] - p2[0]);
                grad_vec[1] = g * (p1[1] - p2[1]);
                grad_vec[2] = g * (p1[2] - p2[2]);
                
                // 使用临时变量更新梯度，避免半精度原子操作
                for (int i = 0; i < 3; i++) {
                    // 为xyz1更新梯度
                    grad_xyz1[(b * n + j) * 3 + i] = __float2half(
                        __half2float(grad_xyz1[(b * n + j) * 3 + i]) + grad_vec[i]
                    );
                    
                    // 为xyz2更新梯度
                    grad_xyz2[(b * m + best_i) * 3 + i] = __float2half(
                        __half2float(grad_xyz2[(b * m + best_i) * 3 + i]) - grad_vec[i]
                    );
                }
            }
            block.sync();
        }
    }
}

// 修改工具函数实现
bool check_tensor(const at::Tensor& tensor, const std::string& name) {
    if (!tensor.is_contiguous()) {
        printf("%s tensor is not contiguous!\n", name.c_str());
        return false;
    }
    if (!tensor.is_cuda()) {
        printf("%s tensor is not on CUDA device!\n", name.c_str());
        return false;
    }
    
    // 根据张量名检查类型
    if (name.find("idx") != std::string::npos) {
        // 索引张量应该是整数类型
        if (tensor.scalar_type() != at::kInt && tensor.scalar_type() != at::kLong) {
            printf("%s tensor should be of integer type (Int or Long)!\n", name.c_str());
            return false;
        }
    } else {
        // 其他张量应该是浮点类型
        if (tensor.scalar_type() != at::kFloat) {
            printf("%s tensor is not of type float!\n", name.c_str());
            return false;
        }
    }
    return true;
}

} // namespace chamfer3d