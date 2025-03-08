#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include "chamfer3d/chamfer_kernel.cuh"

namespace chamfer3d {

// PyTorch接口实现
int chamfer_forward(
    at::Tensor xyz1, 
    at::Tensor xyz2, 
    at::Tensor dist1, 
    at::Tensor dist2, 
    at::Tensor idx1, 
    at::Tensor idx2
) {
    const auto batch_size = xyz1.size(0);
    const auto n = xyz1.size(1);
    const auto m = xyz2.size(1);
    
    // 检查输入
    if (!check_tensor(xyz1, "xyz1") || !check_tensor(xyz2, "xyz2") ||
        !check_tensor(dist1, "dist1") || !check_tensor(dist2, "dist2") ||
        !check_tensor(idx1, "idx1") || !check_tensor(idx2, "idx2")) {
        return 0;
    }
    
    // 为40系显卡优化的配置
    // 获取当前设备属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // 根据SM数量和性能动态调整线程配置
    const int sm_count = prop.multiProcessorCount;
    const int threads_per_block = 512;
    
    // 为每个SM分配至少一个block，最多32个block
    int x_blocks = min(32, max(1, sm_count / 2));
    int y_blocks = min(32, max(1, sm_count / 2));
    
    dim3 grid(x_blocks, y_blocks, 1);
    
    // 使用CUDA流提高并行性
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    chamferDistanceKernel<<<grid, threads_per_block, 0, stream1>>>(
        batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(), 
        dist1.data_ptr<float>(), idx1.data_ptr<int>()
    );
    
    // 第二次调用交换点云顺序
    chamferDistanceKernel<<<grid, threads_per_block, 0, stream2>>>(
        batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(), 
        dist2.data_ptr<float>(), idx2.data_ptr<int>()
    );
    
    // 同步流
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // 销毁流
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in chamfer_forward: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

int chamfer_backward(
    at::Tensor xyz1, 
    at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2
) {
    const auto batch_size = xyz1.size(0);
    const auto n = xyz1.size(1);
    const auto m = xyz2.size(1);
    
    // 检查输入
    if (!check_tensor(xyz1, "xyz1") || !check_tensor(xyz2, "xyz2") ||
        !check_tensor(gradxyz1, "gradxyz1") || !check_tensor(gradxyz2, "gradxyz2") ||
        !check_tensor(graddist1, "graddist1") || !check_tensor(graddist2, "graddist2") ||
        !check_tensor(idx1, "idx1") || !check_tensor(idx2, "idx2")) {
        return 0;
    }
    
    // 获取当前设备属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // 根据SM数量和性能动态调整线程配置
    const int sm_count = prop.multiProcessorCount;
    const int threads_per_block = 256;
    
    // 为每个SM分配至少一个block
    int x_blocks = min(32, max(1, sm_count / 4));
    int y_blocks = min(16, max(1, sm_count / 2));
    
    dim3 grid(x_blocks, y_blocks, 1);
    
    // 使用CUDA流提高并行性
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // 计算共享内存大小：float数组 + int数组
    size_t shared_mem_size = threads_per_block * (sizeof(float) + sizeof(int));
    
    // 调用CUDA内核
    chamferDistanceGradKernel<<<grid, threads_per_block, shared_mem_size, stream1>>>(
        batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(), 
        graddist1.data_ptr<float>(), idx1.data_ptr<int>(), 
        gradxyz1.data_ptr<float>(), gradxyz2.data_ptr<float>()
    );
    
    chamferDistanceGradKernel<<<grid, threads_per_block, shared_mem_size, stream2>>>(
        batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(), 
        graddist2.data_ptr<float>(), idx2.data_ptr<int>(), 
        gradxyz2.data_ptr<float>(), gradxyz1.data_ptr<float>()
    );
    
    // 同步流
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // 销毁流
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in chamfer_backward: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chamfer_forward", &chamfer3d::chamfer_forward, "Chamfer forward");
    m.def("chamfer_backward", &chamfer3d::chamfer_backward, "Chamfer backward");
}

} // namespace chamfer3d