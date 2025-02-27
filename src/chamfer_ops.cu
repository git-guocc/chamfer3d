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
    

    chamferDistanceKernel<<<dim3(32,16,1),512>>>(
        batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(), 
        dist1.data_ptr<float>(), idx1.data_ptr<int>()
    );
    
    // 第二次调用交换点云顺序
    chamferDistanceKernel<<<dim3(32,16,1),512>>>(
        batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(), 
        dist2.data_ptr<float>(), idx2.data_ptr<int>()
    );
    
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
    
    // 计算每批次需要的线程块数
    const int threads = 256;
    const int blocks_per_batch = (n + threads - 1) / threads;
    const int blocks_per_batch2 = (m + threads - 1) / threads;
    
    // 调用CUDA内核
    chamferDistanceGradKernel<<<dim3(1,16,1),256>>>(
        batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(), 
        graddist1.data_ptr<float>(), idx1.data_ptr<int>(), 
        gradxyz1.data_ptr<float>(), gradxyz2.data_ptr<float>()
    );
    
    chamferDistanceGradKernel<<<dim3(1,16,1),256>>>(
        batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(), 
        graddist2.data_ptr<float>(), idx2.data_ptr<int>(), 
        gradxyz2.data_ptr<float>(), gradxyz1.data_ptr<float>()
    );
    
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