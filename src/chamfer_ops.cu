#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <unordered_map>
#include <mutex>
#include "chamfer3d/chamfer_kernel.cuh"

namespace chamfer3d {

// CUDA Graph缓存
struct GraphCacheKey {
    int batch_size;
    int n;
    int m;
    
    bool operator==(const GraphCacheKey& other) const {
        return batch_size == other.batch_size && n == other.n && m == other.m;
    }
};

// 自定义哈希函数
struct GraphCacheKeyHash {
    std::size_t operator()(const GraphCacheKey& key) const {
        return std::hash<int>()(key.batch_size) ^ 
               std::hash<int>()(key.n) << 1 ^ 
               std::hash<int>()(key.m) << 2;
    }
};

// 存储CUDA Graph的全局缓存
static std::unordered_map<GraphCacheKey, cudaGraph_t, GraphCacheKeyHash> graph_cache_forward;
static std::unordered_map<GraphCacheKey, cudaGraphExec_t, GraphCacheKeyHash> graph_exec_cache_forward;
static std::mutex graph_mutex;

// 检查是否支持半精度计算
bool is_mixed_precision_supported() {
    int device_id;
    cudaGetDevice(&device_id);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    
    // 检查计算能力是否支持半精度计算
    if (props.major >= 7) {
        return true;
    }
    return false;
}

// 检查半精度张量
bool check_tensor_half(const at::Tensor& tensor, const std::string& name) {
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
        // 其他张量应该是半精度类型
        if (tensor.scalar_type() != at::kHalf) {
            printf("%s tensor is not of type half!\n", name.c_str());
            return false;
        }
    }
    return true;
}

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
    
    // 尝试使用CUDA Graph (对于重复调用的固定尺寸输入)
    GraphCacheKey key = {batch_size, n, m};
    bool use_graph = false;
    
    // 检查是否启用了CUDA图表并且参数大小满足一定条件(适合重用)
    if (batch_size > 1 && n > 1000 && m > 1000) {
        std::lock_guard<std::mutex> lock(graph_mutex);
        
        // 检查缓存中是否有匹配的图表
        if (graph_exec_cache_forward.find(key) != graph_exec_cache_forward.end()) {
            // 使用缓存的图表执行
            cudaGraphExec_t graphExec = graph_exec_cache_forward[key];
            cudaGraphLaunch(graphExec, stream1);
            use_graph = true;
        } else if (graph_cache_forward.find(key) == graph_cache_forward.end()) {
            // 创建并捕获新的图表
            cudaGraph_t graph;
            cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
            
            // 执行要捕获的操作
            chamferDistanceKernel<<<grid, threads_per_block, 0, stream1>>>(
                batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(), 
                dist1.data_ptr<float>(), idx1.data_ptr<int>()
            );
            
            chamferDistanceKernel<<<grid, threads_per_block, 0, stream1>>>(
                batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(), 
                dist2.data_ptr<float>(), idx2.data_ptr<int>()
            );
            
            // 结束捕获
            cudaStreamEndCapture(stream1, &graph);
            graph_cache_forward[key] = graph;
            
            // 实例化图表
            cudaGraphExec_t graphExec;
            cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
            graph_exec_cache_forward[key] = graphExec;
            
            // 启动图表
            cudaGraphLaunch(graphExec, stream1);
            use_graph = true;
        }
    }
    
    if (!use_graph) {
        // 如果没有使用图表，执行常规内核调用
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
    } else {
        // 如果使用了图表，只需要同步stream1
        cudaStreamSynchronize(stream1);
    }
    
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

// 混合精度前向传播实现
int chamfer_forward_mixed_precision(
    at::Tensor xyz1, 
    at::Tensor xyz2, 
    at::Tensor dist1, 
    at::Tensor dist2, 
    at::Tensor idx1, 
    at::Tensor idx2
) {
    // 检查是否支持半精度计算
    if (!is_mixed_precision_supported()) {
        printf("Mixed precision computation not supported on this device!\n");
        return 0;
    }
    
    const auto batch_size = xyz1.size(0);
    const auto n = xyz1.size(1);
    const auto m = xyz2.size(1);
    
    // 检查输入
    if (!check_tensor_half(xyz1, "xyz1") || !check_tensor_half(xyz2, "xyz2") ||
        !check_tensor_half(dist1, "dist1") || !check_tensor_half(dist2, "dist2") ||
        !check_tensor(idx1, "idx1") || !check_tensor(idx2, "idx2")) {
        return 0;
    }

    // 为40系显卡优化的配置
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    const int sm_count = prop.multiProcessorCount;
    const int threads_per_block = 512;
    
    // 针对Tensor Core优化的block配置
    int x_blocks = min(32, max(1, sm_count / 2));
    int y_blocks = min(32, max(1, sm_count / 2));
    
    dim3 grid(x_blocks, y_blocks, 1);
    
    // 使用CUDA流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 调用半精度内核
    chamferDistanceKernelHalf<<<grid, threads_per_block, 0, stream1>>>(
        batch_size, n, reinterpret_cast<__half*>(xyz1.data_ptr<at::Half>()), 
        m, reinterpret_cast<__half*>(xyz2.data_ptr<at::Half>()), 
        reinterpret_cast<__half*>(dist1.data_ptr<at::Half>()), 
        idx1.data_ptr<int>()
    );
    
    chamferDistanceKernelHalf<<<grid, threads_per_block, 0, stream2>>>(
        batch_size, m, reinterpret_cast<__half*>(xyz2.data_ptr<at::Half>()), 
        n, reinterpret_cast<__half*>(xyz1.data_ptr<at::Half>()), 
        reinterpret_cast<__half*>(dist2.data_ptr<at::Half>()), 
        idx2.data_ptr<int>()
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
        printf("CUDA error in chamfer_forward_mixed_precision: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

// 混合精度反向传播实现
int chamfer_backward_mixed_precision(
    at::Tensor xyz1, 
    at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2
) {
    // 检查是否支持半精度计算
    if (!is_mixed_precision_supported()) {
        printf("Mixed precision computation not supported on this device!\n");
        return 0;
    }
    
    const auto batch_size = xyz1.size(0);
    const auto n = xyz1.size(1);
    const auto m = xyz2.size(1);
    
    // 检查输入
    if (!check_tensor_half(xyz1, "xyz1") || !check_tensor_half(xyz2, "xyz2") ||
        !check_tensor_half(gradxyz1, "gradxyz1") || !check_tensor_half(gradxyz2, "gradxyz2") ||
        !check_tensor_half(graddist1, "graddist1") || !check_tensor_half(graddist2, "graddist2") ||
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
    chamferDistanceGradKernelHalf<<<grid, threads_per_block, shared_mem_size, stream1>>>(
        batch_size, n, reinterpret_cast<__half*>(xyz1.data_ptr<at::Half>()), 
        m, reinterpret_cast<__half*>(xyz2.data_ptr<at::Half>()), 
        reinterpret_cast<__half*>(graddist1.data_ptr<at::Half>()), 
        idx1.data_ptr<int>(), 
        reinterpret_cast<__half*>(gradxyz1.data_ptr<at::Half>()), 
        reinterpret_cast<__half*>(gradxyz2.data_ptr<at::Half>())
    );
    
    chamferDistanceGradKernelHalf<<<grid, threads_per_block, shared_mem_size, stream2>>>(
        batch_size, m, reinterpret_cast<__half*>(xyz2.data_ptr<at::Half>()), 
        n, reinterpret_cast<__half*>(xyz1.data_ptr<at::Half>()), 
        reinterpret_cast<__half*>(graddist2.data_ptr<at::Half>()), 
        idx2.data_ptr<int>(), 
        reinterpret_cast<__half*>(gradxyz2.data_ptr<at::Half>()), 
        reinterpret_cast<__half*>(gradxyz1.data_ptr<at::Half>())
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
        printf("CUDA error in chamfer_backward_mixed_precision: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chamfer_forward", &chamfer3d::chamfer_forward, "Chamfer forward");
    m.def("chamfer_backward", &chamfer3d::chamfer_backward, "Chamfer backward");
    m.def("chamfer_forward_mixed_precision", &chamfer3d::chamfer_forward_mixed_precision, 
          "Chamfer forward with mixed precision");
    m.def("chamfer_backward_mixed_precision", &chamfer3d::chamfer_backward_mixed_precision, 
          "Chamfer backward with mixed precision");
    m.def("is_mixed_precision_supported", &chamfer3d::is_mixed_precision_supported,
          "Check if mixed precision computation is supported");
}

} // namespace chamfer3d