#include "glow/cuda_utils.hpp"

namespace glow {

std::string describeCudaDevice(int device) {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0)
        throw CudaError(std::string("No usable CUDA GPU found (") + cudaGetErrorString(err) +
                        "). Check that the NVIDIA driver is installed.");
    GLOW_CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    GLOW_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::ostringstream os;
    os << prop.name << " (sm_" << prop.major << prop.minor << ", " << (prop.totalGlobalMem >> 20) << " MiB)";
    return os.str();
}

}  // namespace glow
