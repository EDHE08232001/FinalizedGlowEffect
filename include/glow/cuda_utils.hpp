// Small CUDA helpers shared by the whole project: error checking and RAII buffers.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace glow {

/// Thrown whenever a CUDA runtime call fails.
class CudaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline void cudaCheck(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error '" << cudaGetErrorString(err) << "' (" << static_cast<int>(err) << ") at " << file << ':'
           << line << " in `" << expr << '`';
        throw CudaError(os.str());
    }
}

}  // namespace glow

#define GLOW_CUDA_CHECK(expr) ::glow::cudaCheck((expr), #expr, __FILE__, __LINE__)

namespace glow {

/// Owning device allocation (cudaMalloc / cudaFree).
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { resize(count); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), count_(std::exchange(o.count_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = std::exchange(o.ptr_, nullptr);
            count_ = std::exchange(o.count_, 0);
        }
        return *this;
    }

    /// Reallocates only when the requested size differs (contents are not preserved).
    void resize(std::size_t count) {
        if (count == count_) return;
        release();
        if (count > 0) GLOW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        count_ = count;
    }

    void release() noexcept {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

/// Owning page-locked host allocation (cudaMallocHost / cudaFreeHost) for fast async copies.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t count) { resize(count); }
    ~PinnedBuffer() { release(); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void resize(std::size_t count) {
        if (count == count_) return;
        release();
        if (count > 0) GLOW_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        count_ = count;
    }

    void release() noexcept {
        if (ptr_) cudaFreeHost(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

/// Owning non-blocking CUDA stream.
class CudaStream {
public:
    CudaStream() { GLOW_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~CudaStream() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const { return stream_; }
    void synchronize() const { GLOW_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

/// Returns "<name> (sm_XY, N MiB)" for the current device, or throws if no CUDA device is usable.
std::string describeCudaDevice(int device = 0);

}  // namespace glow
