// Host-side launchers for the CUDA kernels in src/cuda/*.cu.
// All launchers are asynchronous on the given stream.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace glow::kernels {

/// BGR uint8 HWC images (batch, contiguous) -> RGB float NCHW, normalised with ImageNet mean/std.
void preprocessBgrToNchw(const std::uint8_t* d_bgr, float* d_out, int batch, int width, int height,
                         cudaStream_t stream);

/// Per-pixel argmax over channels of float NCHW logits -> uint8 class index (N x H x W).
void argmaxNchw(const float* d_logits, std::uint8_t* d_classes, int batch, int channels, int height, int width,
                cudaStream_t stream);

/// int32 label map -> uint8 class index (for models that already output labels).
void labelsToU8(const std::int32_t* d_labels, std::uint8_t* d_classes, int count, cudaStream_t stream);

}  // namespace glow::kernels
