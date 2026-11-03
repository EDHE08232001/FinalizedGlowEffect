// Pre/post-processing kernels around the TensorRT segmentation network.
#include "glow/kernels.hpp"

#include <cfloat>

namespace glow::kernels {
namespace {

__constant__ float kMean[3] = {0.485f, 0.456f, 0.406f};  // RGB, ImageNet
__constant__ float kInvStd[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

__global__ void preprocessKernel(const unsigned char* __restrict__ bgr, float* __restrict__ out, int width,
                                 int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= width || y >= height) return;

    const size_t plane = static_cast<size_t>(width) * height;
    const size_t pix = static_cast<size_t>(y) * width + x;
    const unsigned char* p = bgr + (b * plane + pix) * 3;
    float* o = out + b * plane * 3 + pix;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float v = p[2 - c] * (1.0f / 255.0f);  // BGR -> RGB
        o[c * plane] = (v - kMean[c]) * kInvStd[c];
    }
}

__global__ void argmaxKernel(const float* __restrict__ logits, unsigned char* __restrict__ classes, int channels,
                             int height, int width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= width || y >= height) return;

    const size_t plane = static_cast<size_t>(width) * height;
    const size_t pix = static_cast<size_t>(y) * width + x;
    const float* p = logits + b * plane * channels + pix;
    float best = -FLT_MAX;
    int bestIdx = 0;
    for (int c = 0; c < channels; ++c) {
        const float v = p[c * plane];
        if (v > best) {
            best = v;
            bestIdx = c;
        }
    }
    classes[b * plane + pix] = static_cast<unsigned char>(bestIdx);
}

__global__ void labelsKernel(const int* __restrict__ labels, unsigned char* __restrict__ classes, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) classes[i] = static_cast<unsigned char>(min(max(labels[i], 0), 255));
}

}  // namespace

void preprocessBgrToNchw(const std::uint8_t* d_bgr, float* d_out, int batch, int width, int height,
                         cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch);
    preprocessKernel<<<grid, block, 0, stream>>>(d_bgr, d_out, width, height);
}

void argmaxNchw(const float* d_logits, std::uint8_t* d_classes, int batch, int channels, int height, int width,
                cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch);
    argmaxKernel<<<grid, block, 0, stream>>>(d_logits, d_classes, channels, height, width);
}

void labelsToU8(const std::int32_t* d_labels, std::uint8_t* d_classes, int count, cudaStream_t stream) {
    const int block = 256;
    labelsKernel<<<(count + block - 1) / block, block, 0, stream>>>(d_labels, d_classes, count);
}

}  // namespace glow::kernels
