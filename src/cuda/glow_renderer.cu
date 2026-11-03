// GPU glow renderer.
//
// 1. extractMask  : class map (model resolution) -> binary mask written into mip level 0 (frame resolution)
// 2. downsample   : builds the mip pyramid with 2x2 box filtering (one bilinear fetch per output texel)
// 3. composite    : samples the pyramid at LOD = size/10 (trilinear + 5 taps) to get a soft halo,
//                   then blends the glow colour into the frame.
#include "glow/glow_renderer.hpp"

#include <algorithm>
#include <cmath>

namespace glow {
namespace {

__global__ void extractMaskKernel(cudaSurfaceObject_t level0, const unsigned char* __restrict__ classMap, int maskW,
                                  int maskH, int width, int height, int target) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int mx = min(maskW - 1, (x * maskW) / width);
    const int my = min(maskH - 1, (y * maskH) / height);
    const unsigned char v = classMap[my * maskW + mx] == target ? 255 : 0;
    surf2Dwrite(v, level0, x * static_cast<int>(sizeof(unsigned char)), y);
}

__global__ void downsampleKernel(cudaSurfaceObject_t dst, cudaTextureObject_t src, int dstW, int dstH) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;
    // Unnormalised coords: texel centres sit at i + 0.5, so (2x+1, 2y+1) averages the 2x2 block bilinearly.
    const float v = tex2D<float>(src, 2.0f * x + 1.0f, 2.0f * y + 1.0f);
    surf2Dwrite(static_cast<unsigned char>(__saturatef(v) * 255.0f + 0.5f), dst, x, y);
}

__global__ void compositeKernel(cudaTextureObject_t pyramid, const unsigned char* __restrict__ frame,
                                unsigned char* __restrict__ out, int width, int height, float lod, float strength,
                                float3 color, int mode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float u = (x + 0.5f) / width;
    const float v = (y + 0.5f) / height;
    const float off = 0.75f * exp2f(lod);  // spread the taps by ~one texel of the sampled level
    const float du = off / width, dv = off / height;
    float a = 2.0f * tex2DLod<float>(pyramid, u, v, lod);
    a += tex2DLod<float>(pyramid, u + du, v + dv, lod);
    a += tex2DLod<float>(pyramid, u - du, v + dv, lod);
    a += tex2DLod<float>(pyramid, u + du, v - dv, lod);
    a += tex2DLod<float>(pyramid, u - du, v - dv, lod);
    a = __saturatef(a * (1.0f / 6.0f) * strength);

    const size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    const float c[3] = {color.x, color.y, color.z};
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float s = frame[idx + k];
        const float o = (mode == 1) ? s + c[k] * a : s + (c[k] - s) * a;
        out[idx + k] = static_cast<unsigned char>(fminf(fmaxf(o, 0.0f), 255.0f) + 0.5f);
    }
}

dim3 gridFor(int w, int h, dim3 block) { return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y); }

}  // namespace

GlowRenderer::GlowRenderer(cudaStream_t stream) : stream_(stream) {}

GlowRenderer::~GlowRenderer() { releaseMipmap(); }

void GlowRenderer::releaseMipmap() {
    if (mipTex_) cudaDestroyTextureObject(mipTex_);
    for (auto t : levelTex_) cudaDestroyTextureObject(t);
    for (auto s : surfaces_) cudaDestroySurfaceObject(s);
    if (mipmap_) cudaFreeMipmappedArray(mipmap_);
    mipTex_ = 0;
    levelTex_.clear();
    surfaces_.clear();
    mipmap_ = nullptr;
    width_ = height_ = levels_ = 0;
}

void GlowRenderer::ensureSize(int width, int height) {
    if (width == width_ && height == height_) return;
    releaseMipmap();

    int levels = 1;
    for (int m = std::max(width, height); m > 1; m >>= 1) ++levels;

    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    const cudaExtent extent = make_cudaExtent(width, height, 0);
    GLOW_CUDA_CHECK(cudaMallocMipmappedArray(&mipmap_, &desc, extent, levels, cudaArraySurfaceLoadStore));

    for (int l = 0; l < levels; ++l) {
        cudaArray_t level = nullptr;
        GLOW_CUDA_CHECK(cudaGetMipmappedArrayLevel(&level, mipmap_, l));

        cudaResourceDesc res = {};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = level;
        cudaSurfaceObject_t surf = 0;
        GLOW_CUDA_CHECK(cudaCreateSurfaceObject(&surf, &res));
        surfaces_.push_back(surf);

        cudaTextureDesc td = {};
        td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode = cudaFilterModeLinear;
        td.readMode = cudaReadModeNormalizedFloat;
        td.normalizedCoords = 0;
        cudaTextureObject_t tex = 0;
        GLOW_CUDA_CHECK(cudaCreateTextureObject(&tex, &res, &td, nullptr));
        levelTex_.push_back(tex);
    }

    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeMipmappedArray;
    res.res.mipmap.mipmap = mipmap_;
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    td.mipmapFilterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeNormalizedFloat;
    td.normalizedCoords = 1;
    td.maxMipmapLevelClamp = static_cast<float>(levels - 1);
    GLOW_CUDA_CHECK(cudaCreateTextureObject(&mipTex_, &res, &td, nullptr));

    const std::size_t bytes = static_cast<std::size_t>(width) * height * 3;
    frame_.resize(bytes);
    out_.resize(bytes);
    width_ = width;
    height_ = height;
    levels_ = levels;
}

void GlowRenderer::setMask(const std::uint8_t* deviceClassMap, int maskW, int maskH) {
    classMap_.resize(static_cast<std::size_t>(maskW) * maskH);
    GLOW_CUDA_CHECK(cudaMemcpyAsync(classMap_.get(), deviceClassMap, classMap_.bytes(), cudaMemcpyDeviceToDevice, stream_));
    maskW_ = maskW;
    maskH_ = maskH;
}

void GlowRenderer::setMask(const cv::Mat& hostClassMap) {
    CV_Assert(hostClassMap.type() == CV_8UC1 && hostClassMap.isContinuous());
    classMap_.resize(hostClassMap.total());
    GLOW_CUDA_CHECK(cudaMemcpyAsync(classMap_.get(), hostClassMap.data, classMap_.bytes(), cudaMemcpyHostToDevice, stream_));
    GLOW_CUDA_CHECK(cudaStreamSynchronize(stream_));  // host memory may go away after we return
    maskW_ = hostClassMap.cols;
    maskH_ = hostClassMap.rows;
}

void GlowRenderer::render(const cv::Mat& frameBgr, const GlowParams& params, cv::Mat& outBgr) {
    CV_Assert(frameBgr.type() == CV_8UC3 && !frameBgr.empty());
    if (!classMap_.get()) throw std::logic_error("GlowRenderer::render called before setMask");

    const int w = frameBgr.cols, h = frameBgr.rows;
    ensureSize(w, h);

    const std::size_t rowBytes = static_cast<std::size_t>(w) * 3;
    GLOW_CUDA_CHECK(cudaMemcpy2DAsync(frame_.get(), rowBytes, frameBgr.data, frameBgr.step, rowBytes, h,
                                      cudaMemcpyHostToDevice, stream_));

    const dim3 block(32, 8);
    extractMaskKernel<<<gridFor(w, h, block), block, 0, stream_>>>(surfaces_[0], classMap_.get(), maskW_, maskH_, w, h,
                                                                    params.targetClass);

    // Only build as many levels as the requested blur needs (+1 for trilinear interpolation).
    const float lod = std::clamp(params.lod(), 0.0f, static_cast<float>(levels_ - 1));
    const int needed = std::min(levels_ - 1, static_cast<int>(std::ceil(lod)) + 2);
    int lw = w, lh = h;
    for (int l = 1; l <= needed; ++l) {
        lw = std::max(1, lw / 2);
        lh = std::max(1, lh / 2);
        downsampleKernel<<<gridFor(lw, lh, block), block, 0, stream_>>>(surfaces_[l], levelTex_[l - 1], lw, lh);
    }

    const auto c = params.colorBgr();
    compositeKernel<<<gridFor(w, h, block), block, 0, stream_>>>(mipTex_, frame_.get(), out_.get(), w, h, lod,
                                                                  params.strength(), make_float3(c[0], c[1], c[2]),
                                                                  static_cast<int>(params.mode));
    GLOW_CUDA_CHECK(cudaGetLastError());

    outBgr.create(h, w, CV_8UC3);
    GLOW_CUDA_CHECK(cudaMemcpy2DAsync(outBgr.data, outBgr.step, out_.get(), rowBytes, rowBytes, h,
                                      cudaMemcpyDeviceToHost, stream_));
    GLOW_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace glow
