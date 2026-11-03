// GPU glow renderer: class map -> binary mask -> mipmap blur -> composite with the original frame.
#pragma once

#include "glow/cuda_utils.hpp"
#include "glow/glow_params.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace glow {

class GlowRenderer {
public:
    /// All work is queued on `stream` (pass the segmenter's stream to keep things ordered).
    explicit GlowRenderer(cudaStream_t stream);
    ~GlowRenderer();

    GlowRenderer(const GlowRenderer&) = delete;
    GlowRenderer& operator=(const GlowRenderer&) = delete;

    /// Copies a class map (device pointer, uint8, maskW x maskH) into the renderer's own buffer.
    /// It stays valid until the next call, so render() can be repeated while parameters change.
    void setMask(const std::uint8_t* deviceClassMap, int maskW, int maskH);
    /// Same as setMask() but from host memory (CV_8UC1).
    void setMask(const cv::Mat& hostClassMap);

    /// Renders the glow on top of `frameBgr` (CV_8UC3) using the current mask. Blocks until done.
    void render(const cv::Mat& frameBgr, const GlowParams& params, cv::Mat& outBgr);

private:
    void ensureSize(int width, int height);
    void releaseMipmap();

    cudaStream_t stream_;
    int width_ = 0, height_ = 0, levels_ = 0;
    int maskW_ = 0, maskH_ = 0;

    DeviceBuffer<std::uint8_t> classMap_;
    DeviceBuffer<std::uint8_t> frame_;  // BGR, tightly packed
    DeviceBuffer<std::uint8_t> out_;    // BGR, tightly packed

    cudaMipmappedArray_t mipmap_ = nullptr;
    std::vector<cudaSurfaceObject_t> surfaces_;     // one per level
    std::vector<cudaTextureObject_t> levelTex_;     // one per level, bilinear (2x2 box downsampling)
    cudaTextureObject_t mipTex_ = 0;                // whole pyramid, trilinear
};

}  // namespace glow
