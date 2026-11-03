// TensorRT semantic segmentation: frames in, per-pixel class maps out (kept on the GPU).
#pragma once

#include "glow/cuda_utils.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}  // namespace nvinfer1

namespace glow {

struct EngineBuildOptions {
    int batch = 4;        ///< Max/opt batch for dynamic-batch ONNX models.
    int inputSize = 384;  ///< Used only when the ONNX input has dynamic H/W.
    bool fp16 = true;     ///< Build with FP16 kernels when the GPU supports them.
};

class Segmenter {
public:
    /// `modelPath` may be a serialized TensorRT engine (.plan / .engine / .trt) or an ONNX model.
    /// ONNX models are compiled once and cached as "<model>.b<batch>.<fp16|fp32>.plan" next to the file.
    /// `preferredBatch` = 0 uses the engine's own (or optimal) batch size.
    explicit Segmenter(const std::string& modelPath, int preferredBatch = 0, EngineBuildOptions build = {});
    ~Segmenter();

    Segmenter(const Segmenter&) = delete;
    Segmenter& operator=(const Segmenter&) = delete;

    /// Segments `frames` (BGR, any size; at most batchSize()). Blocks until the result is ready.
    void run(const std::vector<cv::Mat>& frames);

    /// Device pointer to the class map of frame `i` from the last run() (maskWidth() x maskHeight(), uint8).
    const std::uint8_t* deviceMask(int i) const;
    /// Host copy of the class map of frame `i` from the last run().
    cv::Mat hostMask(int i) const;

    int batchSize() const { return batch_; }
    int inputWidth() const { return inW_; }
    int inputHeight() const { return inH_; }
    int maskWidth() const { return outW_; }
    int maskHeight() const { return outH_; }
    int numClasses() const { return numClasses_; }
    cudaStream_t stream() const { return stream_.get(); }
    const std::string& enginePath() const { return enginePath_; }

    /// Compiles an ONNX model into a TensorRT engine file.
    static void buildEngine(const std::string& onnxPath, const std::string& planPath, const EngineBuildOptions& opt);
    /// Path that the ONNX cache for the given options would use.
    static std::string cachedPlanPath(const std::string& onnxPath, const EngineBuildOptions& opt);

private:
    void loadEngine(const std::string& planPath);
    void configure(int preferredBatch);

    struct Destroyer {
        template <typename T>
        void operator()(T* p) const noexcept { delete p; }
    };

    std::unique_ptr<nvinfer1::IRuntime, Destroyer> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, Destroyer> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, Destroyer> context_;

    CudaStream stream_;
    std::string enginePath_;
    std::string inputName_, outputName_;
    bool outputIsLabels_ = false;  // int32 labels instead of float logits
    int batch_ = 1, inW_ = 0, inH_ = 0, outW_ = 0, outH_ = 0, numClasses_ = 0;

    PinnedBuffer<std::uint8_t> hostBgr_;  // batch x H x W x 3 staging
    DeviceBuffer<std::uint8_t> devBgr_;
    DeviceBuffer<float> devInput_;
    DeviceBuffer<float> devOutput_;        // float logits or int32 labels (same element size)
    DeviceBuffer<std::uint8_t> devMasks_;  // batch x outH x outW
};

}  // namespace glow
