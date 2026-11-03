#include "glow/segmenter.hpp"

#include "glow/kernels.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace glow {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > Severity::kWARNING) return;
        const char* tag = severity == Severity::kWARNING ? "warning" : "error";
        std::cerr << "[TensorRT " << tag << "] " << msg << '\n';
    }
};

TrtLogger& logger() {
    static TrtLogger instance;  // must outlive every runtime / builder
    return instance;
}

std::string dimsToString(const nvinfer1::Dims& d) {
    std::ostringstream os;
    os << '[';
    for (int i = 0; i < d.nbDims; ++i) os << (i ? "x" : "") << d.d[i];
    os << ']';
    return os.str();
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool isOnnx(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".onnx";
}

template <typename T>
struct TrtDelete {
    void operator()(T* p) const noexcept { delete p; }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Engine building (ONNX -> plan)
// ---------------------------------------------------------------------------------------------------------------------

std::string Segmenter::cachedPlanPath(const std::string& onnxPath, const EngineBuildOptions& opt) {
    fs::path p(onnxPath);
    const std::string suffix = ".b" + std::to_string(opt.batch) + (opt.fp16 ? ".fp16" : ".fp32") + ".plan";
    return (p.parent_path() / (p.stem().string() + suffix)).string();
}

void Segmenter::buildEngine(const std::string& onnxPath, const std::string& planPath, const EngineBuildOptions& opt) {
    std::cout << "Building TensorRT engine from " << onnxPath << " (batch " << opt.batch << ", "
              << (opt.fp16 ? "FP16" : "FP32") << "). This takes a minute or two the first time...\n";

    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
    if (!builder) throw std::runtime_error("createInferBuilder failed");
#if NV_TENSORRT_MAJOR >= 10
    const uint32_t flags = 0;  // explicit batch is the only mode in TensorRT 10
#else
    const uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger()));
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::ostringstream os;
        os << "Failed to parse ONNX model " << onnxPath << ':';
        for (int i = 0; i < parser->getNbErrors(); ++i) os << "\n  " << parser->getError(i)->desc();
        throw std::runtime_error(os.str());
    }
    if (network->getNbInputs() < 1 || network->getNbOutputs() < 1)
        throw std::runtime_error("ONNX model must have at least one input and one output");

    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
#if NV_TENSORRT_MAJOR < 11
    if (opt.fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
#else
    // TensorRT 11+ networks are strongly typed: precision follows the ONNX model (export an FP16 model for FP16).
    if (opt.fp16) std::cout << "Note: TensorRT " << NV_TENSORRT_MAJOR << " uses the precision stored in the ONNX file.\n";
#endif

    // Dynamic dimensions need an optimisation profile.
    nvinfer1::ITensor* input = network->getInput(0);
    nvinfer1::Dims dims = input->getDimensions();
    bool dynamic = false;
    for (int i = 0; i < dims.nbDims; ++i) dynamic |= dims.d[i] < 0;
    if (dynamic) {
        if (dims.nbDims != 4) throw std::runtime_error("Expected an NCHW input, got " + dimsToString(dims));
        auto withBatch = [&](int b) {
            nvinfer1::Dims d = dims;
            d.d[0] = b;
            if (d.d[1] < 0) d.d[1] = 3;
            if (d.d[2] < 0) d.d[2] = opt.inputSize;
            if (d.d[3] < 0) d.d[3] = opt.inputSize;
            return d;
        };
        auto* profile = builder->createOptimizationProfile();
        const int maxB = std::max(1, opt.batch);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, withBatch(dims.d[0] < 0 ? 1 : dims.d[0]));
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, withBatch(dims.d[0] < 0 ? maxB : dims.d[0]));
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, withBatch(dims.d[0] < 0 ? maxB : dims.d[0]));
        config->addOptimizationProfile(profile);
    }

    TrtPtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan || plan->size() == 0) throw std::runtime_error("TensorRT engine build failed");

    if (fs::path(planPath).has_parent_path()) fs::create_directories(fs::path(planPath).parent_path());
    std::ofstream out(planPath, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write engine file: " + planPath);
    out.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    std::cout << "Saved engine to " << planPath << '\n';
}

// ---------------------------------------------------------------------------------------------------------------------
// Segmenter
// ---------------------------------------------------------------------------------------------------------------------

Segmenter::Segmenter(const std::string& modelPath, int preferredBatch, EngineBuildOptions build) {
    if (!fs::exists(modelPath)) throw std::runtime_error("Model file not found: " + modelPath);

    std::string plan = modelPath;
    if (isOnnx(modelPath)) {
        if (preferredBatch > 0) build.batch = preferredBatch;
        plan = cachedPlanPath(modelPath, build);
        const bool stale = !fs::exists(plan) || fs::last_write_time(plan) < fs::last_write_time(modelPath);
        if (stale) buildEngine(modelPath, plan, build);
        else std::cout << "Using cached engine " << plan << '\n';
    }
    loadEngine(plan);
    configure(preferredBatch);
}

Segmenter::~Segmenter() = default;

void Segmenter::loadEngine(const std::string& planPath) {
    enginePath_ = planPath;
    const std::vector<char> blob = readFile(planPath);
    runtime_.reset(nvinfer1::createInferRuntime(logger()));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_)
        throw std::runtime_error("Failed to deserialize " + planPath +
                                 ". Engines are tied to the GPU and TensorRT version they were built with; "
                                 "rebuild it from the ONNX model on this machine.");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("createExecutionContext failed");
}

void Segmenter::configure(int preferredBatch) {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            if (inputName_.empty()) inputName_ = name;
        } else if (outputName_.empty()) {
            outputName_ = name;
        }
    }
    if (inputName_.empty() || outputName_.empty()) throw std::runtime_error("Engine needs one input and one output");
    if (engine_->getTensorDataType(inputName_.c_str()) != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error("Engine input must be FP32 (NCHW, ImageNet-normalised RGB)");

    // Resolve the input shape, choosing a batch size inside the optimisation profile for dynamic engines.
    nvinfer1::Dims in = engine_->getTensorShape(inputName_.c_str());
    if (in.nbDims != 4) throw std::runtime_error("Expected a 4D NCHW input, got " + dimsToString(in));
    bool dynamic = false;
    for (int i = 0; i < in.nbDims; ++i) dynamic |= in.d[i] < 0;
    if (dynamic) {
        const nvinfer1::Dims opt = engine_->getProfileShape(inputName_.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
        const nvinfer1::Dims mn = engine_->getProfileShape(inputName_.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
        const nvinfer1::Dims mx = engine_->getProfileShape(inputName_.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        nvinfer1::Dims chosen = opt;
        if (in.d[0] < 0 && preferredBatch > 0)
            chosen.d[0] = static_cast<decltype(chosen.d[0] + 0)>(std::clamp<long long>(preferredBatch, mn.d[0], mx.d[0]));
        if (!context_->setInputShape(inputName_.c_str(), chosen))
            throw std::runtime_error("Cannot set input shape " + dimsToString(chosen));
        in = chosen;
    }
    if (in.d[1] != 3) throw std::runtime_error("Expected 3 input channels, got " + dimsToString(in));
    batch_ = static_cast<int>(in.d[0]);
    inH_ = static_cast<int>(in.d[2]);
    inW_ = static_cast<int>(in.d[3]);
    if (preferredBatch > 0 && preferredBatch != batch_)
        std::cout << "Note: engine batch size is fixed at " << batch_ << " (requested " << preferredBatch << ")\n";

    // Output: float logits NCHW, or int32 labels N x H x W / N x 1 x H x W.
    const nvinfer1::Dims out = context_->getTensorShape(outputName_.c_str());
    const nvinfer1::DataType outType = engine_->getTensorDataType(outputName_.c_str());
    if (outType == nvinfer1::DataType::kFLOAT && out.nbDims == 4 && out.d[1] > 1) {
        numClasses_ = static_cast<int>(out.d[1]);
        outH_ = static_cast<int>(out.d[2]);
        outW_ = static_cast<int>(out.d[3]);
    } else if (outType == nvinfer1::DataType::kINT32 && (out.nbDims == 3 || (out.nbDims == 4 && out.d[1] == 1))) {
        outputIsLabels_ = true;
        numClasses_ = 21;
        outH_ = static_cast<int>(out.d[out.nbDims - 2]);
        outW_ = static_cast<int>(out.d[out.nbDims - 1]);
    } else {
        throw std::runtime_error("Unsupported output " + outputName_ + " " + dimsToString(out) +
                                 " (need float NxCxHxW logits or int32 labels)");
    }
    if (numClasses_ > 255) throw std::runtime_error("At most 255 classes are supported");

    const std::size_t inPixels = static_cast<std::size_t>(inW_) * inH_;
    const std::size_t outPixels = static_cast<std::size_t>(outW_) * outH_;
    hostBgr_.resize(batch_ * inPixels * 3);
    devBgr_.resize(batch_ * inPixels * 3);
    devInput_.resize(batch_ * inPixels * 3);
    devOutput_.resize(batch_ * outPixels * (outputIsLabels_ ? 1 : numClasses_));
    devMasks_.resize(batch_ * outPixels);
    static_assert(sizeof(float) == sizeof(std::int32_t), "label buffer reuses the float allocation");

    if (!context_->setTensorAddress(inputName_.c_str(), devInput_.get()) ||
        !context_->setTensorAddress(outputName_.c_str(), devOutput_.get()))
        throw std::runtime_error("setTensorAddress failed");

    std::cout << "Engine: " << enginePath_ << "\n  input  " << inputName_ << ' ' << dimsToString(in) << "\n  output "
              << outputName_ << ' ' << dimsToString(out) << " -> " << numClasses_ << " classes\n";
}

void Segmenter::run(const std::vector<cv::Mat>& frames) {
    if (frames.empty()) return;
    if (static_cast<int>(frames.size()) > batch_)
        throw std::invalid_argument("Segmenter::run: more frames than the batch size");

    const std::size_t frameBytes = static_cast<std::size_t>(inW_) * inH_ * 3;
    for (int i = 0; i < batch_; ++i) {
        // Pad partial batches by repeating the last frame (static-batch engines need a full batch).
        const cv::Mat& src = frames[std::min<std::size_t>(i, frames.size() - 1)];
        if (src.empty() || src.type() != CV_8UC3) throw std::invalid_argument("Segmenter expects 8-bit BGR frames");
        cv::Mat dst(inH_, inW_, CV_8UC3, hostBgr_.get() + i * frameBytes);
        cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
    }

    cudaStream_t s = stream_.get();
    GLOW_CUDA_CHECK(cudaMemcpyAsync(devBgr_.get(), hostBgr_.get(), batch_ * frameBytes, cudaMemcpyHostToDevice, s));
    kernels::preprocessBgrToNchw(devBgr_.get(), devInput_.get(), batch_, inW_, inH_, s);
    if (!context_->enqueueV3(s)) throw std::runtime_error("TensorRT enqueueV3 failed");
    if (outputIsLabels_)
        kernels::labelsToU8(reinterpret_cast<const std::int32_t*>(devOutput_.get()), devMasks_.get(),
                            batch_ * outW_ * outH_, s);
    else
        kernels::argmaxNchw(devOutput_.get(), devMasks_.get(), batch_, numClasses_, outH_, outW_, s);
    GLOW_CUDA_CHECK(cudaGetLastError());
    stream_.synchronize();
}

const std::uint8_t* Segmenter::deviceMask(int i) const {
    if (i < 0 || i >= batch_) throw std::out_of_range("Segmenter::deviceMask index");
    return devMasks_.get() + static_cast<std::size_t>(i) * outW_ * outH_;
}

cv::Mat Segmenter::hostMask(int i) const {
    cv::Mat m(outH_, outW_, CV_8UC1);
    GLOW_CUDA_CHECK(cudaMemcpy(m.data, deviceMask(i), m.total(), cudaMemcpyDeviceToHost));
    return m;
}

}  // namespace glow
