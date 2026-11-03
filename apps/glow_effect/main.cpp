// Glow Effect - entry point.
//
//   input (image / folder / video / camera)
//     -> Segmenter   (TensorRT, per-pixel classes)
//     -> GlowRenderer (CUDA: mask -> mipmap blur -> composite)
//     -> Studio UI preview and/or files in ./output
#include "cli.hpp"
#include "modes.hpp"

#include "glow/cuda_utils.hpp"
#include "glow/glow_params.hpp"
#include "glow/segmenter.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

fs::path executableDir(const char* argv0) {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return fs::path(buf).parent_path();
#else
    std::error_code ec;
    const fs::path self = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return self.parent_path();
#endif
    std::error_code ec2;
    const fs::path p = fs::absolute(argv0, ec2);
    return ec2 ? fs::current_path() : p.parent_path();
}

/// --engine, then $GLOW_ENGINE, then models/ next to the working dir or the executable.
std::string findEngine(const std::string& requested, const char* argv0) {
    if (!requested.empty()) return requested;
    if (const char* env = std::getenv("GLOW_ENGINE"); env && *env) return env;

    const fs::path exe = executableDir(argv0);
    const std::vector<fs::path> dirs = {fs::current_path() / "models", exe / "models", exe / ".." / "models",
                                        exe / ".." / ".." / "models", exe / ".." / ".." / ".." / "models"};
    for (const char* group : {"engine", "onnx"}) {
        for (const auto& d : dirs) {
            std::error_code ec;
            if (!fs::is_directory(d, ec)) continue;
            std::vector<fs::path> found;
            for (const auto& e : fs::directory_iterator(d, ec)) {
                const std::string ext = e.path().extension().string();
                const bool isEngine = ext == ".plan" || ext == ".engine" || ext == ".trt";
                if (e.is_regular_file() && (group[0] == 'e' ? isEngine : ext == ".onnx")) found.push_back(e.path());
            }
            if (!found.empty()) {
                std::sort(found.begin(), found.end());
                return fs::weakly_canonical(found.front()).string();
            }
        }
    }
    return {};
}

std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && app::isImageFile(e.path().string())) out.push_back(e.path().string());
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    app::Options opt;
    if (auto code = app::parseArgs(argc, argv, opt)) return *code;

    if (opt.listClasses) {
        const auto& names = glow::vocClassNames();
        std::cout << "Pascal VOC classes (use the name or the number with --class):\n";
        for (std::size_t i = 0; i < names.size(); ++i) std::printf("  %2zu  %s\n", i, names[i].c_str());
        return 0;
    }

    try {
        glow::EngineBuildOptions build;
        build.fp16 = opt.fp16;
        build.inputSize = opt.inputSize;
        if (opt.batch > 0) build.batch = opt.batch;

        // ---- Engine build only ------------------------------------------------------------------------------------
        if (!opt.buildOnnx.empty()) {
            const std::string plan =
                opt.buildPlan.empty() ? glow::Segmenter::cachedPlanPath(opt.buildOnnx, build) : opt.buildPlan;
            glow::Segmenter::buildEngine(opt.buildOnnx, plan, build);
            return 0;
        }

        // ---- Input ------------------------------------------------------------------------------------------------
        if (opt.input.empty()) {
            app::printUsage(argv[0]);
            std::cout << "\nNo input given. Type (or drag in) an image, folder or video path, or 'camera': ";
            std::getline(std::cin, opt.input);
            opt.input = app::cleanPath(opt.input);
            if (opt.input.empty()) return 2;
        }
        int camera = 0;
        const app::InputKind kind = app::detectInput(opt.input, &camera);

        // ---- Model ------------------------------------------------------------------------------------------------
        const std::string engine = findEngine(opt.engine, argv[0]);
        if (engine.empty()) {
            std::cerr << "Error: no model found. Pass --engine <file.plan|file.onnx>, set GLOW_ENGINE, or put a model\n"
                         "in ./models (run scripts/export_onnx.py to create one). See README.md.\n";
            return 2;
        }
        const std::string gpu = glow::describeCudaDevice();
        std::cout << "GPU: " << gpu << '\n';
        const int batch = opt.batch > 0 ? opt.batch : (kind == app::InputKind::Camera ? 1 : 0);
        glow::Segmenter seg(engine, batch, build);

        if (!opt.classText.empty()) {
            const int cls = glow::parseClass(opt.classText, seg.numClasses());
            if (cls < 0) {
                std::cerr << "Error: unknown class '" << opt.classText << "' (see --list-classes)\n";
                return 2;
            }
            opt.params.targetClass = cls;
        }
        opt.params.targetClass = std::clamp(opt.params.targetClass, 0, seg.numClasses() - 1);
        std::cout << "Glowing class: " << glow::className(opt.params.targetClass, seg.numClasses()) << '\n';

        switch (kind) {
            case app::InputKind::Image: return app::runImages({opt.input}, seg, opt);
            case app::InputKind::Directory: {
                const auto images = listImages(opt.input);
                if (images.empty()) {
                    std::cerr << "Error: no images (.jpg .png .bmp .tif .webp) in " << opt.input << '\n';
                    return 1;
                }
                return app::runImages(images, seg, opt);
            }
            case app::InputKind::Video: return app::runVideo(opt.input, 0, seg, opt);
            case app::InputKind::Camera: return app::runVideo("", camera, seg, opt);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
