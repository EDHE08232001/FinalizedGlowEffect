#include "cli.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace app {
namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

int toInt(const std::string& flag, const std::string& v, int lo, int hi) {
    try {
        std::size_t used = 0;
        const int x = std::stoi(v, &used);
        if (used != v.size() || x < lo || x > hi) throw std::out_of_range(v);
        return x;
    } catch (const std::exception&) {
        throw std::invalid_argument(flag + " expects an integer in [" + std::to_string(lo) + ", " +
                                    std::to_string(hi) + "], got '" + v + "'");
    }
}

}  // namespace

void printUsage(const char* argv0) {
    const std::string exe = fs::path(argv0).filename().string();
    std::cout <<
        "Glow Effect - TensorRT segmentation + CUDA mipmap glow\n\n"
        "Usage:\n"
        "  " << exe << " [options] <input>\n"
        "  " << exe << " --build-engine <model.onnx> [engine.plan] [--batch N] [--fp32]\n\n"
        "<input> is an image, a folder of images, a video file, or 'camera' / 'camera:N'.\n"
        "Leave it out to be asked interactively (you can drag a file into the terminal).\n\n"
        "Options:\n"
        "  -e, --engine PATH     TensorRT engine (.plan/.engine) or ONNX model (.onnx, compiled + cached)\n"
        "                        default: $GLOW_ENGINE, then the first model found in ./models\n"
        "  -o, --output PATH     output file (video/image) or folder (default: ./output)\n"
        "  -c, --class NAME|ID   object class that glows (default: person). See --list-classes\n"
        "  -i, --intensity N     glow strength in percent, 0-200 (default 90)\n"
        "  -s, --size N          glow radius, 0-100 (default 33)\n"
        "      --hue N           glow colour hue in degrees, 0-360 (default 300 = purple)\n"
        "  -m, --mode MODE       blend | add (default blend)\n"
        "  -b, --batch N         frames per inference batch (dynamic engines only; default: engine's)\n"
        "      --input-size N    network resolution when compiling ONNX models with dynamic H/W (384)\n"
        "      --fp32            build ONNX engines in FP32 instead of FP16\n"
        "      --no-display      headless: no windows, just write results\n"
        "      --no-save         do not write the processed video to disk\n"
        "      --list-classes    print the class names and exit\n"
        "  -h, --help            show this help\n\n"
        "Examples:\n"
        "  " << exe << " photos/dog.jpg --class dog\n"
        "  " << exe << " clips/street.mp4 -c car --mode add -o output/street_glow.mp4\n"
        "  " << exe << " camera --class person\n";
}

std::optional<int> parseArgs(int argc, char** argv, Options& opt) {
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "-h" || a == "--help") {
                printUsage(argv[0]);
                return 0;
            } else if (a == "-e" || a == "--engine") {
                opt.engine = cleanPath(next());
            } else if (a == "-o" || a == "--output") {
                opt.output = cleanPath(next());
            } else if (a == "-c" || a == "--class") {
                opt.classText = next();
            } else if (a == "-i" || a == "--intensity") {
                opt.params.intensity = toInt(a, next(), 0, 200);
            } else if (a == "-s" || a == "--size") {
                opt.params.size = toInt(a, next(), 0, 100);
            } else if (a == "--hue") {
                opt.params.hue = toInt(a, next(), 0, 360);
            } else if (a == "-m" || a == "--mode") {
                const std::string m = lower(next());
                if (m == "blend") opt.params.mode = glow::BlendMode::Blend;
                else if (m == "add" || m == "additive") opt.params.mode = glow::BlendMode::Additive;
                else throw std::invalid_argument("--mode must be 'blend' or 'add'");
            } else if (a == "-b" || a == "--batch") {
                opt.batch = toInt(a, next(), 1, 64);
            } else if (a == "--input-size") {
                opt.inputSize = toInt(a, next(), 32, 4096);
            } else if (a == "--fp32") {
                opt.fp16 = false;
            } else if (a == "--no-display" || a == "--headless") {
                opt.display = false;
            } else if (a == "--no-save") {
                opt.save = false;
            } else if (a == "--list-classes") {
                opt.listClasses = true;
            } else if (a == "--build-engine") {
                opt.buildOnnx = cleanPath(next());
                if (i + 1 < argc && argv[i + 1][0] != '-') opt.buildPlan = cleanPath(argv[++i]);
            } else if (!a.empty() && a[0] == '-' && a != "-") {
                throw std::invalid_argument("Unknown option " + a + " (see --help)");
            } else if (opt.input.empty()) {
                opt.input = cleanPath(a);
            } else {
                throw std::invalid_argument("Only one input is supported (got '" + opt.input + "' and '" + a + "')");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }
    return std::nullopt;
}

std::string cleanPath(std::string s) {
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && isSpace(s.back())) s.pop_back();
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](unsigned char c) { return !isSpace(c); }));
    if (s.size() >= 2 && (s.front() == '"' || s.front() == '\'') && s.back() == s.front()) s = s.substr(1, s.size() - 2);
#ifndef _WIN32
    // Terminals on macOS/Linux escape spaces when files are dragged in: "my\ file.mp4".
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) ++i;
        out += s[i];
    }
    s = out;
#endif
    return s;
}

bool isImageFile(const std::string& path) {
    static const char* exts[] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
    const std::string e = lower(fs::path(path).extension().string());
    return std::any_of(std::begin(exts), std::end(exts), [&](const char* x) { return e == x; });
}

InputKind detectInput(const std::string& input, int* cameraIndex) {
    const std::string l = lower(input);
    if (l == "camera" || l == "cam" || l == "webcam" || l.rfind("camera:", 0) == 0) {
        const auto colon = l.find(':');
        *cameraIndex = colon == std::string::npos ? 0 : toInt("camera", l.substr(colon + 1), 0, 99);
        return InputKind::Camera;
    }
    if (!fs::exists(input)) throw std::runtime_error("Input not found: " + input);
    if (fs::is_directory(input)) return InputKind::Directory;
    return isImageFile(input) ? InputKind::Image : InputKind::Video;
}

}  // namespace app
