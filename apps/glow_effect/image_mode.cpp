#include "modes.hpp"
#include "studio_ui.hpp"

#include "glow/glow_renderer.hpp"
#include "glow/segmenter.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace app {

std::string prepareOutputPath(const std::string& path) {
    const fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    return path;
}

namespace {

/// Where the result for `image` goes: an explicit file (single image), or <folder>/<stem>_glow.png.
std::string outputFor(const std::string& image, const Options& opt, std::size_t count) {
    if (!opt.output.empty() && count == 1 && isImageFile(opt.output)) return opt.output;
    const fs::path dir = opt.output.empty() ? fs::path("output") : fs::path(opt.output);
    return (dir / (fs::path(image).stem().string() + "_glow.png")).string();
}

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int runImages(const std::vector<std::string>& images, glow::Segmenter& seg, Options& opt) {
    glow::GlowRenderer renderer(seg.stream());

    // ---- Headless: process everything and write the results. ----------------------------------------------------
    if (!opt.display) {
        int failures = 0;
        for (const auto& path : images) {
            const cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Skipping unreadable image " << path << '\n';
                ++failures;
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            seg.run({img});
            renderer.setMask(seg.deviceMask(0), seg.maskWidth(), seg.maskHeight());
            cv::Mat out;
            renderer.render(img, opt.params, out);
            const double ms = msSince(t0);
            const std::string dst = prepareOutputPath(outputFor(path, opt, images.size()));
            if (!cv::imwrite(dst, out)) {
                std::cerr << "Could not write " << dst << '\n';
                ++failures;
                continue;
            }
            std::printf("%s -> %s  (%.1f ms)\n", path.c_str(), dst.c_str(), ms);
        }
        return failures == 0 ? 0 : 1;
    }

    // ---- Interactive studio. ------------------------------------------------------------------------------------
    StudioUi ui(opt.params, seg.numClasses(), /*isVideo=*/false);
    std::vector<std::string> list = images;
    std::size_t index = 0;
    bool needLoad = true, needRender = false, needShow = false;
    cv::Mat img, classMap, result;
    UiStatus status;

    while (!list.empty()) {
        if (needLoad) {
            img = cv::imread(list[index], cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Skipping unreadable image " << list[index] << '\n';
                list.erase(list.begin() + static_cast<std::ptrdiff_t>(index));
                if (index >= list.size()) index = 0;
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            seg.run({img});
            classMap = seg.hostMask(0);
            renderer.setMask(seg.deviceMask(0), seg.maskWidth(), seg.maskHeight());
            std::printf("[%zu/%zu] %s  %dx%d  segmentation %.1f ms\n", index + 1, list.size(), list[index].c_str(),
                        img.cols, img.rows, msSince(t0));
            status.source = fs::path(list[index]).filename().string() + "  (" + std::to_string(index + 1) + "/" +
                            std::to_string(list.size()) + ")";
            needLoad = false;
            needRender = true;
        }
        if (ui.consumeParamsChanged() || needRender) {
            renderer.render(img, opt.params, result);
            needRender = false;
            needShow = true;
        }
        if (ui.consumeViewChanged() || needShow) {
            ui.show(img, result, classMap, status);
            needShow = false;
        }

        switch (ui.poll(30)) {
            case UiAction::Quit: return 0;
            case UiAction::Next:
                index = (index + 1) % list.size();
                needLoad = true;
                break;
            case UiAction::Prev:
                index = (index + list.size() - 1) % list.size();
                needLoad = true;
                break;
            case UiAction::Save: {
                const std::string dst = prepareOutputPath(outputFor(list[index], opt, list.size()));
                if (cv::imwrite(dst, result)) {
                    std::cout << "Saved " << dst << '\n';
                    ui.notify("Saved " + dst);
                } else {
                    ui.notify("Could not write " + dst);
                }
                break;
            }
            default: break;
        }
    }
    std::cerr << "No readable images.\n";
    return 1;
}

}  // namespace app
