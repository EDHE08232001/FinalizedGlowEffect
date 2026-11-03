#include "bounded_queue.hpp"
#include "modes.hpp"
#include "studio_ui.hpp"

#include "glow/glow_renderer.hpp"
#include "glow/segmenter.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace app {
namespace {

std::string lowerExt(const std::string& path) {
    std::string e = fs::path(path).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return std::tolower(c); });
    return e;
}

/// Opens a writer, falling back to MJPG/.avi when the platform has no MP4 encoder.
bool openWriter(cv::VideoWriter& w, std::string& path, double fps, cv::Size size) {
    const std::string ext = lowerExt(path);
    const int fourcc = ext == ".avi" ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G') : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    prepareOutputPath(path);
    if (w.open(path, fourcc, fps, size)) return true;
    if (ext != ".avi") {
        path = (fs::path(path).replace_extension(".avi")).string();
        std::cerr << "MP4 encoder unavailable, falling back to MJPG: " << path << '\n';
        return w.open(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size);
    }
    return false;
}

std::string formatEta(double seconds) {
    if (seconds < 0 || seconds > 1e6) return "--:--";
    const int s = static_cast<int>(seconds + 0.5);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d", s / 60, s % 60);
    return buf;
}

}  // namespace

int runVideo(const std::string& source, int cameraIndex, glow::Segmenter& seg, Options& opt) {
    const bool isCamera = source.empty();
    cv::VideoCapture cap;
    if (isCamera ? !cap.open(cameraIndex) : !cap.open(source)) {
        std::cerr << "Error: could not open " << (isCamera ? "camera " + std::to_string(cameraIndex) : source) << '\n';
        return 1;
    }
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!(fps > 1.0 && fps < 1000.0)) fps = 30.0;
    const long long total = isCamera ? 0 : static_cast<long long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const std::string stem = isCamera ? "camera" + std::to_string(cameraIndex) : fs::path(source).stem().string();
    std::printf("Input: %s  %dx%d @ %.2f fps%s\n", isCamera ? stem.c_str() : source.c_str(), width, height, fps,
                total > 0 ? ("  (" + std::to_string(total) + " frames)").c_str() : "");

    // ---- Output -------------------------------------------------------------------------------------------------
    cv::VideoWriter writer;
    std::string outPath;
    const bool wantOutput = opt.save && (!isCamera || !opt.output.empty());
    if (wantOutput) {
        if (opt.output.empty()) outPath = (fs::path("output") / (stem + "_glow.mp4")).string();
        else if (fs::is_directory(opt.output) || lowerExt(opt.output).empty())
            outPath = (fs::path(opt.output) / (stem + "_glow.mp4")).string();
        else outPath = opt.output;
        if (!openWriter(writer, outPath, fps, cv::Size(width, height))) {
            std::cerr << "Error: could not open output video " << outPath << '\n';
            return 1;
        }
        std::cout << "Writing " << outPath << '\n';
    }

    // ---- Pipeline: reader thread -> [GPU on this thread] -> writer thread -----------------------------------------
    const int batch = seg.batchSize();
    BoundedQueue<cv::Mat> inQueue(static_cast<std::size_t>(batch) * 3);
    BoundedQueue<cv::Mat> outQueue(static_cast<std::size_t>(batch) * 3 + 8);

    std::thread reader([&] {
        cv::Mat frame;
        while (cap.read(frame) && !frame.empty()) {
            if (frame.type() != CV_8UC3) continue;
            if (!inQueue.push(frame.clone())) break;
        }
        inQueue.close();
    });
    std::thread writerThread([&] {
        while (auto f = outQueue.pop()) writer.write(*f);
    });

    glow::GlowRenderer renderer(seg.stream());
    std::optional<StudioUi> ui;
    if (opt.display) ui.emplace(opt.params, seg.numClasses(), /*isVideo=*/true);

    std::vector<cv::Mat> frames;
    cv::Mat lastFrame, lastResult, lastMask;
    UiStatus status;
    long long done = 0;
    double avgFps = 0.0;
    bool paused = false, quit = false;
    auto lastReport = Clock::now();
    const auto start = Clock::now();

    auto updateStatus = [&] {
        status.source = total > 0 ? "frame " + std::to_string(done) + "/" + std::to_string(total)
                                  : "frame " + std::to_string(done);
        status.fps = avgFps;
        status.paused = paused;
    };
    auto saveSnapshot = [&] {
        if (lastResult.empty()) return;
        const std::string dst = prepareOutputPath(
            (fs::path(opt.output.empty() || !fs::is_directory(opt.output) ? "output" : opt.output) /
             (stem + "_frame" + std::to_string(done) + ".png")).string());
        const bool ok = cv::imwrite(dst, lastResult);
        if (ui) ui->notify(ok ? "Saved " + dst : "Could not write " + dst);
        std::cout << (ok ? "Saved " : "Could not write ") << dst << '\n';
    };
    auto handle = [&](UiAction a) {
        if (a == UiAction::Quit) quit = true;
        else if (a == UiAction::TogglePause) paused = !paused;
        else if (a == UiAction::Save) saveSnapshot();
    };

    while (!quit) {
        // Paused: keep the last frame on screen and re-render it when parameters change.
        if (paused && ui) {
            if (ui->consumeParamsChanged() && !lastFrame.empty()) {
                renderer.setMask(lastMask);
                renderer.render(lastFrame, opt.params, lastResult);
            }
            updateStatus();
            if (ui->consumeViewChanged()) ui->show(lastFrame, lastResult, lastMask, status);
            handle(ui->poll(30));
            if (!paused) ui->show(lastFrame, lastResult, lastMask, status);
            continue;
        }

        // Gather a batch. Files wait for a full batch; cameras take what is there to keep latency low.
        frames.clear();
        auto first = inQueue.pop();
        if (!first) break;
        frames.push_back(std::move(*first));
        while (static_cast<int>(frames.size()) < batch) {
            auto f = isCamera ? inQueue.tryPop() : inQueue.pop();
            if (!f) break;
            frames.push_back(std::move(*f));
        }

        const auto t0 = Clock::now();
        seg.run(frames);
        for (std::size_t i = 0; i < frames.size() && !quit; ++i) {
            renderer.setMask(seg.deviceMask(static_cast<int>(i)), seg.maskWidth(), seg.maskHeight());
            cv::Mat out;
            renderer.render(frames[i], opt.params, out);
            ++done;
            if (writer.isOpened()) outQueue.push(out);
            lastFrame = frames[i];
            lastResult = out;
            if (ui) {
                lastMask = seg.hostMask(static_cast<int>(i));
                updateStatus();
                ui->show(frames[i], out, lastMask, status);
                handle(ui->poll(1));
            }
        }
        const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        const double batchFps = frames.size() / std::max(sec, 1e-6);
        avgFps = avgFps == 0.0 ? batchFps : 0.9 * avgFps + 0.1 * batchFps;

        if (std::chrono::duration<double>(Clock::now() - lastReport).count() > 1.0) {
            lastReport = Clock::now();
            if (total > 0)
                std::printf("\r  %lld/%lld frames  %5.1f fps  ETA %s   ", done, total, avgFps,
                            formatEta((total - done) / std::max(avgFps, 1e-3)).c_str());
            else
                std::printf("\r  %lld frames  %5.1f fps   ", done, avgFps);
            std::fflush(stdout);
        }
    }

    inQueue.abort();
    reader.join();
    outQueue.close();
    writerThread.join();
    writer.release();

    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    std::printf("\nProcessed %lld frames in %.1f s (%.1f fps average)\n", done, elapsed, done / std::max(elapsed, 1e-6));
    if (!outPath.empty()) std::cout << "Saved video: " << outPath << '\n';
    return 0;
}

}  // namespace app
