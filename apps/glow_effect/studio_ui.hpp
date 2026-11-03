// "Glow Studio": the interactive preview window + control panel (OpenCV HighGUI, main thread only).
#pragma once

#include "glow/glow_params.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace app {

enum class ViewMode : int { Result = 0, Split = 1, Segmentation = 2 };

enum class UiAction { None, Quit, Next, Prev, Save, TogglePause };

/// What the HUD shows besides the parameters.
struct UiStatus {
    std::string source;  ///< e.g. "dog.jpg (2/7)" or "frame 120/900"
    double fps = 0.0;    ///< 0 = hidden
    bool paused = false;
};

class StudioUi {
public:
    StudioUi(glow::GlowParams& params, int numClasses, bool isVideo);
    ~StudioUi();

    /// Draws the current frame. `classMap` (CV_8UC1, model resolution) enables click-to-pick and the
    /// segmentation view; `original` and `result` must have the same size.
    void show(const cv::Mat& original, const cv::Mat& result, const cv::Mat& classMap, const UiStatus& status);

    /// Pumps the GUI event loop for `delayMs` and translates keys into actions / parameter edits.
    UiAction poll(int delayMs);

    /// True once after the glow parameters changed (slider, key or click).
    bool consumeParamsChanged();
    /// True once after something that needs a redraw but not a re-render changed (view, help).
    bool consumeViewChanged();

    /// Shows a short message at the bottom of the preview.
    void notify(const std::string& msg);

    ViewMode view() const { return view_; }
    bool windowClosed() const;

private:
    static void onTrackbar(int, void* self);
    static void onMouse(int event, int x, int y, int flags, void* self);

    void syncTrackbars();
    void readTrackbars();
    void drawControlsCard();
    void drawHud(cv::Mat& canvas, const UiStatus& status) const;
    void drawHelp(cv::Mat& canvas) const;
    void pickClassAt(int x, int y);

    glow::GlowParams& params_;
    int numClasses_;
    bool isVideo_;
    ViewMode view_ = ViewMode::Result;
    bool showHelp_ = false;
    bool paramsChanged_ = true;
    bool viewChanged_ = true;
    bool syncing_ = false;
    int colorPreset_ = 0;

    // Layout of the last shown canvas, for mapping mouse clicks back to the class map.
    cv::Mat lastClassMap_;
    cv::Size tileSize_;
    int tiles_ = 1;

    std::string flash_;
    long long flashUntilMs_ = 0;
};

}  // namespace app
