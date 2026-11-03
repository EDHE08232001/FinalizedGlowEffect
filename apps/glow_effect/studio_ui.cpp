#include "studio_ui.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>

namespace app {
namespace {

const char* kMainWin = "Glow Studio";
const char* kCtrlWin = "Glow Controls";

const char* kTbClass = "Class";
const char* kTbIntensity = "Intensity %";
const char* kTbSize = "Glow size";
const char* kTbHue = "Hue";
const char* kTbMode = "Blend|Add";
const char* kTbView = "View";

constexpr int kMaxDisplayW = 1600;
constexpr int kMaxDisplayH = 860;
constexpr std::array<int, 6> kHuePresets = {300, 190, 45, 0, 120, 220};
const char* kViewNames[] = {"Result", "Before | After", "Segmentation"};

long long nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

cv::Scalar glowColor(const glow::GlowParams& p) {
    const auto c = p.colorBgr();
    return {c[0], c[1], c[2]};
}

/// Text with a dark outline so it stays readable on any background.
void putLabel(cv::Mat& img, const std::string& text, cv::Point org, double scale, cv::Scalar color, int thick = 1) {
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), thick + 2, cv::LINE_AA);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thick, cv::LINE_AA);
}

void shadeRect(cv::Mat& img, cv::Rect r, double alpha = 0.55) {
    r &= cv::Rect(0, 0, img.cols, img.rows);
    if (r.empty()) return;
    cv::Mat roi = img(r);
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(18, 18, 22));
    cv::addWeighted(dark, alpha, roi, 1.0 - alpha, 0.0, roi);
}

cv::Mat colorizeSegmentation(const cv::Mat& frame, const cv::Mat& classMap, int target) {
    cv::Mat classes;
    cv::resize(classMap, classes, frame.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat color(frame.size(), CV_8UC3);
    std::array<cv::Vec3b, 256> lut;
    for (int i = 0; i < 256; ++i) {
        const auto c = glow::classColorBgr(i);
        lut[i] = cv::Vec3b(c[0], c[1], c[2]);
    }
    for (int y = 0; y < classes.rows; ++y) {
        const uchar* src = classes.ptr<uchar>(y);
        cv::Vec3b* dst = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < classes.cols; ++x) dst[x] = lut[src[x]];
    }
    cv::Mat out;
    cv::addWeighted(frame, 0.45, color, 0.55, 0.0, out);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(classes == target, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), std::max(2, frame.cols / 400), cv::LINE_AA);
    return out;
}

}  // namespace

StudioUi::StudioUi(glow::GlowParams& params, int numClasses, bool isVideo)
    : params_(params), numClasses_(numClasses), isVideo_(isVideo) {
    // GUI_NORMAL hides the Qt toolbar/status bar (when OpenCV is built with Qt) for a cleaner studio look.
    cv::namedWindow(kMainWin, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
    cv::namedWindow(kCtrlWin, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
    cv::moveWindow(kCtrlWin, 20, 40);
    cv::moveWindow(kMainWin, 540, 40);

    cv::createTrackbar(kTbClass, kCtrlWin, nullptr, std::max(1, numClasses_ - 1), &StudioUi::onTrackbar, this);
    cv::createTrackbar(kTbIntensity, kCtrlWin, nullptr, 200, &StudioUi::onTrackbar, this);
    cv::createTrackbar(kTbSize, kCtrlWin, nullptr, 100, &StudioUi::onTrackbar, this);
    cv::createTrackbar(kTbHue, kCtrlWin, nullptr, 360, &StudioUi::onTrackbar, this);
    cv::createTrackbar(kTbMode, kCtrlWin, nullptr, 1, &StudioUi::onTrackbar, this);
    cv::createTrackbar(kTbView, kCtrlWin, nullptr, 2, &StudioUi::onTrackbar, this);
    cv::setMouseCallback(kMainWin, &StudioUi::onMouse, this);
    syncTrackbars();
    drawControlsCard();
}

StudioUi::~StudioUi() {
    try {
        cv::destroyAllWindows();
        cv::waitKey(1);
    } catch (...) {
    }
}

void StudioUi::syncTrackbars() {
    syncing_ = true;
    cv::setTrackbarPos(kTbClass, kCtrlWin, params_.targetClass);
    cv::setTrackbarPos(kTbIntensity, kCtrlWin, params_.intensity);
    cv::setTrackbarPos(kTbSize, kCtrlWin, params_.size);
    cv::setTrackbarPos(kTbHue, kCtrlWin, params_.hue);
    cv::setTrackbarPos(kTbMode, kCtrlWin, static_cast<int>(params_.mode));
    cv::setTrackbarPos(kTbView, kCtrlWin, static_cast<int>(view_));
    syncing_ = false;
}

void StudioUi::readTrackbars() {
    glow::GlowParams p = params_;
    p.targetClass = std::clamp(cv::getTrackbarPos(kTbClass, kCtrlWin), 0, numClasses_ - 1);
    p.intensity = cv::getTrackbarPos(kTbIntensity, kCtrlWin);
    p.size = cv::getTrackbarPos(kTbSize, kCtrlWin);
    p.hue = cv::getTrackbarPos(kTbHue, kCtrlWin);
    p.mode = static_cast<glow::BlendMode>(cv::getTrackbarPos(kTbMode, kCtrlWin));
    const auto v = static_cast<ViewMode>(std::clamp(cv::getTrackbarPos(kTbView, kCtrlWin), 0, 2));

    if (p.targetClass != params_.targetClass || p.intensity != params_.intensity || p.size != params_.size ||
        p.hue != params_.hue || p.mode != params_.mode) {
        params_ = p;
        paramsChanged_ = true;
        drawControlsCard();
    }
    if (v != view_) {
        view_ = v;
        viewChanged_ = true;
    }
}

void StudioUi::onTrackbar(int, void* self) {
    auto* ui = static_cast<StudioUi*>(self);
    if (!ui->syncing_) ui->readTrackbars();
}

void StudioUi::onMouse(int event, int x, int y, int, void* self) {
    if (event == cv::EVENT_LBUTTONDOWN) static_cast<StudioUi*>(self)->pickClassAt(x, y);
}

void StudioUi::pickClassAt(int x, int y) {
    if (lastClassMap_.empty() || tileSize_.width <= 0 || tileSize_.height <= 0) return;
    const int lx = x % tileSize_.width;
    if (x < 0 || y < 0 || x >= tileSize_.width * tiles_ || y >= tileSize_.height) return;
    const int mx = std::min(lastClassMap_.cols - 1, lx * lastClassMap_.cols / tileSize_.width);
    const int my = std::min(lastClassMap_.rows - 1, y * lastClassMap_.rows / tileSize_.height);
    const int cls = lastClassMap_.at<uchar>(my, mx);
    if (cls >= numClasses_) return;
    if (cls != params_.targetClass) {
        params_.targetClass = cls;
        paramsChanged_ = true;
        syncTrackbars();
        drawControlsCard();
    }
    notify("Picked: " + glow::className(cls, numClasses_));
}

void StudioUi::notify(const std::string& msg) {
    flash_ = msg;
    flashUntilMs_ = nowMs() + 2500;
    viewChanged_ = true;
}

void StudioUi::drawControlsCard() {
    cv::Mat card(250, 480, CV_8UC3, cv::Scalar(30, 28, 34));
    const cv::Scalar fg(235, 235, 235), dim(160, 160, 170), accent = glowColor(params_);

    cv::putText(card, "GLOW STUDIO", {18, 34}, cv::FONT_HERSHEY_DUPLEX, 0.85, fg, 1, cv::LINE_AA);
    cv::rectangle(card, {340, 14, 120, 26}, accent, cv::FILLED, cv::LINE_AA);

    const std::string cls = glow::className(params_.targetClass, numClasses_);
    cv::putText(card, "Target: " + cls + "  (#" + std::to_string(params_.targetClass) + ")", {18, 70},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv::LINE_AA);

    const char* lines[] = {
        "Click an object in the preview to make it glow",
        ", .   previous / next class     m   blend / add",
        "- +   intensity                 [ ]  glow size",
        "c     colour preset             v   change view",
        isVideo_ ? "space pause / resume            s   save frame" : "b n   prev / next image         s   save image",
        "r     reset                     h   help overlay",
        "q / Esc  quit",
    };
    int y = 102;
    for (const char* l : lines) {
        cv::putText(card, l, {18, y}, cv::FONT_HERSHEY_SIMPLEX, 0.45, dim, 1, cv::LINE_AA);
        y += 21;
    }
    cv::imshow(kCtrlWin, card);
}

void StudioUi::drawHud(cv::Mat& canvas, const UiStatus& st) const {
    const double s = std::clamp(canvas.cols / 1100.0, 0.45, 0.8);
    const int lh = static_cast<int>(30 * s) + 6;
    const cv::Scalar fg(240, 240, 240), dim(185, 185, 195);

    const std::string modeName = params_.mode == glow::BlendMode::Blend ? "Blend" : "Add";
    const std::string l1 = glow::className(params_.targetClass, numClasses_);
    const std::string l2 = modeName + "  |  Intensity " + std::to_string(params_.intensity) + "%  |  Size " +
                           std::to_string(params_.size) + "  |  Hue " + std::to_string(params_.hue);
    std::string l3 = std::string(kViewNames[static_cast<int>(view_)]);
    if (!st.source.empty()) l3 += "  |  " + st.source;
    if (st.fps > 0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "  |  %.1f fps", st.fps);
        l3 += buf;
    }
    if (st.paused) l3 += "  |  PAUSED";

    int base = 0;
    const int w = std::max({cv::getTextSize(l2, cv::FONT_HERSHEY_SIMPLEX, s * 0.8, 1, &base).width,
                            cv::getTextSize(l3, cv::FONT_HERSHEY_SIMPLEX, s * 0.8, 1, &base).width, 200}) + 40;
    shadeRect(canvas, {8, 8, w, lh * 3 + 14});
    cv::rectangle(canvas, {20, 18, static_cast<int>(18 * s) + 4, static_cast<int>(18 * s) + 4}, glowColor(params_),
                  cv::FILLED, cv::LINE_AA);
    putLabel(canvas, l1, {static_cast<int>(28 * s) + 34, 18 + static_cast<int>(20 * s)}, s * 1.0, fg, 2);
    putLabel(canvas, l2, {20, 18 + lh + static_cast<int>(20 * s)}, s * 0.8, dim);
    putLabel(canvas, l3, {20, 18 + 2 * lh + static_cast<int>(20 * s)}, s * 0.8, dim);

    putLabel(canvas, "H: help   click: pick object", {14, canvas.rows - 14}, s * 0.7, dim);

    if (!flash_.empty() && nowMs() < flashUntilMs_) {
        const cv::Size ts = cv::getTextSize(flash_, cv::FONT_HERSHEY_SIMPLEX, s, 2, &base);
        const cv::Point org((canvas.cols - ts.width) / 2, canvas.rows - 50);
        shadeRect(canvas, {org.x - 16, org.y - ts.height - 14, ts.width + 32, ts.height + 28}, 0.7);
        putLabel(canvas, flash_, org, s, cv::Scalar(120, 255, 160), 2);
    }
}

void StudioUi::drawHelp(cv::Mat& canvas) const {
    const char* lines[] = {
        "KEYBOARD & MOUSE",
        "",
        "left click      make the clicked object's class glow",
        ", / .           previous / next class",
        "- / +           decrease / increase intensity",
        "[ / ]           smaller / larger glow",
        "c               cycle glow colour presets",
        "m               toggle blend / additive",
        "v  or  1 2 3    result / before-after / segmentation view",
        isVideo_ ? "space           pause / resume" : "b / n           previous / next image",
        "s               save the current result",
        "r               reset all settings",
        "h               hide this help",
        "q / Esc         quit",
    };
    const double s = std::clamp(canvas.cols / 1300.0, 0.4, 0.75);
    const int lh = static_cast<int>(34 * s);
    const int boxW = static_cast<int>(820 * s), boxH = lh * static_cast<int>(std::size(lines)) + 40;
    const cv::Rect box((canvas.cols - boxW) / 2, (canvas.rows - boxH) / 2, boxW, boxH);
    shadeRect(canvas, box, 0.8);
    int y = box.y + 20 + lh;
    for (const char* l : lines) {
        putLabel(canvas, l, {box.x + 30, y}, s, cv::Scalar(240, 240, 240));
        y += lh;
    }
}

void StudioUi::show(const cv::Mat& original, const cv::Mat& result, const cv::Mat& classMap, const UiStatus& status) {
    if (result.empty()) return;
    lastClassMap_ = classMap;

    cv::Mat composed;
    switch (view_) {
        case ViewMode::Split: cv::hconcat(original, result, composed); tiles_ = 2; break;
        case ViewMode::Segmentation:
            composed = classMap.empty() ? result : colorizeSegmentation(original, classMap, params_.targetClass);
            tiles_ = 1;
            break;
        default: composed = result; tiles_ = 1; break;
    }

    const double scale = std::min({1.0, static_cast<double>(kMaxDisplayW) / composed.cols,
                                   static_cast<double>(kMaxDisplayH) / composed.rows});
    cv::Mat canvas;
    if (scale < 1.0) cv::resize(composed, canvas, cv::Size(), scale, scale, cv::INTER_AREA);
    else canvas = composed.clone();
    tileSize_ = cv::Size(canvas.cols / tiles_, canvas.rows);

    if (tiles_ == 2) {
        const int mid = canvas.cols / 2;
        cv::line(canvas, {mid, 0}, {mid, canvas.rows}, cv::Scalar(255, 255, 255), 2);
        putLabel(canvas, "BEFORE", {mid - 110, canvas.rows - 14}, 0.6, cv::Scalar(255, 255, 255));
        putLabel(canvas, "AFTER", {mid + 14, canvas.rows - 14}, 0.6, cv::Scalar(255, 255, 255));
    }
    drawHud(canvas, status);
    if (showHelp_) drawHelp(canvas);
    cv::imshow(kMainWin, canvas);
}

UiAction StudioUi::poll(int delayMs) {
    const int key = cv::waitKey(std::max(1, delayMs));
    if (!flash_.empty() && nowMs() >= flashUntilMs_) {
        flash_.clear();
        viewChanged_ = true;
    }
    if (key < 0) return windowClosed() ? UiAction::Quit : UiAction::None;

    bool changed = false;
    switch (key & 0xFF) {
        case 'q': case 'Q': case 27: return UiAction::Quit;
        case 's': case 'S': return UiAction::Save;
        case ' ': case 'p': case 'P': return UiAction::TogglePause;
        case 'n': case 'N': case 13: return UiAction::Next;
        case 'b': case 'B': case 8: return UiAction::Prev;
        case 'h': case 'H': showHelp_ = !showHelp_; viewChanged_ = true; return UiAction::None;
        case 'v': case 'V':
            view_ = static_cast<ViewMode>((static_cast<int>(view_) + 1) % 3);
            notify(std::string("View: ") + kViewNames[static_cast<int>(view_)]);
            break;
        case '1': case '2': case '3':
            view_ = static_cast<ViewMode>((key & 0xFF) - '1');
            notify(std::string("View: ") + kViewNames[static_cast<int>(view_)]);
            break;
        case '.': case '>':
            params_.targetClass = (params_.targetClass + 1) % numClasses_; changed = true; break;
        case ',': case '<':
            params_.targetClass = (params_.targetClass + numClasses_ - 1) % numClasses_; changed = true; break;
        case '+': case '=': params_.intensity = std::min(200, params_.intensity + 10); changed = true; break;
        case '-': case '_': params_.intensity = std::max(0, params_.intensity - 10); changed = true; break;
        case ']': case '}': params_.size = std::min(100, params_.size + 5); changed = true; break;
        case '[': case '{': params_.size = std::max(0, params_.size - 5); changed = true; break;
        case 'm': case 'M':
            params_.mode = params_.mode == glow::BlendMode::Blend ? glow::BlendMode::Additive : glow::BlendMode::Blend;
            notify(params_.mode == glow::BlendMode::Blend ? "Mode: Blend" : "Mode: Additive");
            changed = true;
            break;
        case 'c': case 'C':
            colorPreset_ = (colorPreset_ + 1) % static_cast<int>(kHuePresets.size());
            params_.hue = kHuePresets[colorPreset_];
            changed = true;
            break;
        case 'r': case 'R': {
            const int keepClass = params_.targetClass;
            params_ = glow::GlowParams{};
            params_.targetClass = keepClass;
            view_ = ViewMode::Result;
            notify("Settings reset");
            changed = true;
            break;
        }
        default: return UiAction::None;
    }
    if (changed) {
        paramsChanged_ = true;
        drawControlsCard();
    }
    viewChanged_ = true;
    syncTrackbars();
    return UiAction::None;
}

bool StudioUi::consumeParamsChanged() {
    const bool c = paramsChanged_;
    paramsChanged_ = false;
    return c;
}

bool StudioUi::consumeViewChanged() {
    const bool c = viewChanged_ || paramsChanged_;
    viewChanged_ = false;
    return c;
}

bool StudioUi::windowClosed() const {
    try {
        return cv::getWindowProperty(kMainWin, cv::WND_PROP_VISIBLE) < 1.0;
    } catch (...) {
        return true;
    }
}

}  // namespace app
