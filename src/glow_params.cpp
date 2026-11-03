#include "glow/glow_params.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace glow {

std::array<float, 3> GlowParams::colorBgr() const {
    // HSV -> RGB with S = V = 1.
    const float h = std::fmod(std::max(0, hue), 360) / 60.0f;
    const float x = 1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f);
    float r = 0, g = 0, b = 0;
    switch (static_cast<int>(h)) {
        case 0: r = 1; g = x; break;
        case 1: r = x; g = 1; break;
        case 2: g = 1; b = x; break;
        case 3: g = x; b = 1; break;
        case 4: r = x; b = 1; break;
        default: r = 1; b = x; break;
    }
    return {b * 255.0f, g * 255.0f, r * 255.0f};
}

const std::vector<std::string>& vocClassNames() {
    static const std::vector<std::string> names = {
        "background", "aeroplane", "bicycle", "bird",  "boat",        "bottle", "bus",
        "car",        "cat",       "chair",   "cow",   "diningtable", "dog",    "horse",
        "motorbike",  "person",    "pottedplant", "sheep", "sofa",    "train",  "tvmonitor"};
    return names;
}

std::string className(int index, int numClasses) {
    const auto& voc = vocClassNames();
    if (numClasses == static_cast<int>(voc.size()) && index >= 0 && index < numClasses) return voc[index];
    return "class " + std::to_string(index);
}

int parseClass(const std::string& text, int numClasses) {
    if (text.empty()) return -1;
    if (std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c); })) {
        const int v = std::stoi(text);
        return (v >= 0 && v < numClasses) ? v : -1;
    }
    std::string lower(text);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    const auto& voc = vocClassNames();
    if (numClasses == static_cast<int>(voc.size())) {
        for (int i = 0; i < numClasses; ++i)
            if (voc[i] == lower) return i;
        // A few friendly aliases.
        if (lower == "people" || lower == "human") return 15;
        if (lower == "plane" || lower == "airplane") return 1;
        if (lower == "bike") return 2;
        if (lower == "motorcycle") return 14;
        if (lower == "tv" || lower == "monitor") return 20;
        if (lower == "table") return 11;
        if (lower == "plant") return 16;
    }
    return -1;
}

std::array<unsigned char, 3> classColorBgr(int index) {
    // Standard Pascal VOC palette (bit-interleaved), returned as BGR.
    unsigned char r = 0, g = 0, b = 0;
    int c = index;
    for (int j = 0; j < 8; ++j) {
        r |= static_cast<unsigned char>(((c >> 0) & 1) << (7 - j));
        g |= static_cast<unsigned char>(((c >> 1) & 1) << (7 - j));
        b |= static_cast<unsigned char>(((c >> 2) & 1) << (7 - j));
        c >>= 3;
    }
    return {b, g, r};
}

}  // namespace glow
