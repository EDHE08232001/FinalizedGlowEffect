// User-tweakable glow parameters and the segmentation class table.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace glow {

enum class BlendMode : int {
    Blend = 0,     ///< Tint the target towards the glow colour (classic look of the original project).
    Additive = 1,  ///< Add light on top of the image (brighter "neon" bloom).
};

struct GlowParams {
    int targetClass = 15;   ///< Segmentation class that glows (15 = "person" for Pascal VOC models).
    int intensity = 90;     ///< Glow strength in percent, 0..200.
    int size = 33;          ///< Glow radius, 0..100. Internally mipmap LOD = size / 10.
    int hue = 300;          ///< Glow colour hue in degrees, 0..360 (300 = magenta/purple).
    BlendMode mode = BlendMode::Blend;

    float lod() const { return static_cast<float>(size) / 10.0f; }
    float strength() const { return static_cast<float>(intensity) / 100.0f; }
    /// Glow colour as B, G, R in 0..255 (fully saturated, full value).
    std::array<float, 3> colorBgr() const;
};

/// The 21 Pascal VOC classes (background + 20 objects) used by the reference models.
const std::vector<std::string>& vocClassNames();

/// Human readable name of a class index for a model with `numClasses` outputs.
std::string className(int index, int numClasses);

/// Parses "person" / "15" / "car" etc. Returns -1 when the string is not understood.
int parseClass(const std::string& text, int numClasses);

/// RGB-ish colour used to draw a class in the segmentation view (B, G, R).
std::array<unsigned char, 3> classColorBgr(int index);

}  // namespace glow
