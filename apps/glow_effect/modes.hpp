// The two workflows of the app: still images (single file or folder) and video streams (file or camera).
#pragma once

#include "cli.hpp"

#include <string>
#include <vector>

namespace glow {
class Segmenter;
}

namespace app {

/// Interactive (or headless batch) processing of still images.
int runImages(const std::vector<std::string>& images, glow::Segmenter& seg, Options& opt);

/// Video file or camera: decode thread -> GPU (segment + glow) -> encode thread, with live preview.
int runVideo(const std::string& source, int cameraIndex, glow::Segmenter& seg, Options& opt);

/// Creates parent folders and returns `path`.
std::string prepareOutputPath(const std::string& path);

}  // namespace app
