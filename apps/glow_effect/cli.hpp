// Command line options for the glow_effect application.
#pragma once

#include "glow/glow_params.hpp"

#include <optional>
#include <string>

namespace app {

enum class InputKind { Image, Directory, Video, Camera };

struct Options {
    std::string input;      ///< image / folder / video path, or "camera[:N]"
    std::string engine;     ///< .plan/.engine/.onnx; empty = auto-detect
    std::string output;     ///< output file or folder; empty = ./output/...
    std::string classText;  ///< --class value, resolved once the model is loaded
    glow::GlowParams params;
    int batch = 0;          ///< 0 = engine default
    int inputSize = 384;    ///< for ONNX models with dynamic H/W
    bool fp16 = true;
    bool display = true;
    bool save = true;       ///< write processed videos to disk
    bool listClasses = false;
    std::string buildOnnx;  ///< --build-engine <onnx> [plan]
    std::string buildPlan;
};

/// Parses argv. Prints help / errors itself; returns an exit code if the program should stop.
std::optional<int> parseArgs(int argc, char** argv, Options& opt);

void printUsage(const char* exe);

/// Strips quotes / whitespace that terminals add when a file is dragged in.
std::string cleanPath(std::string s);

/// Classifies the input. Throws std::runtime_error for missing or unsupported paths.
InputKind detectInput(const std::string& input, int* cameraIndex);

bool isImageFile(const std::string& path);

}  // namespace app
