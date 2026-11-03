#!/usr/bin/env bash
# Configure + build Glow Effect on Linux (or WSL / Jetson).
#
#   ./scripts/build.sh                         # Release build into ./build
#   TENSORRT_ROOT=/opt/TensorRT-10.x ./scripts/build.sh
#   CUDA_ARCH=86 ./scripts/build.sh            # pick the GPU architecture explicitly
#   ./scripts/build.sh -DOpenCV_DIR=/path/to/opencv/build   # extra CMake arguments are passed through
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

args=(-S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE")
[[ -n "${TENSORRT_ROOT:-}" ]] && args+=(-DTensorRT_ROOT="$TENSORRT_ROOT")
[[ -n "${CUDA_ARCH:-}" ]] && args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
command -v ninja >/dev/null 2>&1 && [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]] && args+=(-G Ninja)

# nvcc is often not on PATH after a fresh CUDA install.
if ! command -v nvcc >/dev/null 2>&1 && [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

echo "==> Configuring ($BUILD_TYPE) in $BUILD_DIR"
cmake "${args[@]}" "$@"

echo "==> Building"
cmake --build "$BUILD_DIR" --parallel

echo
echo "Done: $BUILD_DIR/bin/glow_effect"
echo "Try:  $BUILD_DIR/bin/glow_effect --help"
