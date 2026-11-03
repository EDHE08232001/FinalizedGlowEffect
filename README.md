# Glow Effect

Make any object in a photo, video or live camera feed **glow**: a TensorRT segmentation network finds the
object (person, car, dog, ...), and CUDA turns its mask into a soft, coloured halo using a GPU mipmap blur.
Tweak everything live in the **Glow Studio** preview, or run it headless to batch-process files.

```
 image / folder / video / camera
            │
            ▼
 ┌──────────────────────┐   resize to network size, BGR→RGB, ImageNet normalise (CUDA)
 │  Segmenter (TensorRT)│   inference → per-pixel argmax (CUDA)  ⇒  class map
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   mask of the chosen class → mip pyramid (2×2 box filter per level)
 │  GlowRenderer (CUDA) │   trilinear sample at LOD = size/10 → soft halo → blend / add colour
 └──────────┬───────────┘
            ▼
   Glow Studio window  and/or  ./output/*.png | *.mp4
```

**Developers:** Edward He (edhetech01@gmail.com), Leo Wei (kmziran2019@gmail.com)

---

## Contents

1. [Requirements](#1-requirements)
2. [Quick start](#2-quick-start)
3. [Installing the dependencies](#3-installing-the-dependencies)
4. [Building](#4-building)
5. [Running](#5-running)
6. [Glow Studio (the UI)](#6-glow-studio-the-ui)
7. [How it works](#7-how-it-works)
8. [Project layout](#8-project-layout)
9. [Using your own model](#9-using-your-own-model)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Requirements

| | Version | Notes |
|---|---|---|
| NVIDIA GPU | Turing (RTX 20xx) or newer recommended | any GPU supported by your TensorRT version |
| OS | Windows 10/11 or Linux (incl. WSL2, Jetson) | macOS cannot run it (no CUDA) |
| CUDA Toolkit | 11.8 or newer | must match what your TensorRT build expects |
| TensorRT | 8.5 or newer (8.x, 10.x, 11.x) | provides `nvinfer` + `nvonnxparser` |
| OpenCV | 4.2 or newer | **stock build is fine** – CUDA modules are *not* needed |
| CMake | 3.18+ (3.21+ on Windows, 3.24+ for automatic GPU arch) | |
| C++ compiler | MSVC 2019/2022, GCC 9+, or Clang 10+ | C++17 |
| Python (optional) | 3.9+ with `torch`, `torchvision`, `onnx` | only to export the pretrained model |

LibTorch, wxWidgets and Visual Studio project files are **no longer required**.

## 2. Quick start

```bash
# 1. get a model (one time) – a pretrained 21-class Pascal VOC segmentation network
pip install torch torchvision onnx
python scripts/export_onnx.py                 # writes models/deeplabv3_mobilenet.onnx

# 2. build
./scripts/build.sh                            # Linux
.\scripts\build.ps1 -TensorRTRoot C:\TensorRT-10.4.0.26 -OpenCVDir C:\opencv\build   # Windows

# 3. play
./build/bin/glow_effect path/to/photo.jpg --class person
```

The first run compiles the ONNX model into a TensorRT engine for your GPU (a minute or two) and caches it as
`models/deeplabv3_mobilenet.b4.fp16.plan`; later runs start instantly.

## 3. Installing the dependencies

### Linux (Ubuntu / Debian)

```bash
sudo apt install build-essential cmake ninja-build libopencv-dev
```

* **CUDA Toolkit** – install from <https://developer.nvidia.com/cuda-downloads> (adds `/usr/local/cuda`).
* **TensorRT** – either the NVIDIA apt repo (`sudo apt install tensorrt-dev`, found automatically) or the
  tar package from <https://developer.nvidia.com/tensorrt>, extracted somewhere like `/opt/TensorRT-10.x`.
  For the tar package, tell the build where it is and let the loader find the libraries:

  ```bash
  export TENSORRT_ROOT=/opt/TensorRT-10.x
  export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH   # add both lines to ~/.bashrc
  ```

### Windows

1. **Visual Studio 2022 Build Tools** (or full VS) with the *Desktop development with C++* workload.
2. **CUDA Toolkit** – install *after* Visual Studio so the CUDA build integration is added.
3. **TensorRT** – download the Windows zip, extract e.g. to `C:\TensorRT-10.4.0.26`, and add its `lib`
   folder to `PATH` (TensorRT 8.x also needs cuDNN's `bin` on `PATH`).
4. **OpenCV** – run the official Windows installer from <https://opencv.org/releases/>, which extracts to e.g.
   `C:\opencv`. The CMake package lives in `C:\opencv\build`.
5. **CMake** – <https://cmake.org/download/> (tick "add to PATH").

The build copies the OpenCV and TensorRT DLLs next to `glow_effect.exe`, so it runs without further setup.

## 4. Building

### Option A – helper scripts

```bash
./scripts/build.sh                                  # Linux; honours TENSORRT_ROOT, CUDA_ARCH, BUILD_TYPE
CUDA_ARCH=86 ./scripts/build.sh                     # choose the GPU architecture yourself
```

```powershell
.\scripts\build.ps1 -TensorRTRoot C:\TensorRT-10.4.0.26 -OpenCVDir C:\opencv\build
.\scripts\build.ps1 -CudaArch 89 -Generator Ninja   # Ninja from an "x64 Native Tools" prompt
```

### Option B – plain CMake (works everywhere)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DTensorRT_ROOT=/opt/TensorRT-10.x \
      -DOpenCV_DIR=/path/to/opencv/build          # only if OpenCV is not found automatically
cmake --build build --config Release --parallel
```

Or with presets: `cmake --preset release && cmake --build --preset release`.

| CMake option | Default | Meaning |
|---|---|---|
| `TensorRT_ROOT` | `$TENSORRT_ROOT` | folder with TensorRT `include/` and `lib/` |
| `OpenCV_DIR` | auto | folder containing `OpenCVConfig.cmake` |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (CMake ≥ 3.24) else `75;86` | `75` RTX 20xx · `86` RTX 30xx · `89` RTX 40xx · `120` RTX 50xx · `87` Jetson Orin |
| `GLOW_COPY_RUNTIME_DLLS` | `ON` | Windows: copy dependency DLLs next to the exe |

The executable ends up in `build/bin/glow_effect` (`build\bin\Release\glow_effect.exe` with Visual Studio
generators). `cmake --install build --prefix dist` produces a clean copy.

### Option C – Visual Studio, if you still like it

*File → Open → Folder…* on the repo root (VS reads `CMakeLists.txt` / `CMakePresets.json` directly), or
generate a classic solution with `cmake --preset vs2022` and open `build/GlowEffect.sln`.

## 5. Running

```
glow_effect [options] <input>
```

`<input>` can be

| Input | Example | What happens |
|---|---|---|
| an image | `photo.jpg` | opens Glow Studio; `s` saves to `output/photo_glow.png` |
| a folder | `holiday/` | browse every image with `n` / `b` |
| a video | `street.mp4` | live preview while writing `output/street_glow.mp4` |
| a camera | `camera` or `camera:1` | live webcam glow (add `-o file.mp4` to record) |
| nothing | | asks for a path – you can drag a file into the terminal |

### Examples

```bash
glow_effect dog.jpg --class dog                            # make the dog glow
glow_effect street.mp4 -c car --mode add --hue 190         # cyan neon cars
glow_effect camera --class person --size 50                # webcam, bigger halo
glow_effect photos/ --no-display -o output/photos          # batch-process a folder, no window
glow_effect clip.mp4 --no-display -o result.mp4            # render a video on a headless server
glow_effect --build-engine models/my.onnx models/my.plan --batch 8   # pre-build an engine
glow_effect --list-classes                                 # show the class names
```

### Options

| Option | Default | Description |
|---|---|---|
| `-e, --engine PATH` | auto | `.plan` / `.engine` / `.onnx`. Auto = `$GLOW_ENGINE`, then the first model in `./models` |
| `-o, --output PATH` | `./output` | output file (image/video) or folder |
| `-c, --class NAME\|ID` | `person` | the class that glows (`--list-classes`) |
| `-i, --intensity N` | `90` | glow strength in %, 0–200 |
| `-s, --size N` | `33` | glow radius, 0–100 (mip level = N/10) |
| `--hue N` | `300` | glow colour hue, 0–360 (300 purple, 190 cyan, 45 gold, 0 red) |
| `-m, --mode blend\|add` | `blend` | *blend* tints the object; *add* adds light (neon look) |
| `-b, --batch N` | engine's | frames per inference for dynamic-batch engines |
| `--input-size N` | `384` | network resolution when compiling a dynamic-size ONNX model |
| `--fp32` | | build ONNX engines in FP32 instead of FP16 |
| `--no-display` | | headless: no windows, just write files |
| `--no-save` | | preview a video without writing it |
| `--list-classes` | | print the class table |

Classes of the bundled (Pascal VOC) models:
`0 background · 1 aeroplane · 2 bicycle · 3 bird · 4 boat · 5 bottle · 6 bus · 7 car · 8 cat · 9 chair ·
10 cow · 11 diningtable · 12 dog · 13 horse · 14 motorbike · 15 person · 16 pottedplant · 17 sheep · 18 sofa ·
19 train · 20 tvmonitor`

## 6. Glow Studio (the UI)

Two windows open:

* **Glow Studio** – the preview with a heads-up display (current class + colour swatch, mode, intensity,
  size, hue, view, file / frame counter and FPS).
* **Glow Controls** – sliders plus a cheat-sheet card.

**The fastest way to play: click on any object in the preview – its class starts glowing.**

| Slider | Range | Effect |
|---|---|---|
| Class | 0–20 | which segmentation class glows |
| Intensity % | 0–200 | how strong the glow is |
| Glow size | 0–100 | halo radius (higher = softer and wider) |
| Hue | 0–360 | glow colour |
| Blend\|Add | 0–1 | blend (tint) or additive (light) |
| View | 0–2 | result · before/after · segmentation |

| Key | Action |
|---|---|
| left click | glow the class under the cursor |
| `,` / `.` | previous / next class |
| `-` / `+` | intensity down / up |
| `[` / `]` | glow smaller / larger |
| `c` | cycle colour presets (purple, cyan, gold, red, green, blue) |
| `m` | toggle blend / additive |
| `v` or `1` `2` `3` | result · before/after split · segmentation map |
| `n` / `b` (or Enter / Backspace) | next / previous image (folders) |
| `space` | pause / resume (videos & camera) – sliders still update the paused frame |
| `s` | save the current result to `./output` |
| `r` | reset settings (keeps the class) |
| `h` | help overlay |
| `q` / `Esc` | quit |

The **segmentation view** colours every class the network found and outlines the glowing one in white –
handy to see why something does or does not glow.

## 7. How it works

1. **Pre-processing (CUDA)** – frames are resized to the network size (384×384 by default) into pinned memory,
   uploaded, then one kernel converts BGR `uint8` HWC into RGB `float` NCHW with ImageNet mean/std.
2. **Segmentation (TensorRT)** – the engine runs with the named-tensor API (`enqueueV3`, TensorRT ≥ 8.5),
   static or dynamic batch. An `argmax` kernel turns the logits into a `uint8` class map that stays on the GPU.
3. **Mask → mip pyramid (CUDA)** – the class map is up-sampled to frame size and written straight into level 0
   of a CUDA mipmapped array (1 = target class). Each next level is a 2×2 box filter of the previous one,
   computed with one bilinear texture fetch per texel. Only the levels the chosen blur needs are rebuilt.
4. **Composite (CUDA)** – every pixel samples the pyramid trilinearly at LOD = `size / 10` (5 rotated taps for a
   smoother halo), scales it by the intensity to get α, and mixes the glow colour in:
   `blend: out = src + (colour − src)·α`, `add: out = src + colour·α`.
5. **Video pipeline** – a decoder thread, the GPU (main) thread and an encoder thread are connected by bounded
   queues, so decoding, inference and encoding overlap. Frames are segmented in batches (default: the engine's
   batch size, 1 for cameras to keep latency low).

All GPU resources (engine buffers, mip pyramid, texture/surface objects) are allocated once and reused, which is
what keeps per-frame cost low.

## 8. Project layout

```
.
├── CMakeLists.txt            # build definition (library + app)
├── CMakePresets.json         # release / debug / ninja / vs2022 presets
├── cmake/FindTensorRT.cmake  # locates TensorRT on Windows and Linux
├── include/glow/             # public headers of the glow_core library
│   ├── cuda_utils.hpp        #   error checking, RAII device/pinned buffers, streams
│   ├── glow_params.hpp       #   GlowParams, class names, colours
│   ├── segmenter.hpp         #   TensorRT engine loading/building + inference
│   ├── glow_renderer.hpp     #   GPU glow renderer
│   └── kernels.hpp           #   CUDA kernel launchers
├── src/                      # glow_core implementation
│   ├── segmenter.cpp
│   ├── glow_params.cpp
│   ├── cuda_utils.cpp
│   └── cuda/
│       ├── segmentation_kernels.cu   # preprocess, argmax
│       └── glow_renderer.cu          # mask extraction, mipmaps, composite
├── apps/glow_effect/         # the command line app
│   ├── main.cpp              #   entry point, model discovery, dispatch
│   ├── cli.cpp/.hpp          #   argument parsing
│   ├── studio_ui.cpp/.hpp    #   Glow Studio windows, HUD, keys, mouse picking
│   ├── image_mode.cpp        #   single image / folder workflow
│   ├── video_mode.cpp        #   video / camera pipeline
│   └── bounded_queue.hpp     #   producer/consumer queue
├── scripts/
│   ├── build.sh / build.ps1  # one-command builds
│   └── export_onnx.py        # export a pretrained model to models/
└── models/                   # put .onnx / .plan files here (git-ignored)
```

`glow_core` is a normal static library, so you can link it into your own tools:

```cpp
glow::Segmenter seg("models/deeplabv3_mobilenet.onnx");
glow::GlowRenderer glow(seg.stream());
seg.run({frame});
glow.setMask(seg.deviceMask(0), seg.maskWidth(), seg.maskHeight());
glow.render(frame, glow::GlowParams{/*targetClass=*/15}, out);
```

## 9. Using your own model

Any segmentation network works if it has

* **input** `float32 [N, 3, H, W]` – RGB, ImageNet-normalised (`(x/255 − mean) / std`);
* **output** `float32 [N, C, H', W']` logits (or `int32 [N, H', W']` labels).

Static or dynamic batch and any resolution are fine; `H'×W'` may differ from the input. Models with 21 outputs
get Pascal VOC class names, others show `class N`. The original MobileOne-S4 engines of this project
(`mobileone_s4.edhe.plan`, batch 4, and `mobileones4_1.edhe.plan`, batch 1) can simply be copied into `models/`
– as long as they were built on the same GPU and TensorRT version.

`scripts/export_onnx.py --arch {deeplabv3,lraspp,fcn} --size 384 [--static-batch N] [--fp16]` exports the
torchvision models. Use `--fp16` on TensorRT 11+, where precision comes from the ONNX file.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `TensorRT not found` while configuring | pass `-DTensorRT_ROOT=<folder with include/ and lib/>` or set `TENSORRT_ROOT` |
| `Could not find OpenCV` | pass `-DOpenCV_DIR=<folder containing OpenCVConfig.cmake>` (Windows: `C:\opencv\build`) |
| `No CMAKE_CUDA_COMPILER could be found` | add CUDA's `bin` to `PATH` (Linux: `/usr/local/cuda/bin`) or pass `-DCMAKE_CUDA_COMPILER=.../nvcc` |
| `nvcc fatal: Unsupported gpu architecture 'compute_native'` | CMake could not see a GPU: pass `-DCMAKE_CUDA_ARCHITECTURES=86` (your GPU's value) |
| `libnvinfer.so: cannot open shared object file` | `export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH` |
| Windows: `nvinfer*.dll was not found` | add TensorRT's `lib` folder to `PATH` (and cuDNN's `bin` for TensorRT 8) |
| `Failed to deserialize ... .plan` | the engine was built on another GPU / TensorRT version – rebuild it from the ONNX file |
| `no model found` | run `scripts/export_onnx.py`, or pass `--engine`, or set `GLOW_ENGINE` |
| Nothing glows | the class is not in the picture – switch to the segmentation view (`3`) and click the object |
| MP4 cannot be written | OpenCV has no MP4 encoder on your system; the app falls back to `.avi` (MJPG) automatically |
| Windows don't open over SSH / on a server | use `--no-display` |
