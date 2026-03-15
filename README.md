# Glow Effect

A CUDA-accelerated glow/bloom effect application for images and video. Uses TensorRT for real-time semantic segmentation to identify target regions, then applies GPU-accelerated mipmapping and alpha blending to create cinematic glow effects.

**Developers:** Edward He (edhetech01@gmail.com), Leo Wei (kmziran2019@gmail.com)

## How It Works

The pipeline has three stages:

### 1. Segmentation (TensorRT)
Input frames are resized to 384x384, normalized, and fed into a TensorRT-optimized MobileOneS4 segmentation model. The model outputs per-pixel class labels (21 classes). A custom CUDA `argmaxKernel` finds the dominant class per pixel and maps it to a grayscale value. The segmentation mask identifies which regions of the image receive the glow effect, controlled by the `Key Level` parameter.

### 2. Mipmap Glow (CUDA)
The segmentation mask is converted to an RGBA buffer where only pixels matching the target class are opaque. This buffer is uploaded into a CUDA mipmapped array, and successive mipmap levels are generated via a `d_gen_mipmap` kernel (2x2 box filter downsampling). The final blurred image is sampled using `tex2DLod` at a uniform LOD computed from the `Default Scale` parameter, producing a smooth glow halo around the target region.

**Buffering strategies for video:**
- **Triple Buffering** (used in `glow_effect_video` and `glow_effect_video_graph`): 3 pinned host buffers with non-blocking CUDA streams allow overlap of mask conversion, GPU mipmap filtering, and result readback.
- **Ping-Pong Buffering** (used in `glow_effect_video_single_batch_parallel`): 2 pinned host buffers alternate between filling and draining, using `cudaEventSynchronize` to ensure the buffer is free before reuse. Lower memory footprint than triple buffering.

### 3. Compositing
The original frame, a highlight overlay (from `glow_blow`), and the mipmap blur result are blended per-pixel using alpha blending controlled by the `Key Scale` parameter:
```
output[k] = (src[k] * (255 - alpha) + highlight[k] * alpha) >> 8
```

### Video Processing Modes

| Mode | Function | Segmentation | Buffering |
|------|----------|-------------|-----------|
| Standard | `glow_effect_video` | 2x concurrent sub-batches of 4 frames via `std::async` | Triple |
| CUDA Graph | `glow_effect_video_graph` | Same as Standard, but with CUDA Graph capture for reduced kernel launch overhead | Triple |
| Single-Batch Parallel | `glow_effect_video_single_batch_parallel` | Pre-loaded engine, 2 parallel streams per batch | Ping-Pong |

## Project Structure

```
glow_effect/
├── all_main.cpp                 # Entry point: CLI menu, GUI thread launch
├── all_common.h                 # Common headers and global variable externs
├── glow_effect.sln              # Visual Studio solution
├── glow_effect.vcxproj          # Visual Studio project
├── include/
│   └── old_movies.cuh           # CUDA type helpers (uchar4/float4 conversions, Image struct)
├── source/
│   ├── glow_effect.cpp/.hpp     # Core pipeline: glow_blow, apply_mipmap, mix_images, video functions
│   ├── TRTInference.cpp/.hpp    # TensorRT inference: batched, concurrent, CUDA Graph variants
│   ├── TRTGeneration.cpp/.hpp   # TensorRT engine builder from ONNX models
│   ├── ImageProcessingUtil.cpp/.hpp  # Image-to-tensor conversion (CPU & GPU Mat)
│   ├── wx_gui.cpp/.h            # wxWidgets GUI: sliders and buttons
│   ├── control_gui.cpp          # GUI launcher, global parameter definitions
│   ├── all_main.h               # Callback declarations
│   ├── mipmap.h                 # Mipmap filter function declarations
│   └── segmentation_kernels.h   # CUDA argmax/colormap kernel declarations
└── source_cu/
    ├── mipmap.cu                # CUDA mipmap generation, sampling, sync & async variants
    └── segmentation_kernels.cu  # CUDA argmax and color mapping kernels
```

## Dependencies

| Library | Purpose |
|---------|---------|
| **CUDA Toolkit** | GPU kernels, streams, events, mipmapped arrays |
| **TensorRT** | Optimized inference for segmentation model |
| **OpenCV** (with CUDA modules) | Image/video I/O, GPU resize, display |
| **LibTorch (PyTorch C++)** | Tensor operations for preprocessing |
| **wxWidgets** | GUI panel with sliders |
| **NVIDIA NVTX** | Performance profiling markers |

## How to Build and Run

### Prerequisites
- Windows with NVIDIA GPU (Compute Capability 6.0+)
- Visual Studio 2019 or later
- CUDA Toolkit (11.x or later)
- TensorRT (8.x or later)
- OpenCV 4.5+ built with CUDA support
- LibTorch (matching your CUDA version)
- wxWidgets 3.x

### Build Steps

1. Open `glow_effect/glow_effect.sln` in Visual Studio.
2. Configure project properties:
   - **VC++ Directories > Include Directories**: Add paths to CUDA, TensorRT, OpenCV, LibTorch, and wxWidgets headers.
   - **VC++ Directories > Library Directories**: Add paths to the corresponding lib folders.
   - **Linker > Input > Additional Dependencies**: Add `cudart.lib`, `nvinfer.lib`, `nvonnxparser.lib`, `opencv_world455.lib` (or your version), `torch.lib`, `c10.lib`, `torch_cuda.lib`, and wxWidgets libs.
   - **CUDA C/C++ > Device > Code Generation**: Set to your GPU architecture (e.g., `compute_86,sm_86` for RTX 30-series).
3. Set build configuration to **Release x64**.
4. Build the solution (Ctrl+Shift+B).

### Run

1. Ensure a TensorRT `.plan` file exists for the segmentation model. Default paths in the code:
   - Multi-batch: `D:/csi4900/TRT-Plans/mobileone_s4.edhe.plan`
   - Single-batch: `D:/csi4900/TRT-Plans/mobileones4_1.edhe.plan`

   Update these paths in `all_main.cpp` to match your setup.

2. Run the executable. You will be prompted:
   ```
   Do you want to input a single image, an image directory, or a video file? (single/directory/video):
   ```

3. **Single image mode**: Enter the image path. The segmentation runs, and a window shows the glow effect. Adjust parameters using the GUI sliders. Press `q` to quit.

4. **Video mode**: Choose single-batch or multi-batch plan, provide the video path, and optionally enable CUDA Graph acceleration. The processed video is saved to `./VideoOutput/`.

### GUI Controls

The wxWidgets panel launches in a separate thread with three sliders:

| Slider | Range | Default | Effect |
|--------|-------|---------|--------|
| Key Level | 0-255 | 96 | Selects which segmentation class value to highlight |
| Key Scale | 0-1000 | 600 | Controls glow blending intensity |
| Default Scale | 0-100 | 10 | Controls mipmap blur radius (LOD) |

### Keyboard Controls (during image/video display)

- `q` - Quit the application
- `+` - Increase display delay by 30ms (max 300ms)
- `-` - Decrease display delay by 30ms (min 30ms)
- `p` - Pause display
- `Enter` - Next image (in directory mode)
