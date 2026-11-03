# models/

Put your segmentation model here. `glow_effect` automatically uses the first engine
(`.plan` / `.engine` / `.trt`) it finds in this folder, otherwise the first `.onnx` file.
You can always choose explicitly with `--engine <file>` or the `GLOW_ENGINE` environment variable.

## Option A - export a pretrained model (recommended)

```bash
pip install torch torchvision onnx
python scripts/export_onnx.py            # -> models/deeplabv3_mobilenet.onnx
```

The first run of `glow_effect` compiles the ONNX file into a TensorRT engine for *your* GPU
and caches it here as `deeplabv3_mobilenet.b4.fp16.plan` (takes a minute or two, once).

## Option B - bring your own

Any model with this interface works (for example the original MobileOne-S4 engines,
`mobileone_s4.edhe.plan` / `mobileones4_1.edhe.plan`):

| tensor | type / shape | notes |
|--------|--------------|-------|
| input  | float32 `[N, 3, H, W]` | RGB, `(x/255 - mean) / std` with ImageNet mean `0.485 0.456 0.406`, std `0.229 0.224 0.225` |
| output | float32 `[N, C, H', W']` logits, or int32 `[N, H', W']` labels | 21 classes = Pascal VOC names in the UI |

TensorRT engines only run on the GPU model + TensorRT version that built them. If you copy a
`.plan` from another machine and it fails to load, rebuild it from the ONNX file:

```bash
glow_effect --build-engine models/my_model.onnx models/my_model.plan --batch 4
```
