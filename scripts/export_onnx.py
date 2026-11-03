#!/usr/bin/env python3
"""Export a ready-to-use 21-class (Pascal VOC) segmentation model to ONNX for Glow Effect.

The models come pretrained from torchvision, so no training is needed:

    pip install torch torchvision onnx
    python scripts/export_onnx.py                       # -> models/deeplabv3_mobilenet.onnx
    python scripts/export_onnx.py --arch lraspp         # faster, a bit less accurate
    python scripts/export_onnx.py --arch fcn --size 512 # slower, more accurate

glow_effect compiles the ONNX file into a TensorRT engine the first time it runs and caches the
engine next to it (models/<name>.b<batch>.fp16.plan).

Contract expected by glow_effect (any model that follows it works, e.g. your own MobileOne-S4):
    input : float32 [N, 3, H, W], RGB, ImageNet mean/std normalised
    output: float32 [N, C, H', W'] class logits  (or int32 [N, H', W'] labels)
"""

import argparse
import pathlib
import sys

ARCHS = {
    "deeplabv3": ("deeplabv3_mobilenet_v3_large", "DeepLabV3_MobileNet_V3_Large_Weights", "deeplabv3_mobilenet"),
    "lraspp": ("lraspp_mobilenet_v3_large", "LRASPP_MobileNet_V3_Large_Weights", "lraspp_mobilenet"),
    "fcn": ("fcn_resnet50", "FCN_ResNet50_Weights", "fcn_resnet50"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="deeplabv3", help="model architecture (default: deeplabv3)")
    parser.add_argument("--size", type=int, default=384, help="network input resolution (default: 384)")
    parser.add_argument("--output", type=pathlib.Path, help="output .onnx path (default: models/<arch>.onnx)")
    parser.add_argument("--static-batch", type=int, default=0,
                        help="export with a fixed batch size instead of a dynamic one")
    parser.add_argument("--fp16", action="store_true",
                        help="store FP16 weights (needed for FP16 on TensorRT 11+, which is strongly typed)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    try:
        import torch
        import torchvision.models.segmentation as seg
    except ImportError:
        print("This script needs PyTorch: pip install torch torchvision onnx", file=sys.stderr)
        return 1

    fn_name, weights_name, default_name = ARCHS[args.arch]
    weights = getattr(seg, weights_name).DEFAULT
    print(f"Loading {fn_name} ({weights}) ...")
    model = getattr(seg, fn_name)(weights=weights).eval()

    class Wrapper(torch.nn.Module):
        """Returns plain logits and keeps the ONNX interface FP32 even when the body is FP16."""

        def __init__(self, net: torch.nn.Module, half: bool):
            super().__init__()
            self.net = net.half() if half else net
            self.half = half

        def forward(self, x):
            if self.half:
                x = x.half()
            out = self.net(x)["out"]
            return out.float()

    wrapper = Wrapper(model, args.fp16).eval()

    out_path = args.output or pathlib.Path(__file__).resolve().parent.parent / "models" / f"{default_name}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch = args.static_batch or 1
    dummy = torch.randn(batch, 3, args.size, args.size)
    dynamic_axes = None if args.static_batch else {"input": {0: "batch"}, "logits": {0: "batch"}}

    export_kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=args.opset,
                         dynamic_axes=dynamic_axes, do_constant_folding=True)
    print(f"Exporting to {out_path} (input {list(dummy.shape)}, {'dynamic' if dynamic_axes else 'static'} batch)")
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, dummy, str(out_path), dynamo=False, **export_kwargs)
        except TypeError:  # PyTorch < 2.5 has no `dynamo` argument
            torch.onnx.export(wrapper, dummy, str(out_path), **export_kwargs)

    try:
        import onnx

        onnx.checker.check_model(str(out_path))
    except ImportError:
        pass
    print(f"Done. Run e.g.:  glow_effect --engine {out_path} path/to/image.jpg --class person")
    return 0


if __name__ == "__main__":
    sys.exit(main())
