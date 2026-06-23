"""CLI for exporting and quantizing YOLOv5 models for edge deployment."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .export import (
    export_onnx,
    export_torchscript,
    load_model,
    quantize_dynamic_torchscript,
    quantize_onnx_dynamic,
)
from .utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and quantize a YOLOv5 model for Raspberry Pi and edge deployment"
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained YOLOv5 .pt weights")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Directory to write exported models",
    )
    parser.add_argument("--img", type=int, default=640, help="Export image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for export")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the model on")
    parser.add_argument(
        "--include",
        choices=["torchscript", "onnx", "both"],
        default="both",
        help="Which export formats to produce",
    )
    parser.add_argument("--no-trace", action="store_false", dest="use_trace", help="Use scripting instead of tracing for TorchScript export")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--no-dynamic-axes", action="store_false", dest="dynamic_axes", help="Disable dynamic axes in ONNX export")
    parser.add_argument("--quantize-onnx", action="store_true", help="Apply ONNX dynamic quantization after export")
    parser.add_argument("--quantize-ts", action="store_true", help="Apply dynamic quantization to the TorchScript model after export")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    return parser.parse_args()


def make_sample_input(batch: int, image_size: int, device: str) -> torch.Tensor:
    return torch.zeros(batch, 3, image_size, image_size, device=device)


def main() -> None:
    args = parse_args()
    setup_logging(log_file=args.log_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Exporting model %s to %s", args.weights, args.output_dir)
    model = load_model(args.weights, args.device)
    sample_input = make_sample_input(args.batch, args.img, args.device)
    stem = args.weights.stem

    if args.include in {"torchscript", "both"}:
        ts_path = args.output_dir / f"{stem}.torchscript.pt"
        export_torchscript(model, sample_input, ts_path, use_trace=args.use_trace)
        if args.quantize_ts:
            quantized_ts_path = args.output_dir / f"{stem}.torchscript.quant.pt"
            quantize_dynamic_torchscript(ts_path, quantized_ts_path)

    if args.include in {"onnx", "both"}:
        onnx_path = args.output_dir / f"{stem}.onnx"
        export_onnx(
            model,
            sample_input,
            onnx_path,
            opset=args.opset,
            dynamic_axes=args.dynamic_axes,
        )
        if args.quantize_onnx:
            quantized_onnx_path = args.output_dir / f"{stem}.quant.onnx"
            quantize_onnx_dynamic(onnx_path, quantized_onnx_path)

    logging.info("Export CLI finished")


if __name__ == "__main__":
    main()
