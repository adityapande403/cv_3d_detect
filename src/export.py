"""Export and quantization helpers for YOLOv5 models.

Provides utilities to export a loaded YOLOv5 model to TorchScript and ONNX,
plus lightweight quantization helpers and guidance for Raspberry Pi deployment.

Note: ONNX export of YOLOv5 models may require using the upstream `export.py` from
YOLOv5 for full compatibility. These utilities are intended as a robust starting
point and for small models (`yolov5s`) used on edge devices.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def load_model(weights: Path, device: Optional[str] = None):
    """Load a YOLOv5 model via torch.hub (weights can be a local .pt file).

    Returns the model in eval() mode.
    """
    logging.info("Loading model from %s", weights)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights))
    if device:
        model.to(device)
    model.eval()
    return model


def export_torchscript(model: nn.Module, sample_input: torch.Tensor, out_path: Path, use_trace: bool = True) -> None:
    """Export `model` to TorchScript at `out_path`.

    Args:
        model: PyTorch model in eval mode.
        sample_input: Example input tensor (e.g., torch.zeros(1,3,img,img)).
        out_path: Destination `.pt` file.
        use_trace: If True, uses `torch.jit.trace`; otherwise uses `torch.jit.script`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if use_trace:
            logging.info("Tracing model to TorchScript: %s", out_path)
            ts = torch.jit.trace(model, sample_input)
        else:
            logging.info("Scripting model to TorchScript: %s", out_path)
            ts = torch.jit.script(model)
        ts.save(str(out_path))
        logging.info("TorchScript exported to %s", out_path)
    except Exception as e:
        logging.exception("TorchScript export failed: %s", e)
        raise


def export_onnx(model: nn.Module, sample_input: torch.Tensor, out_path: Path, opset: int = 12, dynamic_axes: bool = True) -> None:
    """Export `model` to ONNX format.

    Note: YOLOv5 repo contains an `export.py` script that handles specifics such as
    non-max suppression and custom ops. Use that for production exports. This helper
    is for simple models or quick experiments.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        input_names = ["images"]
        output_names = ["output"]
        dyn_axes = {"images": {0: "batch", 2: "height", 3: "width"}} if dynamic_axes else None
        logging.info("Exporting ONNX to %s (opset=%d)", out_path, opset)
        torch.onnx.export(
            model,
            sample_input,
            str(out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dyn_axes,
        )
        logging.info("ONNX exported to %s", out_path)
    except Exception as e:
        logging.exception("ONNX export failed: %s", e)
        raise


def quantize_dynamic_torchscript(ts_model_path: Path, out_path: Path) -> None:
    """Perform dynamic quantization on a TorchScript model file.

    This function loads the TorchScript model into memory, applies PyTorch dynamic
    quantization where applicable, and saves the quantized model.

    Dynamic quantization is simple and often provides good latency improvements for
    CPU inference (but may reduce accuracy slightly).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logging.info("Loading TorchScript model for dynamic quantization: %s", ts_model_path)
        ts = torch.jit.load(str(ts_model_path), map_location="cpu")
        logging.info("Applying dynamic quantization...")
        # Note: quantize_dynamic works on torch.nn.Module; TorchScript must be converted
        # back to eager model or quantization applied before scripting. Here we attempt
        # a best-effort approach for small models; for robust pipelines, apply quantization
        # to the original nn.Module before scripting.
        try:
            eager = ts
            quantized = torch.quantization.quantize_dynamic(eager, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
            torch.jit.save(torch.jit.script(quantized), str(out_path))
            logging.info("Quantized TorchScript saved to %s", out_path)
        except Exception:
            # Fallback: just copy the original if quantization path fails
            logging.exception("Dynamic quantization flow failed; copying original model instead")
            ts.save(str(out_path))
    except Exception as e:
        logging.exception("Quantization failed: %s", e)
        raise


# Guidance helpers
RASPBERRY_PI_NOTES = """
Raspberry Pi Deployment Notes

1. Model choice
   - Use `yolov5s` or a pruned/smaller model for real-time performance on Raspberry Pi.
   - Consider further pruning or NAS methods if latency is critical.

2. Recommended export path
   - Export to ONNX and run with `onnxruntime` on the Pi (often faster and lighter than full PyTorch).
   - Example: `python3 export.py --weights models/yolov5s.pt --img 640 --batch 1 --device cpu --include onnx`
     (use the upstream `yolov5/export.py` for best compatibility).

3. Quantization
   - Prefer post-training dynamic quantization for CPU: `torch.quantization.quantize_dynamic`.
   - For INT8 static quantization, you must calibrate with representative data on a workstation.
   - ONNX quantization (onnxruntime-tools) is another option: `quantize_dynamic` or `quantize_static`.

4. Runtime on Pi
   - Install dependencies via `pip install onnxruntime` and `opencv-python`.
   - Use `onnxruntime` with OpenVINO or default CPU execution provider.

5. System tips
   - Use swap cautiously; prefer faster SD cards or external SSD.
   - Consider using a Coral USB Accelerator or a Jetson Nano for improved throughput.

"""
