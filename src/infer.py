"""Simple inference script for local evaluation using a YOLOv5 weights file.

This script is intentionally lightweight and uses `torch.hub` to load a custom YOLOv5 model.
For more robust edge inference (Raspberry Pi), export the model to a lightweight runtime
and use `onnxruntime` or a platform-specific runtime.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from .utils import load_config, setup_logging


def infer(weights: Path, source: str, conf: float, iou: float, device: Optional[str]) -> None:
    """Run inference on a single image or source using a YOLOv5 weights file."""
    setup_logging()
    logging.info("Loading model from %s", weights)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights))
    if device:
        model.to(device)
    logging.info("Running inference on %s", source)
    results = model(source)
    logging.info("Results:\n%s", results.pandas().xyxy[0].to_string())
    results.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick inference using YOLOv5 weights")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=str, default="0", help="Image path, video file or camera index")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.weights, args.source, args.conf, args.iou, args.device)
