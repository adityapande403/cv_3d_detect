"""Deployment helpers for running inference outside of notebooks.

Provides a simple CLI to run inference on an image, video file, or camera index using
YOLOv5 weights. Uses `torch.hub` for loading a custom weights file.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from PIL import Image
import torch

from .utils import setup_logging


def load_model(weights: Path, device: Optional[str] = None):
    """Load a YOLOv5 model from a weights file via `torch.hub`.

    Args:
        weights: Path to a `.pt` file or a model name that `torch.hub` understands.
        device: Optional device string like `cpu` or `cuda`.

    Returns:
        A loaded model instance.
    """
    logging.info("Loading model from %s", weights)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights))
    if device:
        model.to(device)
    model.eval()
    return model


def run_inference(model, source: str, conf: float = 0.5, iou: float = 0.45):
    """Run inference using the model on a source and return the results object.

    `source` may be a path to an image, video, or a camera index as a string like '0'.
    """
    logging.info("Running inference on source=%s (conf=%s, iou=%s)", source, conf, iou)
    results = model(source, conf=conf, iou=iou)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv5 inference on an image/video/camera")
    parser.add_argument("--weights", type=Path, required=True, help="Path to weights (.pt)")
    parser.add_argument("--source", type=str, default="0", help="Image/video path or camera index")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None, help="Device: cpu or cuda")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file=args.log_file)
    model = load_model(args.weights, device=args.device)
    results = run_inference(model, args.source, conf=args.conf, iou=args.iou)
    try:
        df = results.pandas().xyxy[0]
        logging.info("Detections:\n%s", df.to_string())
    except Exception:
        logging.debug("Unable to convert results to pandas DataFrame", exc_info=True)
    results.show()


if __name__ == "__main__":
    main()
