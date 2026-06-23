"""Train wrapper for YOLOv5 using project `config.yaml`.

This script builds a robust subprocess call to the YOLOv5 `train.py` inside the vendor
`yolov5/` folder and provides logging, argument parsing and basic validation.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .utils import load_config, setup_logging, ensure_dir


def build_train_command(cfg: Dict[str, Any], overrides: Dict[str, Optional[str]] | None = None) -> List[str]:
    """Construct the command list to run YOLOv5 training."""
    training = cfg.get("training", {})
    paths = cfg.get("paths", {})

    img = overrides.get("img") if overrides and overrides.get("img") else training.get("img_size", 640)
    batch = overrides.get("batch") if overrides and overrides.get("batch") else training.get("batch_size", 16)
    epochs = overrides.get("epochs") if overrides and overrides.get("epochs") else training.get("epochs", 100)

    cmd = [
        "python",
        "train.py",
        "--img",
        str(img),
        "--batch",
        str(batch),
        "--epochs",
        str(epochs),
        "--data",
        training.get("data", "Dataset/dataset.yaml"),
        "--cfg",
        training.get("cfg", "models/yolov5x.yaml"),
        "--weights",
        paths.get("weights", "yolov5s.pt"),
        "--name",
        training.get("name", "yolov5_model"),
        "--exist-ok",
    ]
    return cmd


def train(config_path: Path, img: Optional[int], batch: Optional[int], epochs: Optional[int]) -> int:
    """Run YOLOv5 training via subprocess in the configured `yolov5` repo.

    Returns the subprocess return code.
    """
    cfg = load_config(config_path)
    project_root = config_path.resolve().parent
    yolov5_dir = (project_root / cfg.get("paths", {}).get("yolov5_repo", "yolov5")).resolve()

    logging.debug("Using YOLOv5 repo at %s", yolov5_dir)

    if not yolov5_dir.exists():
        logging.error("yolov5 repo not found at %s", yolov5_dir)
        return 2

    overrides = {"img": img, "batch": batch, "epochs": epochs}
    cmd = build_train_command(cfg, overrides)

    logging.info("Running training command: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, cwd=str(yolov5_dir), capture_output=True, text=True, check=False)
        logging.info("Train stdout:\n%s", result.stdout)
        if result.returncode != 0:
            logging.error("Train stderr:\n%s", result.stderr)
        else:
            logging.info("Training completed successfully.")
        return result.returncode
    except FileNotFoundError as e:
        logging.exception("Failed to execute training command: %s", e)
        return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv5 using project config.yaml")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml", help="Path to config.yaml")
    parser.add_argument("--img", type=int, default=None, help="Image size override")
    parser.add_argument("--batch", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs override")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file=args.log_file)
    rc = train(args.config, args.img, args.batch, args.epochs)
    if rc != 0:
        logging.error("Training exited with code %d", rc)
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
