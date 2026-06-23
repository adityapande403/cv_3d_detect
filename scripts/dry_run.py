"""Dry-run harness to validate config and constructed training command without executing.

Usage:
    python scripts/dry_run.py --config ../config.yaml

Pass `--execute` to actually run the training subprocess (NOT recommended unless you want to start training).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional

from src.utils import load_config, setup_logging
from src.train import build_train_command, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run the YOLOv5 training command from project config")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml")
    parser.add_argument("--img", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="If set, actually execute the training command")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file=args.log_file)
    cfg = load_config(args.config)
    cmd = build_train_command(cfg, {"img": args.img, "batch": args.batch, "epochs": args.epochs})
    logging.info("Constructed command: %s", " ".join(cmd))

    if args.execute:
        logging.warning("--execute specified: running training command (this may start a long job).")
        rc = train(args.config, args.img, args.batch, args.epochs)
        logging.info("Training subprocess exit code: %s", rc)
    else:
        logging.info("Dry-run complete. No subprocess executed.")


if __name__ == "__main__":
    main()
