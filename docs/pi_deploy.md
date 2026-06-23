# Raspberry Pi Deployment Guide

This document describes recommended steps to deploy a YOLOv5-based model for
real-time 3D printing defect detection on a Raspberry Pi (ARM) device.

## 1. Model selection
- Start with `yolov5s` or a pruned version of your trained model to meet latency constraints.
- Reduce image size (`--img 320` or `416`) for faster inference if acceptable for accuracy.

## 2. Export to ONNX (recommended)
- Use this repository's export CLI for a consistent edge export workflow.
- Example (on a workstation with GPU):

```bash
python src/export_cli.py \
  --weights models/yolov5s.pt \
  --output-dir models/exported \
  --img 640 \
  --batch 1 \
  --include onnx \
  --quantize-onnx
```

- Verify ONNX with `onnx.checker.check_model(model)` and test inference with `onnxruntime`.

## 3. Quantize (optional, improves CPU speed)
- ONNX dynamic quantization:

```bash
pip install onnxruntime-tools
python -m onnxruntime_tools.optimizer_cli --input model.onnx --output model_quant.onnx --mode dynamic
```

- PyTorch dynamic quantization (for TorchScript models):

```python
from torch.quantization import quantize_dynamic
quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

- Static quantization requires calibration data and should be done on a workstation.

## 4. Set up Raspberry Pi
- Update and install build dependencies:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y libatlas-base-dev libjpeg-dev libopenblas-base
```

- Create a virtual environment and install runtime packages (prefer prebuilt ARM wheels):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime opencv-python-headless numpy
```

- If you need PyTorch on Pi, use a prebuilt wheel compatible with your Pi's OS and Python version.

## 5. On-device inference example (ONNX + onnxruntime)

```python
import onnxruntime as ort
import cv2
import numpy as np

sess = ort.InferenceSession('model_quant.onnx')
# Preprocess an image, run inference, postprocess outputs
```

## 6. Hardware accelerators
- Consider Coral TPU (Edge TPU) or Intel Neural Compute Stick, or Jetson Nano for higher performance.
- These often require additional conversion steps (TensorFlow Lite for Coral, TensorRT for Jetson).

## 7. Tips
- Use smaller images and batch size=1 for real-time.
- Profile CPU vs memory usage; prefer int8 quantized models for faster inference.
- Document expected latency for each model variant in `docs/`.
