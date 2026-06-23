# 3D Printing Defect Detection (Layer-by-Layer)

**Overview**

This repository provides a production-ready pipeline for real-time layer-by-layer defect detection on FFF/FDM 3D printers using YOLOv5. The project is designed for edge deployment (example target: Raspberry Pi) and emphasizes clean structure, reproducibility, and easy evaluation.

**Features**

- Train and evaluate YOLOv5 models for 3D printing defect detection
- Real-time inference pipeline suitable for edge devices
- Export and optimization guidance for Raspberry Pi (ONNX / TorchScript / quantization)
- Configuration-driven setup using `config.yaml`
- Clear README, requirements, and gitignore for recruiter-friendly presentation

**Tech Stack**

- PyTorch / Ultralytics YOLOv5
- OpenCV for camera and image processing
- PyYAML for configuration
- NumPy, Pandas, Matplotlib for data handling and visualization

**Project Structure (proposed)**

- `src/` — project source (training, inference, utils)
- `yolov5/` — upstream YOLOv5 code (vendor copy or submodule)
- `data/` or `Dataset/` — datasets and annotations
- `models/` — trained model artifacts and exported formats
- `configs/` — configuration files (e.g., `config.yaml`)
- `scripts/` — helper scripts (export, quantize)
- `docs/` — additional documentation and results
- `README.md`, `requirements.txt`, `.gitignore`

**Quick Start**

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Edit `config.yaml` to point to your dataset and set `device` (e.g., `cpu` or `cuda`).

4. Train (example using the provided YOLOv5 training script in `src/train.py` once created):

```bash
python src/train.py --config config.yaml
```

5. Run inference (once `src/infer.py` is implemented):

```bash
python src/infer.py --weights models/yolov5s.pt --source 0
```

**Raspberry Pi Deployment (summary)**

- Use `yolov5s` or a pruned/quantized model for real-time performance on Pi.
- Export to ONNX or TorchScript, then convert to TensorRT / tflite / onnxruntime if desired.
- Install system deps: `libjpeg-dev`, `libpng-dev` and use Python wheels compatible with ARM for PyTorch / torchvision or use `onnxruntime` for inference.

**Results & Evaluation**

Include model metrics, example inference images, and short videos/GIFs in `docs/` to showcase accuracy and latency.

**Contributing**

Contributions are welcome. Please follow these steps:

- Fork the repo
- Create a feature branch
- Open a PR with a clear description and tests where appropriate

**License**

Specify your chosen license here (e.g., MIT).
