# 3D Printing Defect Detection (Layer-by-Layer)

**Real-time computer vision system for detecting defects in FFF/FDM 3D printing** — built to minimize material waste and failed prints through automated quality control.

![3D Printing in Action](https://media.giphy.com/media/BINkTZl70NCv0QpdgU/giphy.gif)  
*Example of 3D printer layer-by-layer operation*

---

## 🎯 Overview

Developed a production-grade layer-by-layer defect detection system using **YOLOv5**. Identifies common defects (stringing, warping, under-extrusion, blobs, etc.) in real-time. Optimized for **edge deployment on Raspberry Pi**.

**Key Achievements**:
- **>90% detection accuracy** across defect categories
- Mean IoU **> 0.5**
- Quasi real-time inference (**under 60 seconds per layer**) on Raspberry Pi
- Significant reduction in material waste by enabling early print pausing

---

## ✨ Features
- Fine-tuned YOLOv5 with data augmentation and hard-negative mining
- Configuration-driven training & inference
- Ready for TensorFlow Lite / ONNX export
- Clean, modular, and well-documented codebase

---

## 🛠️ Tech Stack
- **Core**: Python, PyTorch, Ultralytics YOLOv5
- **CV**: OpenCV
- **Deployment**: TensorFlow Lite, Raspberry Pi
- **Config**: PyYAML

---

## 📁 Project Structure
```bash
├── src/           # Training, inference, export, utils
├── scripts/       # Helper scripts
├── config.yaml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Export CLI
Use the built-in export CLI to create TorchScript and ONNX artifacts for edge deployment.

```bash
python src/export_cli.py \
  --weights models/yolov5s.pt \
  --output-dir models/exported \
  --img 640 \
  --batch 1 \
  --include both \
  --quantize-onnx
```

The CLI supports `--include torchscript`, `--include onnx`, and `--include both`, plus optional quantization flags.
