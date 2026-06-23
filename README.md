# 3D Printing Defect Detection (Layer-by-Layer)

**Real-time computer vision system for detecting defects in FFF/FDM 3D printing** — built to minimize material waste and failed prints through automated quality control.

![3D Printing in Action](https://media.giphy.com/media/BINkTZl70NCv0QpdgU/giphy.gif)  
*Time-lapse of a 3D printer in operation (replace with your own defect detection demo later)*

---

## 🎯 Overview

Developed a production-grade layer-by-layer defect detection system using **YOLOv5**. The model identifies common 3D printing defects (stringing, warping, under-extrusion, blobs, etc.) in real-time and is optimized for **edge deployment on Raspberry Pi**.

**Key Achievements**:
- **>90% detection accuracy** across defect categories
- Mean IoU **> 0.5**
- Quasi real-time inference (**under 60 seconds per layer**) on Raspberry Pi
- Significant reduction in material waste by enabling early print pausing

---

## ✨ Features

- Fine-tuned YOLOv5 with data augmentation and hard-negative mining
- Configuration-driven training and inference pipeline
- Ready for edge deployment (TensorFlow Lite / ONNX)
- Clean, modular codebase with proper documentation
- Reproducible setup for both development and production

---

## 🛠️ Tech Stack

- **Core**: Python, PyTorch, Ultralytics YOLOv5
- **Computer Vision**: OpenCV
- **Deployment**: TensorFlow Lite, Raspberry Pi
- **Configuration**: PyYAML
- **Visualization**: Matplotlib, Seaborn

---

## 📁 Project Structure

```bash
3D_Detect/
├── src/              # Main source code (train, infer, utils)
├── configs/          # config.yaml
├── data/             # Dataset and annotations
├── models/           # Trained weights and exported models
├── yolov5/           # YOLOv5 vendor directory
├── scripts/          # Helper scripts
├── docs/             # Results, images, reports
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
