# HSRP Number Plate Recognition and Challan Generation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-yellow)

## Overview

This system provides an end-to-end solution for High Security Registration Plate (HSRP) detection and recognition, designed for traffic management and automated challan generation. The system uses YOLOv8, a state-of-the-art object detection algorithm, to detect and classify vehicle license plates into standard and HSRP categories with high accuracy.

## Features

- **Multi-Class Detection**: Distinguishes between ordinary license plates and HSRP plates
- **GPU Acceleration**: Optimized for NVIDIA GPUs for real-time processing
- **Flexible Input Support**: Process individual images, video streams, or batch directories
- **Custom UI**: Interactive visualization tools for inspection of detection results
- **Automated Challan Generation**: Based on license plate recognition and verification
- **High Accuracy**: Trained on a diverse dataset for robust performance across conditions

## System Architecture

```
┌─────────────────┐    ┌───────────────┐    ┌─────────────────┐
│ License Plate   │    │ Plate Type    │    │ Challan         │
│ Detection       │───>│ Classification │───>│ Generation      │
└─────────────────┘    └───────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible NVIDIA GPU (recommended)
- PyTorch with CUDA support

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/username/hsrp-recognition.git
cd hsrp-recognition

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Training

```bash
python train.py --epochs 150 --device 0
```

### Prediction

```bash
python predict.py --device 0
```

### HSRP Detection and Challan Generation

```bash
cd HSRP
python run_detection.py
```

For interactive processing:

```bash
python process_custom_image.py
```

## Dataset

The system is trained on a custom dataset containing:
- Ordinary license plates
- High Security Registration Plates (HSRP)

The dataset is organized into:
- 70% training images
- 15% validation images
- 15% test images

## Model Performance

| Model     | mAP50-95 | mAP50 | Precision | Recall | FPS (RTX 3050) |
|-----------|----------|-------|-----------|--------|----------------|
| YOLOv8s   | 0.892    | 0.942 | 0.935     | 0.927  | ~15            |

## Challan Generation Process

1. License plate detection and classification
2. OCR for plate number recognition
3. Verification against RTO database
4. Rules-based challan generation for non-compliant vehicles
5. Digital challan issuance

## Visualizations

The system includes visualization tools for:
- Detection bounding boxes
- Plate type classification
- Extracted license plate regions
- Confidence scores

## Future Developments

- Integration with traffic camera networks
- Mobile application for on-the-go challan generation
- Blockchain-based challan verification system
- Enhanced OCR for improved plate number recognition

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- PyTorch team
- Transport authorities for dataset collaboration 