# License Plate Detection System (GPU-Optimized)

This project implements a YOLOv8-based license plate detection system optimized for GPU usage to detect and classify both ordinary and HSRP (High Security Registration Plate) license plates.

## Prerequisites

- CUDA-compatible GPU
- Python 3.8+
- PyTorch (with CUDA support)
- Ultralytics YOLOv8

## Installation

```bash
# Install required packages
pip install ultralytics opencv-python pyyaml matplotlib

# Check if CUDA is available (should return True)
python -c "import torch; print(torch.cuda.is_available())"
```

## Data Structure

Ensure your data is organized in the following structure:
```
data/
  train/
    images/ - Contains training images
    label/  - Contains training labels (YOLO format .txt files)
  val/
    images/ - Contains validation images
    label/  - Contains validation labels
  test/
    images/ - Contains test images
    label/  - Contains test labels
```

## Training

The model is optimized for GPU training with the following improvements:
- Uses YOLOv8s (small) for better accuracy with reasonable speed
- Increased epochs for better learning
- Optimized augmentation parameters
- Advanced learning rate scheduling
- HSV color augmentation for robustness

To train the model:

```bash
# Train with default GPU settings
python train.py

# Train with custom parameters
python train.py --epochs 200 --batch 32 --mosaic 0.7 --device 0
```

Available parameters:
- `--epochs`: Number of training epochs (default: 150)
- `--batch`: Batch size (default: 16)
- `--mosaic`: Mosaic augmentation factor (default: 0.6)
- `--imgsz`: Image size (default: 640)
- `--device`: GPU device ID (default: '0')
- `--optimizer`: Optimizer type (default: AdamW)
- `--lr0`: Initial learning rate (default: 0.001)
- `--lrf`: Final learning rate factor (default: 0.01)

## Prediction

To run inference on test images:

```bash
# Run prediction with GPU
python predict.py

# Run with custom parameters
python predict.py --conf 0.4 --device 0 --source path/to/custom/images
```

Available parameters:
- `--conf`: Confidence threshold (default: 0.5)
- `--device`: GPU device ID (default: '0')
- `--source`: Custom source directory for images

## Visualization

To visualize the results:

```bash
# Basic visualization
python visualize.py

# Generate additional confusion matrix
python visualize.py --num_images 8 --conf 0.3 --conf_matrix
```

Available parameters:
- `--num_images`: Number of images to visualize (default: 4)
- `--conf`: Confidence threshold (default: 0.25)
- `--device`: GPU device ID (default: '0')
- `--source`: Custom source directory for images
- `--conf_matrix`: Generate confusion matrix

## Performance Tuning

- **GPU Memory Issues**: If you encounter GPU memory errors, reduce batch size or model size
- **Speed vs Accuracy**: For higher accuracy, use `yolov8s.pt` or `yolov8m.pt`; for speed, use `yolov8n.pt`
- **Mosaic Augmentation**: Values around 0.5-0.7 work best; don't set to 1.0

## Results

The model saves evaluation metrics after training and prediction:
- mAP50-95: Mean Average Precision across IoU thresholds
- mAP50: Mean Average Precision at IoU 0.5
- Precision: Precision of detections
- Recall: Recall of detections

## Project Structure

- `data/` - Dataset directory
  - `train/` - Training data
  - `val/` - Validation data
  - `test/` - Test data
- `yolo_params.yaml` - Configuration file for training and inference
- `train.py` - Script for training the model
- `predict.py` - Script for making predictions on new images
- `visualize.py` - Script for visualizing model performance and results
- `classes.txt` - Class names (ordinary, hsrp)

## Tips for Better Results

- Adjust mosaic augmentation based on your dataset size (lower for smaller datasets)
- Try different model sizes (n, s, m, l, x) based on your computational resources
- Experiment with different learning rates and optimizers
- For small datasets, consider transfer learning from a pre-trained model

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Format](https://docs.ultralytics.com/datasets/detect/) 