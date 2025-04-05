# Real-time HSRP License Plate Detection

This system allows real-time detection of license plates (ordinary and HSRP) using a YOLOv8 model with OpenCV integration.

## Features

- Real-time license plate detection using webcam
- Video file processing
- Automatic saving of detected license plates
- Optimized detection with temporal smoothing
- Background cooldown period for stable detection
- Minimal, clean display showing only current detection type
- GPU acceleration with CUDA support
- Visual feedback with detection boxes and labels

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- OpenCV
- Ultralytics YOLOv8

## Setup

1. Ensure the trained model file (`best.pt`) is in this directory
2. Ensure the class names file (`classes.txt`) is in this directory
3. Install required dependencies:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install opencv-python numpy ultralytics
   ```

## Directory Structure

```
realtime_hsrp_detection/
├── best.pt              # YOLOv8 model file
├── classes.txt          # Class names file
├── realtime_detector.py # Main detector implementation
├── run_detector.py      # Simplified wrapper script
├── README.md            # This file
└── detections/          # Folder for saved detections (created automatically)
```

## Usage

### Running with Webcam

```bash
python run_detector.py --mode webcam --camera 0 --conf 0.5 --cooldown 1.0
```

- `--camera`: Camera device ID (default: 0)
- `--conf`: Detection confidence threshold (default: 0.5)
- `--cooldown`: Time in seconds between detections for smoother operation (default: 1.0)
- `--cpu`: Force CPU processing instead of GPU

### Running with Video File

```bash
python run_detector.py --mode video --video path/to/video.mp4
```

### Direct Usage

You can also run the detector directly with more options:

```bash
python realtime_detector.py --model best.pt --conf 0.5 --cooldown 1.0 --device cuda --camera 0
```

or for video:

```bash
python realtime_detector.py --model best.pt --conf 0.5 --cooldown 1.0 --device cuda --video path/to/video.mp4
```

## Controls

- Press `q` to quit the application
- When processing video files, press `spacebar` to pause/resume playback

## Display Information

The detector shows a clean, minimal interface:
- Detection boxes and confidence scores for each plate
- Current detection type (HSRP or Ordinary) displayed at the bottom of the screen
- Progress bar for video playback (video mode only)

## Output

Detected license plates will be saved to the `detections/` folder with the following naming format:
```
[class_name]_[timestamp]_[confidence].jpg
```

Example: `hsrp_20230527_153045_0.89.jpg` 