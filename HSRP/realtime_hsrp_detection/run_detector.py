#!/usr/bin/env python
import os
import argparse
from pathlib import Path

def main():
    """
    Simple wrapper to run the HSRP real-time detector
    """
    print("HSRP Real-time License Plate Detector")
    print("=====================================")
    
    # Get script directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for model file
    model_path = script_dir / "best.pt"
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please ensure you've copied the model file to this directory.")
        return
    
    # Check for classes file
    classes_path = script_dir / "classes.txt"
    if not classes_path.exists():
        print(f"WARNING: Classes file not found at {classes_path}")
        print("Default classes will be used: ['ordinary', 'hsrp']")
    
    # Create detector directory
    detections_dir = script_dir / "detections"
    detections_dir.mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Run HSRP License Plate Detector")
    parser.add_argument("--mode", type=str, choices=["webcam", "video"], default="webcam", 
                       help="Detection mode: webcam or video file")
    parser.add_argument("--video", type=str, default=None, 
                       help="Path to video file (required if mode=video)")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device ID (default: 0)")
    parser.add_argument("--conf", type=float, default=0.5, 
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--cooldown", type=float, default=1.0, 
                       help="Cooldown time between detections (default: 1.0)")
    parser.add_argument("--cpu", action="store_true", 
                       help="Force CPU processing instead of GPU")
    
    args = parser.parse_args()
    
    # Build command
    cmd = ["python", "realtime_detector.py"]
    
    # Add device
    if args.cpu:
        cmd.extend(["--device", "cpu"])
    else:
        cmd.extend(["--device", "cuda"])
    
    # Add confidence threshold
    cmd.extend(["--conf", str(args.conf)])
    
    # Add cooldown time
    cmd.extend(["--cooldown", str(args.cooldown)])
    
    # Add source (webcam or video)
    if args.mode == "video":
        if not args.video:
            print("ERROR: Video path must be provided when mode is 'video'")
            return
        cmd.extend(["--video", args.video])
    else:
        cmd.extend(["--camera", str(args.camera)])
    
    # Print command
    print(f"Running command: {' '.join(cmd)}")
    print("Press Ctrl+C to stop")
    print("=====================================")
    
    # Execute command
    import subprocess
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error running detector: {e}")

if __name__ == "__main__":
    main() 