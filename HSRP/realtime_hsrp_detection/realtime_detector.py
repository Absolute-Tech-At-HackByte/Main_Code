import os
import sys
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from ultralytics import YOLO

class RealtimeHSRPDetector:
    def __init__(self, model_path, confidence=0.5, cooldown_time=1.0, device="cuda"):
        """
        Initialize the Realtime HSRP Detector
        
        Args:
            model_path: Path to the YOLOv8 model
            confidence: Detection confidence threshold
            cooldown_time: Time in seconds between processing frames (hidden from user)
            device: Computing device ('cuda' or 'cpu')
        """
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Setup model
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        # Use CUDA if available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            device = "cpu"
        self.device = device
        
        # Load the model
        self.model = YOLO(model_path)
        print(f"Model loaded successfully on {device}")
        
        # Detection parameters
        self.conf_threshold = confidence
        self.cooldown_time = cooldown_time
        self.last_detection_time = 0
        
        # Load class names
        self.class_names = self._load_class_names()
        print(f"Classes: {self.class_names}")
        
        # Detection history for tracking
        self.detection_history = []
        self.plate_detections = {}
        self.ordinary_count = 0
        self.hsrp_count = 0
        self.current_detection_type = None  # Track the current detection type for display
        
        # Create output directory
        self.output_dir = self.script_dir / "detections"
        self.output_dir.mkdir(exist_ok=True)
        
        # For optimized detection
        self.last_boxes = []
        self.smooth_detections = deque(maxlen=3)  # For temporal smoothing
        
    def _load_class_names(self):
        """Load class names from classes.txt file"""
        classes_path = self.script_dir / "classes.txt"
        if not classes_path.exists():
            return ["ordinary", "hsrp"]  # Default class names
            
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
            return class_names
        
    def process_frame(self, frame):
        """
        Process a video frame for HSRP detection
        
        Args:
            frame: Input frame from video stream
            
        Returns:
            processed_frame: Frame with detections drawn
            detections: List of detection results
        """
        if frame is None:
            return None, []
            
        # Create a display frame (won't modify the original)
        display_frame = frame.copy()
        
        # Check cooldown time (silently - no display)
        current_time = time.time()
        time_since_last = current_time - self.last_detection_time
        
        # Skip processing if in cooldown period
        if time_since_last < self.cooldown_time:
            # If we have previous detections, use them for smooth display
            if self.smooth_detections:
                last_detection = self.smooth_detections[-1]
                self._draw_detections(display_frame, last_detection)
                return display_frame, last_detection
            else:
                # No previous detections to display
                return display_frame, []
        
        # Process with YOLO model
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )[0]
            
            # Get dimensions
            height, width = frame.shape[:2]
            
            # Process detections
            detections = []
            detected_something = False
            
            for box in results.boxes:
                # Get coordinates (normalized to pixel values)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Get class and confidence
                cls_id = int(box.cls)
                conf = float(box.conf)
                
                # Get class name
                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
                
                # Save detection
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })
                
                # Set current detection type for display
                self.current_detection_type = class_name
                
                # Extract license plate image
                plate_img = frame[y1:y2, x1:x2].copy()
                
                # Generate unique ID for this detection
                plate_id = f"{class_name}_{x1}_{y1}_{int(conf*100)}"
                
                # Save this detection if it's new
                if plate_id not in self.plate_detections:
                    # Update detection time
                    self.last_detection_time = current_time
                    detected_something = True
                    
                    # Save the license plate image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{class_name}_{timestamp}_{conf:.2f}.jpg"
                    save_path = self.output_dir / filename
                    cv2.imwrite(str(save_path), plate_img)
                    
                    # Add to detection history
                    self.plate_detections[plate_id] = {
                        "time": time.time(),
                        "class": class_name,
                        "confidence": conf,
                        "image_path": save_path
                    }
                    
                    # Update counters
                    if class_name.lower() == "hsrp":
                        self.hsrp_count += 1
                    else:
                        self.ordinary_count += 1
                    
                    # Print detection info
                    print(f"Detected {class_name} plate with {conf:.2f} confidence")
                    print(f"Saved to {save_path}")
            
            # Add to smoothing buffer
            if detections:
                self.smooth_detections.append(detections)
            
            # Draw results on display frame
            self._draw_detections(display_frame, detections)
            
            return display_frame, detections
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return display_frame, []
    
    def _draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class"]
            conf = det["confidence"]
            
            # Set color based on class
            if class_name.lower() == "hsrp":
                color = (0, 0, 255)  # Red for HSRP
            else:
                color = (0, 255, 0)  # Green for ordinary
                
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {conf:.2f}"
            
            # Calculate label position
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Only display current detection type at the bottom of the frame
        if self.current_detection_type and detections:
            # Background for text at the bottom
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
            
            # Show current detection type
            if self.current_detection_type.lower() == "hsrp":
                status_text = f"HSRP Plate Detected"
                color = (0, 140, 255)  # Orange-red for HSRP
            else:
                status_text = f"Ordinary Plate Detected"
                color = (0, 255, 140)  # Green for ordinary
                
            cv2.putText(frame, status_text, (w//2 - 150, h-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def start_webcam(self, camera_id=0, window_name="HSRP Real-time Detection"):
        """
        Start real-time detection using webcam
        
        Args:
            camera_id: Camera device ID (default 0)
            window_name: Name of the display window
        """
        # Initialize video capture
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Unable to open camera {camera_id}")
            return
            
        print(f"Starting real-time detection from camera {camera_id}")
        print(f"Press 'q' to quit")
        
        # Set video parameters (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # FPS calculation (for program monitoring only, not displayed)
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break
                    
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Calculate FPS (not displayed, for monitoring only)
                fps_frame_count += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time > 1.0:  # Update FPS every second
                    fps = fps_frame_count / elapsed_time
                    fps_frame_count = 0
                    fps_start_time = time.time()
                    # Print FPS to console
                    print(f"Processing at {fps:.1f} FPS")
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")
            
    def start_video_file(self, video_path, window_name="HSRP Real-time Detection"):
        """
        Process a video file for HSRP detection
        
        Args:
            video_path: Path to video file
            window_name: Name of the display window
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
            
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Starting detection on video: {video_path}")
        print(f"Total frames: {frame_count}, FPS: {fps:.2f}")
        print(f"Press 'q' to quit, space to pause/resume")
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Video playback control
        paused = False
        frame_pos = 0
        
        try:
            while True:
                if not paused:
                    # Read frame
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("End of video or failed to read frame")
                        break
                        
                    # Update frame position
                    frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    # Process frame
                    processed_frame, detections = self.process_frame(frame)
                    
                    # Add progress bar
                    height, width = processed_frame.shape[:2]
                    progress = frame_pos / frame_count
                    bar_width = int(width * progress)
                    cv2.rectangle(processed_frame, (0, height - 30), (bar_width, height - 20), (0, 120, 255), -1)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print("Video paused" if paused else "Video resumed")
                
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            print("Video processing completed")

def main():
    """Main function to start real-time HSRP detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time HSRP License Plate Detector")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Cooldown time between detections (seconds)")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device (cuda or cpu)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (optional)")
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get model path
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)
    
    # Create detector
    detector = RealtimeHSRPDetector(
        model_path=model_path,
        confidence=args.conf,
        cooldown_time=args.cooldown,
        device=args.device
    )
    
    # Start detection
    if args.video:
        detector.start_video_file(args.video)
    else:
        detector.start_webcam(args.camera)

if __name__ == "__main__":
    main() 