from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import argparse
import torch
import traceback
import sys

# Function to predict and save images
def predict_and_save(model, image_path, output_path, output_path_txt, conf=0.5, device='cpu'):
    """
    Perform prediction on an image and save results
    
    Args:
        model: YOLO model
        image_path: Path to input image
        output_path: Path to save annotated image
        output_path_txt: Path to save detection results as text
        conf: Confidence threshold
        device: Device to use for inference
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist")
            return
            
        # Perform prediction with selected device
        results = model.predict(image_path, conf=conf, device=device)

        if not results or len(results) == 0:
            print(f"No results returned for {image_path}")
            return
            
        result = results[0]
        
        # Draw boxes on the image
        img = result.plot()  # Plots the predictions directly on the image

        # Save the result
        cv2.imwrite(str(output_path), img)
        
        # Save the bounding box data
        with open(output_path_txt, 'w') as f:
            for box in result.boxes:
                # Extract the class id and bounding box coordinates
                cls_id = int(box.cls)
                x_center, y_center, width, height = box.xywh[0].tolist()
                confidence = float(box.conf)
                
                # Write bbox information in the format [class_id, x_center, y_center, width, height, confidence]
                f.write(f"{cls_id} {x_center} {y_center} {width} {height} {confidence:.4f}\n")
                
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function for prediction"""
    print("=== License Plate Detection Prediction ===")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        DEFAULT_DEVICE = '0'  # Use GPU if available
        print("CUDA is available! Using GPU for prediction.")
    else:
        DEFAULT_DEVICE = 'cpu'  # Fall back to CPU
        print("CUDA is not available. Using CPU for prediction.")

    parser = argparse.ArgumentParser(description="Run prediction on images with trained model")
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--source', type=str, default=None, help='Source directory with images')
    args = parser.parse_args()

    try:
        # Get the current directory
        this_dir = Path(__file__).parent
        os.chdir(this_dir)
        
        # Load YAML configuration
        yaml_path = this_dir / 'yolo_params.yaml'
        if not yaml_path.exists():
            print(f"Error: YAML file not found: {yaml_path}")
            return
            
        with open(yaml_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")
                return
                
            # Check for test directory in config
            if 'test' in data and data['test'] is not None:
                test_path = data['test']
                images_dir = Path(this_dir / test_path)
            else:
                print("No test field found in yolo_params.yaml, please add the test field with the path to the test images")
                return
        
        # Override with command line source if provided
        if args.source is not None:
            images_dir = Path(args.source)
        
        # Check if images directory exists
        print(f"Looking for images in: {images_dir}")
        if not images_dir.exists():
            print(f"Error: Images directory {images_dir} does not exist")
            return

        if not images_dir.is_dir():
            print(f"Error: {images_dir} is not a directory")
            return
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        if not image_files:
            print(f"Error: No image files found in {images_dir}")
            return
        print(f"Found {len(image_files)} image files")

        # Load the YOLO model
        detect_path = this_dir / "runs" / "detect"
        
        if not detect_path.exists():
            print(f"Error: No detection runs found at {detect_path}. Please train the model first.")
            return
            
        # Get available training folders
        train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
        if len(train_folders) == 0:
            print("Error: No training folders found")
            return
            
        # Select training folder
        idx = 0
        if len(train_folders) > 1:
            print("Select the training folder:")
            for i, folder in enumerate(train_folders):
                print(f"{i}: {folder}")
                
            # Get user input
            try:
                choice = input()
                if not choice.isdigit():
                    choice = -1
                else:
                    choice = int(choice)
                    
                if choice < 0 or choice >= len(train_folders):
                    print(f"Invalid choice. Using default folder: {train_folders[0]}")
                else:
                    idx = choice
            except Exception as e:
                print(f"Error getting user input: {e}")
                print(f"Using default folder: {train_folders[0]}")

        # Get model path
        model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
        print(f"Loading model from {model_path}")
        
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return
        
        # Load model on selected device
        model = YOLO(model_path)
        print(f"Running inference on {args.device} with confidence threshold {args.conf}")

        # Setup output directories
        output_dir = this_dir / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create images and labels subdirectories
        images_output_dir = output_dir / 'images'
        labels_output_dir = output_dir / 'labels'
        images_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)

        # Process images
        success_count = 0
        for img_path in image_files:
            try:
                output_path_img = images_output_dir / img_path.name
                output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name
                
                # Predict and save
                success = predict_and_save(
                    model=model,
                    image_path=img_path,
                    output_path=output_path_img,
                    output_path_txt=output_path_txt,
                    conf=args.conf,
                    device=args.device
                )
                
                if success:
                    success_count += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                traceback.print_exc()
        
        print(f"Successfully processed {success_count} of {len(image_files)} images")
        print(f"Predicted images saved in {images_output_dir}")
        print(f"Bounding box labels saved in {labels_output_dir}")
        
        # Evaluate on test set
        try:
            print("\nEvaluating model on test set...")
            metrics = model.val(data=yaml_path, split="test", device=args.device)
            
            print("\n=== EVALUATION METRICS ===")
            print(f"mAP50-95: {metrics.box.map:.4f}")
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"Precision: {metrics.box.p:.4f}")
            print(f"Recall: {metrics.box.r:.4f}")
            print("==========================\n")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main() 