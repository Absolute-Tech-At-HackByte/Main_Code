import os
import shutil
import random
from pathlib import Path
import glob

# Set random seed for reproducibility
random.seed(42)

# Define paths
source_dir = Path("Images")
target_dir = Path("data")

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def main():
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(source_dir.glob(f"*{ext}")))
    
    # Get all annotation files with matching image names
    all_files = []
    for img_file in image_files:
        # Find corresponding annotation file
        txt_file = source_dir / f"{img_file.stem}.txt"
        if txt_file.exists():
            all_files.append((img_file, txt_file))
        else:
            print(f"Warning: No annotation file found for {img_file.name}")
    
    # Make sure we have the classes.txt file
    classes_file = source_dir / "classes.txt"
    if not classes_file.exists():
        print("Error: classes.txt file not found in the Images directory.")
        return
    
    # Shuffle the files to ensure random distribution
    random.shuffle(all_files)
    
    # Calculate split indices
    num_files = len(all_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    
    # Split into train, validation, and test sets
    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]
    
    # Create output directories
    for split in ["train", "val", "test"]:
        create_directory(target_dir / split / "images")
        create_directory(target_dir / split / "labels")
    
    # Copy classes.txt to all label directories
    for split in ["train", "val", "test"]:
        shutil.copy(classes_file, target_dir / split / "labels" / "classes.txt")
    
    # Copy files to respective directories
    def copy_files(file_list, split):
        print(f"Copying {len(file_list)} files to {split} set...")
        for img_file, txt_file in file_list:
            # Copy image
            shutil.copy(img_file, target_dir / split / "images" / img_file.name)
            # Copy annotation
            shutil.copy(txt_file, target_dir / split / "labels" / txt_file.name)
    
    # Perform copying
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    # Create cache file for YOLOv8
    for split in ["train", "val", "test"]:
        cache_path = target_dir / split / "labels.cache"
        open(cache_path, 'w').close()  # Just create an empty file
    
    # Summary
    print("\nDataset split complete:")
    print(f"  Total files: {num_files}")
    print(f"  Train set: {len(train_files)} files ({train_ratio*100:.1f}%)")
    print(f"  Validation set: {len(val_files)} files ({val_ratio*100:.1f}%)")
    print(f"  Test set: {len(test_files)} files ({test_ratio*100:.1f}%)")
    print("\nData directory structure:")
    print(f"  {target_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/ ({len(train_files)} files)")
    print(f"  │   └── labels/ ({len(train_files)} files)")
    print(f"  ├── val/")
    print(f"  │   ├── images/ ({len(val_files)} files)")
    print(f"  │   └── labels/ ({len(val_files)} files)")
    print(f"  └── test/")
    print(f"      ├── images/ ({len(test_files)} files)")
    print(f"      └── labels/ ({len(test_files)} files)")

if __name__ == "__main__":
    main() 