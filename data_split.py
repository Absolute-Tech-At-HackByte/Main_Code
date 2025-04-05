import os
import random
import shutil
from pathlib import Path

# Define paths
data_dir = Path('data')
train_dir = data_dir / 'train'
train_images_dir = train_dir / 'images'
train_labels_dir = train_dir / 'label'

# Create validation and test directories
val_dir = data_dir / 'val'
val_images_dir = val_dir / 'images'
val_labels_dir = val_dir / 'label'

test_dir = data_dir / 'test'
test_images_dir = test_dir / 'images'
test_labels_dir = test_dir / 'label'

# Create directories if they don't exist
for directory in [val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
    os.makedirs(directory, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
total_files = len(image_files)

# Calculate split sizes (70% train, 15% validation, 15% test)
val_size = int(total_files * 0.15)
test_size = int(total_files * 0.15)
train_size = total_files - val_size - test_size

print(f"Total files: {total_files}")
print(f"Train size: {train_size} ({train_size/total_files:.1%})")
print(f"Validation size: {val_size} ({val_size/total_files:.1%})")
print(f"Test size: {test_size} ({test_size/total_files:.1%})")

# Shuffle the data
random.seed(42)  # for reproducibility
random.shuffle(image_files)

# Split the data
val_files = image_files[:val_size]
test_files = image_files[val_size:val_size+test_size]
train_files = image_files[val_size+test_size:]

# Helper function to move files
def move_files(file_list, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    for filename in file_list:
        # Move image
        src_img = os.path.join(src_img_dir, filename)
        dst_img = os.path.join(dst_img_dir, filename)
        shutil.copy2(src_img, dst_img)
        
        # Move corresponding label file (assuming same name with .txt extension)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        src_label = os.path.join(src_label_dir, label_filename)
        dst_label = os.path.join(dst_label_dir, label_filename)
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"Warning: Label file not found for {filename}")

# Copy the classes.txt file to validation and test directories
classes_src = os.path.join(train_labels_dir, 'classes.txt')
if os.path.exists(classes_src):
    shutil.copy2(classes_src, os.path.join(val_labels_dir, 'classes.txt'))
    shutil.copy2(classes_src, os.path.join(test_labels_dir, 'classes.txt'))

# Move files to validation and test directories
print("Moving validation files...")
move_files(val_files, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir)

print("Moving test files...")
move_files(test_files, train_images_dir, train_labels_dir, test_images_dir, test_labels_dir)

print("Data split complete!")
print(f"Train: {len(train_files)} images")
print(f"Validation: {len(val_files)} images")
print(f"Test: {len(test_files)} images") 