import os
import random
from ultralytics import YOLO

# --- CONFIGURATION ---
DATA_DIR = "data"  # The folder containing images/, labels/, and train.txt
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
VAL_FILE = os.path.join(DATA_DIR, "val.txt")
YAML_FILE = os.path.join(DATA_DIR, "data.yaml")
SPLIT_RATIO = 0.8

def prepare_cvat_split():
    # Use the actual image directory to get the list of files
    img_dir = os.path.join(DATA_DIR, "images", "train")
    
    if not os.path.exists(img_dir):
        print(f"❌ Error: Image directory not found at {img_dir}")
        return False

    # Get all valid image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    if len(images) < 10:
        print(f"❌ Error: Found only {len(images)} images. Check your folder path.")
        return False

    # Shuffle and split
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    
    # Crucial: Format the paths relative to the 'data/' folder
    # This ensures YOLO finds them when 'path: data' is in your yaml
    train_lines = [f"data/images/train/{img}\n" for img in images[:split_idx]]
    val_lines = [f"data/images/train/{img}\n" for img in images[split_idx:]]

    # Write fresh files (using 'w' to overwrite old data completely)
    with open(TRAIN_FILE, 'w') as f:
        f.writelines(train_lines)
    with open(VAL_FILE, 'w') as f:
        f.writelines(val_lines)
    
    print(f"✅ Successfully scanned {len(images)} images.")
    print(f"✅ Created {TRAIN_FILE} ({len(train_lines)} lines) and {VAL_FILE} ({len(val_lines)} lines).")
    return True

if __name__ == "__main__":
    if prepare_cvat_split():
        # Load a segmentation model (using 'm' for your 4090 power)
        model = YOLO("yolo11m-seg.pt") 

        # Start Training
        # Note: 'data' points to the .yaml inside your data folder
        model.train(
            data=YAML_FILE,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,      # RTX 4090
            cache=True     # Recommended for small datasets
        )