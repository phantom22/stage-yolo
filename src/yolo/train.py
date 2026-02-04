import os
import sys
from pathlib import Path
import shutil
import numpy as np
import yaml
import time
import json
# import random

from skmultilearn.model_selection import iterative_train_test_split

from yolo_config import *
from yolo_util import zip_run
from dataset import dataset_bin_df

from ultralytics import YOLO

def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    return f"{int(minutes)}m {int(seconds)}s"

def get_next_run_dir(desired_name):
    existing_runs = [d for d in os.listdir(YOLO_RUNS_DIR) if d.startswith(RUNS_PREFIX)]
    indices = [int(r.replace(RUNS_PREFIX, "")) for r in existing_runs if r.replace(RUNS_PREFIX, "").isdigit()]
    next_idx = max(indices) + 1 if indices else 1
    default_name = f"{RUNS_PREFIX}{next_idx}"

    if desired_name is not None:
        desired_dir = YOLO_RUNS_DIR / desired_name
        while desired_dir.is_dir():
            print_warning(f"a folder named 'runs/segment/{desired_name}' already exists")
            i = input(
                "please enter another name (or '"+
                    rb("o")+"' = override existing model | '"+
                    rb("q")+"' = quit | '"+
                    rb("d")+f"' = suggested name '{default_name}'): "
            )

            if i == 'o':
                try:
                    for item in desired_dir.iterdir():
                        if item.is_file() or item.is_symlink():
                            item.unlink()
                            print_fs(f"removed the 'runs/segment/{desired_name}/{item.name}' file")
                        elif item.is_dir():
                            shutil.rmtree(item)
                            print_fs(f"removed the 'runs/segment/{desired_name}/{item.stem}/' folder")
                    print(f"succesfully emptied 'runs/segment/{desired_name}/'")
                except Exception as e:
                    print_error(f"couldn't empty 'runs/segment/{desired_name}/' contents")
                    exit(1)
                break

            if i == 'q':
                print("aborting...")
                exit(0)

            if i == 'd':
                desired_name = default_name
                break

            desired_name = i
            desired_dir = YOLO_RUNS_DIR / desired_name
            
    else:
        desired_name = default_name
    
    return YOLO_RUNS_DIR / desired_name, desired_name

def prepare_split_and_json(run_dir, run_name):
    if not run_dir.is_dir():
        if input(f"allow mkdir 'runs/segment/{run_name}'? (y/n): ") == "y":
            run_dir.mkdir()
            print_fs(f"created 'runs/segment/{run_name}'")
        else:
            print("aborting...")
            exit(0)

    images = sorted([int(f.stem) for f in YOLO_IMG_DIR.iterdir() if f.name.lower().endswith(IMG_EXTENSIONS)])

    if len(images) == 0:
        print_error(f"The image directory does not contain any images!")
        exit(1)

    # --- APPROACH A ---
    # random.shuffle(images)
    # split_idx = int(len(images) * YOLO_SPLIT_RATIO)
    # train_files = images[:split_idx]
    # val_files = images[split_idx:]

    # --- APPROACH B ---
    X = np.array([[x] for x in range(dataset_bin_df.shape[1])])
    Y = dataset_bin_df.T.values
    X_train, Y_train, X_test, Y_test = iterative_train_test_split(
        X, Y, test_size=0.2
    )

    train_ids_x = [int(x[0]) for x in X_train]
    val_ids_x = [int(x[0]) for x in X_test]
    
    train_files = [images[i] for i in train_ids_x]
    val_files = [images[i] for i in val_ids_x]

    json_data_yaml_dump = {
        "train": train_files,
        "val": val_files
    }
    
    with open(run_dir / "data.json", 'w') as f:
        json.dump(json_data_yaml_dump, f, indent=2)
    print_fs(f"written to {run_name}/data.json")

    # 3. Prepare paths for YOLO .txt files
    train_txt = run_dir / "train.txt"
    val_txt = run_dir / "val.txt"

    with open(train_txt, 'w') as f:
        f.writelines([f"{YOLO_IMG_DIR}/{idx}.jpg\n" for idx in train_files])
    print_fs(f"written to {run_name}/train.txt")
    
    with open(val_txt, 'w') as f:
        f.writelines([f"{YOLO_IMG_DIR}/{idx}.jpg\n" for idx in val_files])
    print_fs(f"written to {run_name}/val.txt")

    return train_txt, val_txt

if __name__ == "__main__":
    YOLO_IMG_DIR.mkdir(exist_ok=True)
    YOLO_RUNS_DIR.mkdir(exist_ok=True)

    desired_name = sys.argv[1] if len(sys.argv) > 1 else None

    run_dir, run_name = get_next_run_dir(desired_name)
    train_txt, val_txt = prepare_split_and_json(run_dir, run_name)

    if train_txt:
        with open(MAIN_YAML_FILE, 'r') as f:
            data = yaml.safe_load(f)
            
        data['train'] = str(train_txt)
        data['val'] = str(val_txt)
        
        temp_yaml = run_dir / "data.yaml"
        with open(temp_yaml, 'w') as f:
            yaml.dump(data, f)
        print_fs(f"written to {run_name}/data.yaml")

        if input("all ready, proceed? (y/n): ") != "y":
            if input(f"also remove the 'runs/segment/{run_name}/' folder? (y/n): ") == "y":
                try:
                    shutil.rmtree(run_dir)
                    print_fs(f"succesfully removed 'runs/segment/{run_name}'")
                except Exception as e:
                    print_error(f"couldn't remove 'runs/segment/{run_name}/', reason: {e}")
            print("aborting...")
            exit(0)
        
        model = YOLO("yolo11m-seg.pt") 

        print(f"Training {run_dir.name}...")
        start_t = time.time()
        
        model.train(
            data=temp_yaml,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,      
            name=run_name,
            exist_ok=True,
            cache=True
        )
        
        total_duration = time.time() - start_t
        
        print("\n" + "="*40)
        print(f"Training Finished in {format_time(total_duration)}")
        print(f"JSON log saved to: {run_dir / 'data.json'}")
        print("="*40)

        zip_run(run_name)