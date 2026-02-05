import os
import sys
from pathlib import Path
import shutil
import numpy as np
import yaml
import time
import json
import random

import yaml

from skmultilearn.model_selection import iterative_train_test_split

from yolo_config import *
from util import zip_run
from helpers import visualize
import dataset

from ultralytics import YOLO

def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    return f"{int(minutes)}m {int(seconds)}s"

def get_next_abs_run_dir(desired_name):
    existing_runs = [d for d in os.listdir(YOLO_RUNS_ABS_DIR) if d.startswith(YOLO_RUN_NAME_PREFIX)]
    indices = [int(r.replace(YOLO_RUN_NAME_PREFIX, "")) for r in existing_runs if r.replace(YOLO_RUN_NAME_PREFIX, "").isdigit()]
    next_idx = max(indices) + 1 if indices else 1
    default_name = f"{YOLO_RUN_NAME_PREFIX}{next_idx}"

    if desired_name is not None:
        desired_dir = YOLO_RUNS_ABS_DIR / desired_name
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
            desired_dir = YOLO_RUNS_ABS_DIR / desired_name
            
    else:
        desired_name = default_name
    
    return YOLO_RUNS_ABS_DIR / desired_name, desired_name

def prepare_split_and_json(abs_run_dir, run_name, desired_strategy, indices_restriction=None):
    if not abs_run_dir.is_dir():
        if input(f"allow mkdir 'runs/segment/{run_name}'? (y/n): ") == "y":
            abs_run_dir.mkdir()
            print_fs(f"created 'runs/segment/{run_name}'")
        else:
            print("aborting...")
            exit(0)

    img_ids = sorted([int(f.stem) for f in DATASET_IMG_ABS_DIR.iterdir() if f.name.lower().endswith(IMG_EXTENSIONS)])

    if len(img_ids) == 0:
        print_error(f"The image directory does not contain any img_ids!")
        exit(1)

    if indices_restriction is None:
        # --- APPROACH A ---
        # random.shuffle(img_ids)
        # split_idx = int(len(img_ids) * YOLO_TRAIN_TEST_SPLIT_RATIO)
        # train_files = img_ids[:split_idx]
        # val_files = img_ids[split_idx:]

        # --- APPROACH B ---

        dataset_bin_df = dataset.dataset_bin_df
        X = np.array([[x] for x in range(dataset_bin_df.shape[1])])
        Y = dataset_bin_df.T.values
        X_train, Y_train, X_test, Y_test = iterative_train_test_split(
            X, Y, test_size=0.2
        )

        train_ids_x = [int(x[0]) for x in X_train]
        val_ids_x = [int(x[0]) for x in X_test]
        
        train_files = [img_ids[i] for i in train_ids_x]
        val_files = [img_ids[i] for i in val_ids_x]
    else:
        visualize(indices_restriction)
        random.shuffle(indices_restriction)
        split_idx = int(len(indices_restriction) * YOLO_TRAIN_TEST_SPLIT_RATIO)
        train_files = indices_restriction[:split_idx]
        val_files = indices_restriction[split_idx:]

    json_data_yaml_dump = {
        "train": train_files,
        "val": val_files
    }
    
    with open(abs_run_dir / "data.json", 'w') as f:
        json.dump(json_data_yaml_dump, f, indent=2)
    print_fs(f"written to {run_name}/data.json")

    # 3. Prepare paths for YOLO .txt files
    abs_train_txt = abs_run_dir / "train.txt"
    abs_val_txt = abs_run_dir / "val.txt"

    with open(abs_train_txt, 'w') as f:
        f.writelines([str(YOLO_IMAGES_REL_DIR / desired_strategy / f"{i}.jpg") + "\n" for i in train_files])
    print_fs(f"written to {run_name}/train.txt")
    
    with open(abs_val_txt, 'w') as f:
        f.writelines([str(YOLO_IMAGES_REL_DIR / desired_strategy / f"{i}.jpg") + "\n" for i in val_files])
    print_fs(f"written to {run_name}/val.txt")

    return abs_train_txt, abs_val_txt

if __name__ == "__main__":
    DATASET_IMG_ABS_DIR.mkdir(exist_ok=True)
    YOLO_RUNS_ABS_DIR.mkdir(exist_ok=True)

    nargs = len(sys.argv)
    desired_name = sys.argv[1] if nargs > 1 else None
    desired_strategy = sys.argv[2] if nargs > 2 else YOLO_DEFAULT_STRATEGY

    if desired_strategy is None:
        print_error("there are no available strategies in 'yolo/data/strategies'")
        exit(1)
    
    while desired_strategy not in YOLO_AVAILABLE_TRAIN_STRATEGIES:
        print_error("")

    while desired_strategy not in YOLO_AVAILABLE_TRAIN_STRATEGIES:
        print_error("the specified strategy does not exist")
        print(*YOLO_AVAILABLE_TRAIN_STRATEGIES, sep=", ")
        i = input(
            f"please specify an existing strategy from the list above (or '"+
                rb("q")+"' = quit | '"+
                rb("d")+f"' = default strategy '{YOLO_DEFAULT_STRATEGY}'): "
        )

        if i == 'q':
            print("aborting...")
            exit(0)

        if i == 'd':
            desired_strategy = YOLO_DEFAULT_STRATEGY
            break

        desired_strategy = i

    abs_run_dir, run_name = get_next_abs_run_dir(desired_name)

    strategy_data_yaml = YOLO_STRATEGIES_ABS_DIR / (desired_strategy+".yaml")

    if not strategy_data_yaml.is_file():
        print_error(f"the specified strategy file '{YOLO_DEFAULT_STRATEGY}' does not exist")
        exit(1)

    with open(strategy_data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    ####
    if 'gt' in data:
        gt = {}
        for entry in data['gt']:
            k,v = entry
            try:
                code = getattr(dataset, k)
            except Exception as e:
                print_error("the .yaml file contains an invalid code")
                exit(1)
            gt[code] = v
    else:
        gt = None

    if 'lt' in data:
        lt = {}
        for entry in data['lt']:
            k,v = entry
            try:
                code = getattr(dataset, k)
            except Exception as e:
                print_error("the .yaml file contains an invalid code")
                exit(1)
            lt[code] = v
    else:
        lt = None

    if not gt is None or not lt is None:
        indices_restriction = dataset.get_detailed_dataset_manifest().query(gt=gt, lt=lt)
    else:
        indices_restriction = None
    ####

    abs_train_txt, val_txt = prepare_split_and_json(abs_run_dir, run_name, desired_strategy, indices_restriction)

    data['path'] = "."
    data['train'] = f"runs/segment/{run_name}/train.txt"
    data['val'] = f"runs/segment/{run_name}/val.txt"
    data['strategy'] = desired_strategy
    
    final_data_yaml = abs_run_dir / "data.yaml"
    with open(final_data_yaml, 'w') as f:
        yaml.dump(data, f)
    print_fs(f"written to {run_name}/data.yaml")

    if input(f"all ready, proceed to train the '{run_name}' model with the '{desired_strategy}' strategy? (y/n): ") != "y":
        if input(f"also remove the 'runs/segment/{run_name}/' folder? (y/n): ") == "y":
            try:
                shutil.rmtree(abs_run_dir)
                print_fs(f"succesfully removed 'runs/segment/{run_name}'")
            except Exception as e:
                print_error(f"couldn't remove 'runs/segment/{run_name}/', reason: {e}")
        print("aborting...")
        exit(0)

    dataset_img_sym_link = YOLO_ABS_DIR / "data/images" / desired_strategy
    
    if dataset_img_sym_link.is_file() or dataset_img_sym_link.is_symlink():
        dataset_img_sym_link.unlink()
        print_fs(f"removed existing 'yolo/data/images/{desired_strategy}' file/symlink.")
    elif dataset_img_sym_link.is_dir():
        print_error(f"a folder named 'yolo/data/images/{desired_strategy}' exists where it should be! This directory is meant to host symlinks that point to the actual image folder 'dataset/images'.")
        exit(1)

    dataset_img_sym_link.symlink_to(DATASET_IMG_ABS_DIR)
    print_fs(f"created 'yolo/data/images/{desired_strategy}' symlink that points to 'dataset/images'")

    model = YOLO("yolo11m-seg.pt") 

    print(f"Training {gb(run_name)}...")
    start_t = time.time()
    
    model.train(
        data=final_data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,      
        name=run_name,
        exist_ok=True,
        cache=True
    )
    
    total_duration = time.time() - start_t
    
    print("="*40)
    print(f"Training Finished in {gb(format_time(total_duration))}")
    print("="*40)

    dataset_img_sym_link.unlink()
    print_fs(f"removed 'yolo/data/images/{desired_strategy}' symlink")

    zip_run(run_name)