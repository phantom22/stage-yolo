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
import zipfile

import math

from skmultilearn.model_selection import iterative_train_test_split

from yolo_config import *
from util import zip_run
from helpers import visualize
import dataset
from yaml_parse import *

from ultralytics import YOLO

def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    return f"{int(minutes)}m {int(seconds)}s"

def get_next_abs_run_dir(config, desired_name):
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
    
    config["abs_run_dir"] = YOLO_RUNS_ABS_DIR / desired_name
    config["name"] = desired_name

def prepare_train_val_txt_files(config):
    abs_run_dir = config["abs_run_dir"]
    run_name = config["name"]
    strategy = config["strategy"]

    ind = config["ind"]
    indices = ind["indices"]
    split_ratio = ind["split_ratio"]
    train = ind["train"]
    val = ind["val"]
    
    if not abs_run_dir.is_dir():
        if input(f"allow mkdir 'runs/segment/{run_name}'? (y/n): ") == "y":
            abs_run_dir.mkdir()
            print_fs(f"created 'runs/segment/{run_name}'")
        else:
            print("aborting...")
            exit(0)

    if val is not None:
        train_files = train
        val_files = val
    elif indices is None:
        # --- APPROACH A ---

        # random.shuffle(IMAGE_IDS)
        # split_idx = int(len(IMAGE_IDS) * YOLO_TRAIN_TEST_SPLIT_RATIO)
        # train_files = IMAGE_IDS[:split_idx]
        # val_files = IMAGE_IDS[split_idx:]

        # --- APPROACH B ---

        dataset_bin_df = dataset.dataset_bin_df
        X = np.array([[x] for x in range(dataset_bin_df.shape[1])])
        Y = dataset_bin_df.T.values
        X_train, Y_train, X_test, Y_test = iterative_train_test_split(
            X, Y, test_size=0.2
        )

        train_ids_x = [int(x[0]) for x in X_train]
        val_ids_x = [int(x[0]) for x in X_test]

        train_files = [IMAGE_IDS[i] for i in train_ids_x]
        val_files = [IMAGE_IDS[i] for i in val_ids_x]
    else:
        if input("visualize indices selection? (y/n): ") == "y":
            visualize(indices)

        random.shuffle(indices)
        split_idx = int(len(indices) * YOLO_TRAIN_TEST_SPLIT_RATIO)
        train_files = indices[:split_idx]
        val_files = indices[split_idx:]

    abs_train_txt = abs_run_dir / "train.txt"
    abs_val_txt = abs_run_dir / "val.txt"

    cumulative_train = []

    if base_model_dir is not None:
        data_json_path = base_model_dir / "data.json"
        base_model_name = base_model_dir.name

        if not data_json_path.is_file():
            print_error(f"there is no '{base_model_name}/data.json' file")
            exit(1)

        with data_json_path.open("r") as f:
            data_json = json.load(f)

            if "val" not in data_json:
                print_error(f"{base_model_name}/data.json: file does not contain the 'val' attribute")
                exit(1)
            if "train" not in data_json:
                print_error(f"{base_model_name}/data.json: file does not contain the 'train' attribute")
                exit(1)
            if "cumulative_train" not in data_json:
                print_error(f"{base_model_name}/data.json: file does not contain the 'cumulative_train' attribute")
                exit(1)

            val_files = list(set(val_files) | set(data_json["val"]))

            cumulative_train_set = set(data_json["train"]) | set(data_json["cumulative_train"])

            train_files = [v for v in train_files if v not in cumulative_train_set]
            cumulative_train = list(set(train_files) | cumulative_train_set)
            if len(train_files) == 0:
                    print_error(f"{strategy}.yaml: instants.index_selection[{config['instant']}]: the resulting train indices were either empty or all pointed to files that were already used in training, during previous instants")
                    exit(1)
                
        mode = "a"
    else:
        mode = "w"

    train_files.sort()
    val_files.sort()
    cumulative_train.sort()

    json_data_yaml_dump = {
        "train": train_files,
        "val": val_files,
        "cumulative_train": cumulative_train
    }
    
    with open(abs_run_dir / "data.json", 'w') as f:
        json.dump(json_data_yaml_dump, f, indent=2)
    print_fs(f"written to {run_name}/data.json")

    with open(abs_train_txt, mode) as f:
        f.writelines([str(YOLO_IMAGES_REL_DIR / strategy_labels / f"{i}.jpg") + "\n" for i in train_files])
    print_fs(f"written to {run_name}/train.txt")

    with open(abs_val_txt, mode) as f:
        f.writelines([str(YOLO_IMAGES_REL_DIR / strategy_labels / f"{i}.jpg") + "\n" for i in val_files])
    print_fs(f"written to {run_name}/val.txt")

if __name__ == "__main__":
    nargs = len(sys.argv)

    args = {
        "name": sys.argv[1] if nargs > 1 else None,
        "strategy": sys.argv[2] if nargs > 2 else YOLO_DEFAULT_STRATEGY,
        "base": sys.argv[3] if nargs > 3 else YOLO_DEFAULT_BASE_MODEL
    }

    if args["name"] == "help":
        print(
            gb("USAGE:") + "\n" +
                "  train.py\n" +
                "  train.py <model_name>\n" +
                "  train.py <model_name> <strategy>\n" +
                "  train.py <model_name> <strategy> <base_model>\n" +
                "  train.py help\n"
        )
        exit(0)

    DATASET_IMG_ABS_DIR.mkdir(exist_ok=True)
    YOLO_RUNS_ABS_DIR.mkdir(exist_ok=True)

    #######################################
    ##             STRATEGY              ##
    #######################################

    while args["strategy"] not in YOLO_AVAILABLE_TRAIN_STRATEGIES:
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
            args["strategy"] = YOLO_DEFAULT_STRATEGY
            break

        args["strategy"] = i

    strategy = args["strategy"]
    strategy_yaml_path = YOLO_STRATEGIES_ABS_DIR / (args["strategy"]+".yaml")

    if not strategy_yaml_path.is_file():
        print_error(f"{strategy}.yaml: the specified strategy file does not exist")
        exit(1)

    with open(strategy_yaml_path, 'r') as f:
        strategy_yaml = yaml.safe_load(f)

    config = yaml_get_strategy_details(args["strategy"], strategy_yaml)

    #######################################
    ##          STRATEGY LABELS          ##
    #######################################
    
    if not config["labels"] in YOLO_AVAILABLE_TRAIN_LABELS:
        print_error(f"{strategy}.yaml: 'label': labels/{config['labels']} does not exist")
        exit(1)

    #######################################
    ##             BASE MODEL            ##
    #######################################

    if args["base"] != YOLO_DEFAULT_BASE_MODEL:
        base_model_dir = YOLO_RUNS_ABS_DIR / config["base"]
        base_model_name = base_model_dir / "weights/best.pt"

        if not base_model_name.is_file():
            print_error(f"{config['base']}/weights/base.pt: file does not exist")
            exit(1)

        base_data_yaml_path = base_model_dir / "data.yaml"

        if not base_data_yaml_path.is_file():
            print_error(f"{config['base']}/data.yaml: file does not exist")
            exit(1)
        
        with open(base_data_yaml_path, 'r') as f:
            base_data_yaml = yaml.safe_load(f)

        base_model_config = yaml_get_run_details(config["base"], base_data_yaml)
        config["base"] = base_model_name

        if config["strategy"] != base_model_config["strategy"]:
            print_warning(f"{config['strategy']}.yaml: this does not match '{base_model_config['strategy']}' strategy")
        
        if config["labels"] != base_model_config["labels"]:
            print_error(f"{config['strategy']}.yaml: 'labels' conflicts with '{base_model_config['strategy']}' labels")
            exit(1)

        config["instant"] = base_model_config["instant"]+1
    else:
        config["base"] = args["base"]

    #######################################
    ##       PREPARE YOLO FILESYSTEM     ##
    #######################################

    get_next_abs_run_dir(config, args["name"])
    abs_run_dir = config["abs_run_dir"]
    run_name = config["name"]

    yaml_parse_strategy(config, strategy_yaml)

    print(config["instant"])
    exit(0)

    prepare_train_val_txt_files(config)

    #######################################
    ##     PREPARE STRATEGY data.yaml    ##
    #######################################

    strategy_yaml['instant'] = config["instant"]
    strategy_yaml['path'] = "."
    strategy_yaml['train'] = f"runs/segment/{run_name}/train.txt"
    strategy_yaml['val'] = f"runs/segment/{run_name}/val.txt"
    strategy_yaml['strategy'] = args["strategy"]
    
    final_data_yaml = abs_run_dir / "data.yaml"
    with open(final_data_yaml, 'w') as f:
        yaml.dump(strategy_yaml, f)
    print_fs(f"written to {run_name}/data.yaml")

    if instant == 0:
        confirmation_text = f"proceed to train [\n  run_name='{run_name}', strategy='{strategy}', labels='{strategy_labels}',\n  split_ratio={restriction_spec['split_ratio']}, base_model='{base_model}'\n]? (y/n): "
    else:
        confirmation_text = f"proceed to train [\n  run_name='{run_name}', strategy='{strategy}', labels='{strategy_labels}',\n  split_ratio={restriction_spec['split_ratio']}, base_model='{base_model_dir.name}', instant={instant}\n]? (y/n): "

    if input(confirmation_text) != "y":
        if input(f"also remove the 'runs/segment/{run_name}/' folder? (y/n): ") == "y":
            try:
                shutil.rmtree(abs_run_dir)
                print_fs(f"succesfully removed 'runs/segment/{run_name}'")
            except Exception as e:
                print_error(f"couldn't remove 'runs/segment/{run_name}/', reason: {e}")
        print("aborting...")
        exit(0)

    dataset_img_sym_link = YOLO_ABS_DIR / "data/images" / strategy_labels
    
    if dataset_img_sym_link.is_file() or dataset_img_sym_link.is_symlink():
        dataset_img_sym_link.unlink()
        print_fs(f"removed existing 'yolo/data/images/{strategy}' file/symlink.")
    elif dataset_img_sym_link.is_dir():
        print_error(f"a folder named 'yolo/data/images/{strategy}' exists where it should be! This directory is meant to host symlinks that point to the actual image folder 'dataset/images'.")
        exit(1)

    dataset_img_sym_link.symlink_to(DATASET_IMG_ABS_DIR)
    print_fs(f"created 'yolo/data/images/{strategy}' symlink that points to 'dataset/images'")

    model = YOLO(base_model) 

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
    print_fs(f"removed 'yolo/data/images/{strategy}' symlink")

    zip_run(run_name)