from yolo_config import *

import os
import re
import sys

from pathlib import Path

import cv2
import numpy as np
import yaml
import json

if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

WINDOW_NAME = "Navigation"

def numerical_sort_key(path):
    return int(path.stem)

def run_navigable_inference(model_run_path, confidence_treshold, no_train):
    model_name = model_run_path.name
    model_path = model_run_path / "weights/best.pt"
    run_data_yaml_path = model_run_path / "data.yaml"
    if not model_path.exists():
        print_error(f"Model not found at {model_path}")
        exit(1)
    if not run_data_yaml_path.exists():
        print_error(f"{model_name}/data.yaml: file not found")
        exit(1)

    with open(run_data_yaml_path, 'r') as f:
        run_data_yaml = yaml.safe_load(f)
        run_strategy = run_data_yaml.get("strategy")
        run_palette = run_data_yaml.get("palette")
        run_instant = run_data_yaml.get("instant")
        if run_strategy is None:
            print_error(f"{model_name}/data.yaml: file does not specify the used strategy")
            exit(1)

        if run_palette is None:
            print_error(f"{model_name}/data.yaml: file does not specify the used palette")
            exit(1)

        if run_instant is None:
            print_warning(f"{model_name}/data.yaml:file does not specify the current instant")
        palette_bgr = [tuple(color[::-1]) for color in run_palette]

    cache_dir_name = YOLO_CACHE_ABS_DIR.name
    if YOLO_CACHE_ABS_DIR.exists():
        for f in YOLO_CACHE_ABS_DIR.iterdir():
            f.unlink()
        print_fs(f"emptied 'yolo/{cache_dir_name}' contents")
    else:    
        YOLO_CACHE_ABS_DIR.mkdir()
        print_fs(f"created 'yolo/{cache_dir_name}' directory")

    if no_train:
        run_data_json_path = model_run_path / "data.json"
        if not run_data_json_path.is_file():
            print_error(f"{model_name}/data.json: file not found")
        
        with open(run_data_json_path, 'r') as f:
            data_json = json.load(f)
            if "cumulative_train" not in data_json:
                print_error(f"{model_name}/data.json: no 'cumulative_train' key")
                exit(1)
            if "train" not in data_json:
                print_error(f"{model_name}/data.json: no 'train' key")
                exit(1)
            cumulative_train_set = list(set(data_json["cumulative_train"]) | set(data_json["train"]))
        image_ids = [f for f in IMAGE_IDS if f not in cumulative_train_set]
        num_images = len(image_ids)
    else:
        image_ids = IMAGE_IDS
        num_images = NUM_IMAGES

    if no_train:
        print(f"running '{gb(model_name)}' model with {rb(str(confidence_treshold))} threshold value, {gb('without train')}")
    else:
        print(f"running '{gb(model_name)}' model with {rb(str(confidence_treshold))} threshold value, {rb('with train')}")
    model = YOLO(model_run_path / "weights/best.pt")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    session_cache = {}
    idx = 0
    while 0 <= idx < num_images:
        img_path = DATASET_IMG_ABS_DIR / f"{image_ids[idx]}.jpg"
        img_name = image_ids[idx]
        
        if img_name in session_cache:
            combined_img = session_cache[img_name]
        else:
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                idx += 1; continue

            # --- RESIZE STRATEGY ---
            # If image is not 640x480, resize it. 
            # Note: OpenCV resize expects (Width, Height)
            if original_img.shape[1] != YOLO_TARGET_SIZE[0] or original_img.shape[0] != YOLO_TARGET_SIZE[1]:
                original_img = cv2.resize(original_img, YOLO_TARGET_SIZE)

            # Inference on the resized image
            results = model.predict(source=original_img, imgsz=640, conf=confidence_treshold, device=0, verbose=False)
            
            for r in results:
                inf_time = r.speed['inference']
                # Annotator uses the resized original as background
                annotator = Annotator(original_img.copy(), line_width=2)
                
                if r.masks is not None:
                    # masks are already at 640x480 because the input image was that size
                    masks = r.masks.data.cpu().numpy() 
                    clss = r.boxes.cls.cpu().numpy()
                    color_list = [palette_bgr[int(c)] for c in clss]
                    annotator.masks(masks, colors=color_list, alpha=0.5)
                    
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        cls_name = model.names[cls]
                        label = f"{model.names[cls]} {box.conf[0]:.2f}"
                        bg_color = palette_bgr[cls]
                        text_color = (0, 0, 0) if cls_name == "disposable cutlery" else (255, 255, 255)
                        
                        # Bounding boxes are now perfectly aligned with the 640x480 image
                        annotator.box_label(box.xyxy[0], label, color=bg_color, txt_color=text_color)

                inferred_img = annotator.result()
                
                # Side-by-side (both are now 640x480)
                combined_img = cv2.hconcat([original_img, inferred_img])
                
                # Small Green info text
                info_text = f"{idx+1}/{num_images} {img_name}.jpg {inf_time:.1f}ms"
                pos, font, scale, thickness = (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
                cv2.putText(combined_img, info_text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(combined_img, info_text, pos, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
                
                cv2.imwrite(YOLO_CACHE_ABS_DIR / f"compare_{img_name}.jpg", combined_img)
                session_cache[img_name] = combined_img

        cv2.imshow(WINDOW_NAME, combined_img)
        
        raw_key = cv2.waitKeyEx(0)

        # window explicitly closes by clicking on the 'x' on windows
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if raw_key in [97, 81, 65361, 2424832]: # A or LEFT
            idx = max(0, idx - 1)
        elif raw_key in [100, 83, 65363, 2555904]: # D or RIGHT
            idx = min(num_images - 1, idx + 1)
        elif raw_key in [27, 113]: # ESC or Q
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if YOLO_DEFAULT_MODEL is None:
        print_warning("there are available models to use.")
        exit(0)

    argc = len(sys.argv)
    desired_model = sys.argv[1] if argc > 1 else YOLO_DEFAULT_MODEL
    try:
        confidence_treshold = float(sys.argv[2]) if argc > 2 and sys.argv[2] else YOLO_CONF_THRESHOLD
    except Exception as e:
        print_error("the confidence threshold must be a number between 0 and 1")
        exit(1)

    if confidence_treshold > 1 or confidence_treshold < 0:
        print_error("the confidence threshold must be a number between 0 and 1")
        exit(1)

    if argc < 4:
        no_train = True
    no_train = sys.argv[3] == "True" if argc > 3 else True

    if desired_model == "help":
        print(
            gb("USAGE:") + "\n" +
                "  run.py\n" +
                "  run.py <model_name>\n" +
                "  run.py <model_name> <confidence treshold=[0,1]>\n" +
                "  run.py <model_name> <confidence treshold=[0,1]> <no train=True|False>\n" +
                "  run.py help\n"
        )
        exit(0)

    while desired_model not in YOLO_AVAILABLE_MODELS:
        print_error("the specified model does not exist")
        print(*YOLO_AVAILABLE_MODELS, sep=", ")
        i = input(
            f"please specify an existing model from the list above (or '"+
                rb("q")+"' = quit | '"+
                rb("d")+f"' = default model '{YOLO_DEFAULT_MODEL}'): "
        )

        if i == 'q':
            print("aborting...")
            exit(0)

        if i == 'd':
            desired_model = YOLO_DEFAULT_MODEL
            break

        desired_model = i

    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator
    run_navigable_inference((YOLO_RUNS_REL_DIR / desired_model).resolve(), confidence_treshold, no_train)