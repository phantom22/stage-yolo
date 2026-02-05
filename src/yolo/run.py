from yolo_config import *

import os
import re
import sys

from pathlib import Path

from ultralytics import YOLO

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

WINDOW_NAME = "Navigation"


palette_rgb = [[248,0,255], [77,74,121], [26,81,50], [57,52,46], [98,90,88], [132,126,135], [170,170,170], [39,59,118], [47,71,136], [54,83,153], [69,107,187], [84,131,221], [109,171,102], [87,153,230], [90,175,238], [210,5,5], [222,45,19], [228,65,26], [233,85,33], [236,95,37], [239,105,40], [244,125,47], [250,145,54], [255,164,60], [248,176,93], [255,255,255], [34,31,219] ]
palette_bgr = [tuple(color[::-1]) for color in palette_rgb]

def numerical_sort_key(path):
    return int(path.stem)

def run_navigable_inference(model_path):
    if not model_path.exists():
        print_error(f"Model not found at {model_path}")
        exit(1)

    print(f"running '{gb(model_path.parents[1].name)}' model")
    
    cache_dir_name = YOLO_CACHE_ABS_DIR.name
    if YOLO_CACHE_ABS_DIR.exists():
        for f in YOLO_CACHE_ABS_DIR.iterdir():
            f.unlink()
        print_fs(f"emptied 'yolo/{cache_dir_name}' contents")
    else:    
        YOLO_CACHE_ABS_DIR.mkdir()
        print_fs(f"created 'yolo/{cache_dir_name}' directory")

    model = YOLO(model_path)

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    image_files = sorted([f for f in DATASET_IMG_ABS_DIR.iterdir() if f.suffix.lower() in IMG_EXTENSIONS], key=numerical_sort_key)
    num_images = len(image_files)

    session_cache = {}
    idx = 0
    while 0 <= idx < num_images:
        img_path = image_files[idx]
        img_name = img_path.name
        
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
            results = model.predict(source=original_img, imgsz=640, conf=YOLO_CONF_THRESHOLD, device=0, verbose=False)
            
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
                        label = f"{model.names[cls]} {box.conf[0]:.2f}"
                        bg_color = palette_bgr[cls]
                        text_color = (0, 0, 0) if cls == 25 else (255, 255, 255)
                        
                        # Bounding boxes are now perfectly aligned with the 640x480 image
                        annotator.box_label(box.xyxy[0], label, color=bg_color, txt_color=text_color)

                inferred_img = annotator.result()
                
                # Side-by-side (both are now 640x480)
                combined_img = cv2.hconcat([original_img, inferred_img])
                
                # Small Green info text
                info_text = f"{idx+1}/{num_images} {inf_time:.1f}ms"
                pos, font, scale, thickness = (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
                cv2.putText(combined_img, info_text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(combined_img, info_text, pos, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
                
                cv2.imwrite(YOLO_CACHE_ABS_DIR / f"compare_{img_name}", combined_img)
                session_cache[img_name] = combined_img

        cv2.imshow(WINDOW_NAME, combined_img)
        
        raw_key = cv2.waitKeyEx(0)
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

    desired_model = sys.argv[1] if len(sys.argv) > 1 else YOLO_DEFAULT_MODEL

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

    run_navigable_inference((YOLO_RUNS_REL_DIR / desired_model / "weights/best.pt").resolve())