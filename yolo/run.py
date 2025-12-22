import os
import re
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

# --- CONFIGURATION ---
MODEL_PATH = "runs/segment/26classes_80-20/weights/best.pt"
INPUT_DIR = "data/images/train"
OUTPUT_DIR = "inferred"
CONF_THRESHOLD = 0.25
WINDOW_NAME = "Navigation"

palette_rgb = [[248,0,255], [77,74,121], [26,81,50], [57,52,46], [98,90,88], [132,126,135], [170,170,170], [39,59,118], [47,71,136], [54,83,153], [69,107,187], [84,131,221], [109,171,102], [87,153,230], [90,175,238], [210,5,5], [222,45,19], [228,65,26], [233,85,33], [236,95,37], [239,105,40], [244,125,47], [250,145,54], [255,164,60], [248,176,93], [255,255,255], [34,31,219]]
palette_bgr = [tuple(color[::-1]) for color in palette_rgb]

def numerical_sort_key(path):
    numbers = re.findall(r'\d+', path.name)
    return int(numbers[0]) if numbers else path.name

def run_navigable_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    
    input_path = Path(INPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions], key=numerical_sort_key)
    
    session_cache = {}
    idx = 0
    num_images = len(image_files)

    while 0 <= idx < num_images:
        img_path = image_files[idx]
        img_name = img_path.name
        
        if img_name in session_cache:
            combined_img = session_cache[img_name]
        else:
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                idx += 1; continue

            results = model.predict(source=str(img_path), imgsz=640, conf=CONF_THRESHOLD, device=0, verbose=False)
            
            for r in results:
                inf_time = r.speed['inference']
                annotator = Annotator(original_img.copy(), line_width=2)
                
                if r.masks is not None:
                    masks = r.masks.data.cpu().numpy() 
                    clss = r.boxes.cls.cpu().numpy()
                    color_list = [palette_bgr[int(c)] for c in clss]
                    annotator.masks(masks, colors=color_list, alpha=0.5)
                    
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = f"{model.names[cls]} {box.conf[0]:.2f}"
                        bg_color = palette_bgr[cls]
                        
                        # Logic for Class 25 (White Background)
                        # text_color is (B, G, R). For white background, we use black text.
                        text_color = (0, 0, 0) if cls == 25 else (255, 255, 255)
                        
                        annotator.box_label(box.xyxy[0], label, color=bg_color, txt_color=text_color)

                inferred_img = annotator.result()
                h, w, _ = inferred_img.shape
                original_resized = cv2.resize(original_img, (w, h))
                combined_img = cv2.hconcat([original_resized, inferred_img])
                
                # Top-left info text (always green/shadow)
                info_text = f"{idx+1}/{num_images} {inf_time:.1f}ms"
                pos, font, scale, thickness = (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
                cv2.putText(combined_img, info_text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(combined_img, info_text, pos, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"nav_{img_name}"), combined_img)
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
    run_navigable_inference()