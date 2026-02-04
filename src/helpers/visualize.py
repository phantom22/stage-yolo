import cv2
from pathlib import Path
import re
import warnings

SRC_DIR = Path(__file__).resolve().parents[1]
WINDOW_NAME = "visualize"

def numerical_sort_key(path):
    numbers = re.findall(r'\d+', path.name)
    return int(numbers[0]) if numbers else path.name

def visualize(indices):
    if isinstance(indices, int):
        target_indices = [str(indices)]
        num_images = 1
    else:
        target_indices = [str(i) for i in indices]
        num_images = len(indices)

    if num_images == 0:
        warnings.warn("no images to visualize", UserWarning)
        # \033[93m is Yellow, \033[1m is Bold, \033[0m resets formatting
        print(f"\033[93m\033[1mWARNING:\033[0m no images to visualize.")
        return

    input_path = Path(SRC_DIR / "yolo/data/images/train")
    image_files = sorted([f for f in input_path.iterdir() if f.stem in target_indices], key=numerical_sort_key)

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    idx = 0

    while 0 <= idx < num_images:
        img_path = image_files[idx]
        img_name = img_path.name
        
        im = cv2.imread(str(img_path))

        info_text = f"{idx+1}/{num_images} {img_name}"
        pos, font, scale, thickness = (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
        cv2.putText(im, info_text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(im, info_text, pos, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, im)
        
        raw_key = cv2.waitKeyEx(0)
        if raw_key in [97, 81, 65361, 2424832]: # A or LEFT
            idx = max(0, idx - 1)
        elif raw_key in [100, 83, 65363, 2555904]: # D or RIGHT
            idx = min(num_images - 1, idx + 1)
        elif raw_key in [27, 113]: # ESC or Q
            break

    cv2.destroyAllWindows()