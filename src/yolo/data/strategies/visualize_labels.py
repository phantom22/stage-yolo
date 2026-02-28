from pathlib import Path
import os
import sys
import yaml

import cv2
import numpy as np

STRATEGIES_ABS_DIR = Path(__file__).resolve().parent
LABELS_ABS_DIR = STRATEGIES_ABS_DIR.parent / "labels"
IMAGES_ABS_DIR = STRATEGIES_ABS_DIR.parents[2] / "dataset/images"
os.chdir(LABELS_ABS_DIR)

def sort_files(fname):
    return int(fname.stem)

def print_usage():
    print(f"\033[38;5;120m\033[1mUSAGE:\033[0m\n  visualize_labels.py <strategy>\n  visualize_labels.py help")

if __name__ == "__main__":
    nargs = len(sys.argv)

    if nargs != 2:
        print_usage()
        exit(1)

    strategy = sys.argv[1]
    if strategy == "help":
        print_usage()
        exit(0)

    strategy_yaml_path = STRATEGIES_ABS_DIR / f"{strategy}.yaml"
    if not strategy_yaml_path.is_file():
        print(f"\033[38;5;210m\033[1mERROR:\033[0m the strategy '{strategy}' does not exist in '{STRATEGIES_ABS_DIR}'")
        exit(1)

    with open(strategy_yaml_path, "r") as f:
        strategy_yaml = yaml.safe_load(f)

    if "labels" not in strategy_yaml:
        print(f"\033[38;5;210m\033[1mERROR:\033[0m the strategy '{strategy}' does not specify which labels were used")
        exit(1)

    if "names" not in strategy_yaml:
        print(f"\033[38;5;210m\033[1mERROR:\033[0m the strategy '{strategy}' does not have a names property")
        exit(1)

    names = strategy_yaml["names"]

    labels = strategy_yaml["labels"]
    labels_dir = LABELS_ABS_DIR / labels
    if not labels_dir.is_dir():
        print(f"\033[38;5;210m\033[1mERROR:\033[0m the strategy '{strategy}' points to '{labels}' labels, which do not exist in '{LABELS_ABS_DIR}'")
        exit(1)

    if "palette" not in strategy_yaml:
        print(f"\033[38;5;210m\033[1mERROR:\033[0m the strategy '{strategy}' does not have a palette")
        exit(1)

    palette_bgr = [tuple(color[::-1]) for color in strategy_yaml["palette"]]
    
    WINDOW_NAME = labels

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    images = [f for f in IMAGES_ABS_DIR.iterdir() if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    images.sort(key=sort_files)

    num_images = len(images)

    idx = 0

    while 0 <= idx < num_images:
        i = images[idx]
        im = cv2.imread(i)
        h, w, _ = im.shape

        with open(labels_dir / f"{i.stem}.txt", "r") as f:
            for line in f:
                parts = list(map(float, line.split()))

                cls = int(parts[0])
                class_color = palette_bgr[cls]
                class_name = names[cls]
                text_color = (255,255,255) if class_name != "disposable cutlery" else (0,0,0)

                coords = parts[1:]

                points = np.array(coords).reshape(-1, 2)
                points[:, 0] *= w  # Scale X
                points[:, 1] *= h  # Scale Y
                points = points.astype(np.int32)

                overlay = im.copy()
                cv2.fillPoly(overlay, [points], color=class_color)
                im = cv2.addWeighted(overlay, 0.4, im, 0.6, 0)

                cv2.polylines(im, [points], isClosed=True, color=class_color, thickness=1)

                # 1. Calculate the moments of the polygon
                M = cv2.moments(points)

                # 2. Calculate the center coordinates (cx, cy)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    # Fallback: if the polygon is extremely thin/degenerate, 
                    # use the first point as a backup
                    cx, cy = points[0][0], points[0][1]

                # 3. Draw the text at the center
                # Use a slight offset or centering adjustment if needed
                text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2

                cv2.putText(im, class_name, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            info_text = f"{idx+1}/{num_images} {i.name}"
            pos, font, scale, thickness = (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
            cv2.putText(im, info_text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(im, info_text, pos, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, im)
            
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