import os
import sys
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
YOLO_ABS_DIR = FILE_PATH.parent
os.chdir(YOLO_ABS_DIR)

SRC_ABS_DIR = YOLO_ABS_DIR.parent
DATASET_IMG_ABS_DIR = SRC_ABS_DIR / "dataset/images"

# --- TO ALLOW IMPORTS FROM src/ (parent folder)

if str(SRC_ABS_DIR) not in sys.path:
    sys.path.append(str(SRC_ABS_DIR))

# --- TRAIN CONFIGURATION ---

abs_yolo_data_dir = YOLO_ABS_DIR / "data"
YOLO_IMAGES_REL_DIR = Path("data/images")
YOLO_LABELS_REL_DIR = Path("data/labels")

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')
YOLO_STRATEGIES_ABS_DIR = abs_yolo_data_dir / "strategies"
YOLO_RUNS_REL_DIR = Path("runs/segment")
YOLO_RUNS_ABS_DIR = YOLO_ABS_DIR / YOLO_RUNS_REL_DIR
YOLO_RUN_NAME_PREFIX = "train"
YOLO_TRAIN_TEST_SPLIT_RATIO = 0.8

YOLO_AVAILABLE_TRAIN_LABELS = [f.stem for f in (abs_yolo_data_dir / "labels").iterdir() if f.is_dir()]
YOLO_AVAILABLE_TRAIN_STRATEGIES = [f.stem for f in YOLO_STRATEGIES_ABS_DIR.iterdir() if f.is_file() and f.name.lower().endswith('.yaml')]
YOLO_DEFAULT_STRATEGY = YOLO_AVAILABLE_TRAIN_STRATEGIES[0] if len(YOLO_AVAILABLE_TRAIN_STRATEGIES) > 0 else None

YOLO_ZIP_NAME = Path("full_data.zip")

# --- RUN CONFIGURATION ---

YOLO_AVAILABLE_MODELS = [f.name for f in YOLO_RUNS_ABS_DIR.iterdir() if f.is_dir() and (f/"weights/best.pt").is_file()]
YOLO_DEFAULT_MODEL = YOLO_AVAILABLE_MODELS[0] if len(YOLO_AVAILABLE_MODELS) > 0 else None
YOLO_CACHE_ABS_DIR = YOLO_ABS_DIR / "cache"
YOLO_CONF_THRESHOLD = 0.6
YOLO_TARGET_SIZE = (640, 480)  # Width, Height

def yb(v):
    """yellow-bold"""
    return f"\033[93m\033[1m{v}\033[0m"

def rb(v):
    """red-bold"""
    return f"\033[38;5;210m\033[1m{v}\033[0m"

def gi(v):
    """gray-italics"""
    return f"\033[38;5;242m\033[1;3m{v}\033[0m"

def gb(v):
    """green-bold"""
    return f"\033[38;5;120m\033[1m{v}\033[0m"

def print_error(msg):
    print(rb('ERROR:') + " " + msg)

def print_warning(msg):
    print(yb('WARNING:') + " " + msg)

def print_fs(msg):
    print(gi(f'FILE-SYSTEM: {msg}'))

def print_hint(msg):
    print(gb('HINT:') + " " + msg)

__all__ = [
    'YOLO_IMAGES_REL_DIR',
    'YOLO_LABELS_REL_DIR',

    'YOLO_ABS_DIR',
    'SRC_ABS_DIR',
    'DATASET_IMG_ABS_DIR',
    'IMG_EXTENSIONS',
    'YOLO_STRATEGIES_ABS_DIR',
    'YOLO_RUNS_REL_DIR',
    'YOLO_RUNS_ABS_DIR',
    'YOLO_RUN_NAME_PREFIX',
    'YOLO_TRAIN_TEST_SPLIT_RATIO',

    'YOLO_DEFAULT_STRATEGY',
    'YOLO_AVAILABLE_TRAIN_LABELS',
    'YOLO_AVAILABLE_TRAIN_STRATEGIES',
    'YOLO_ZIP_NAME',

    'YOLO_AVAILABLE_MODELS',
    'YOLO_DEFAULT_MODEL',
    'YOLO_CACHE_ABS_DIR',
    'YOLO_CONF_THRESHOLD',
    'YOLO_TARGET_SIZE',

    'yb',
    'rb',
    'gi',
    'gb',
    'print_error',
    'print_warning',
    'print_fs',
    'print_hint'
]