import os
import sys
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
YOLO_DIR = FILE_PATH.parent
os.chdir(YOLO_DIR)

SRC_DIR = YOLO_DIR.parent

# --- TO ALLOW IMPORTS FROM src/ (parent folder)

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# --- TRAIN CONFIGURATION ---

YOLO_DATA_DIR = YOLO_DIR / "data"
YOLO_IMG_DIR = YOLO_DATA_DIR / "images/train"
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')
MAIN_YAML_FILE = YOLO_DATA_DIR / "data.yaml"
YOLO_REL_RUNS_DIR = Path("runs/segment")
YOLO_RUNS_DIR = YOLO_DIR / YOLO_REL_RUNS_DIR
RUNS_PREFIX = "train"
YOLO_SPLIT_RATIO = 0.8

# --- RUN CONFIGURATION ---
AVAILABLE_MODELS = [f.name for f in YOLO_RUNS_DIR.iterdir() if f.is_dir()]
DEFAULT_MODEL = AVAILABLE_MODELS[0] if len(AVAILABLE_MODELS) > 0 else None
YOLO_RUN_CACHE_DIR = YOLO_DIR / "cache"
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
    'YOLO_DIR',
    'SRC_DIR',
    'YOLO_DATA_DIR',
    'YOLO_IMG_DIR',
    'IMG_EXTENSIONS',
    'MAIN_YAML_FILE',
    'YOLO_REL_RUNS_DIR',
    'YOLO_RUNS_DIR',
    'RUNS_PREFIX',
    'YOLO_SPLIT_RATIO',

    'AVAILABLE_MODELS',
    'DEFAULT_MODEL',
    'YOLO_RUN_CACHE_DIR',
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