import os
import sys
import zipfile
import shutil
from pathlib import Path

from yolo_config import *

RELATIVE_PATHS = [
    Path("weights/last.pt"),
    # Path("weights/best.pt"),
    Path("args.yaml"),
    Path("BoxF1_curve.png"),
    Path("BoxP_curve.png"),
    Path("BoxPR_curve.png"),
    Path("BoxR_curve.png"),
    # Path("confusion_matrix_normalized.png"),
    Path("confusion_matrix.png"),
    # Path("data.json"),
    Path("labels.jpg"),
    # Path("data.yaml"),
    Path("MaskF1_curve.png"),
    Path("MaskP_curve.png"),
    Path("MaskPR_curve.png"),
    Path("MaskR_curve.png"),
    Path("results.csv"),
    Path("results.png"),
    ## Path("train_batch*.jpg"),
    Path("train.txt"),
    Path("val.txt"),
    ## Path("val_batch*_labels.jpg"),
    ## Path("val_batch*_pred.jpg"),
    # Path("val.txt"),
]

PATTERNS = [
    "train_batch*.jpg",
    "val_batch*_labels.jpg",
    "val_batch*_pred.jpg"
]

def zip_run(run_dir_name):
    run_rel_path = YOLO_RUNS_REL_DIR / run_dir_name
    zip_file_path = run_rel_path / YOLO_ZIP_NAME

    if zip_file_path.is_file():
        print(f"\033[38;5;210m\033[1mError:\033[0m the specified run is already zipped.")
        exit(1)

    relative_paths = [run_rel_path / p for p in RELATIVE_PATHS]

    files_to_move = relative_paths
    for p in PATTERNS:
        files_to_move.extend(run_rel_path.glob(p))

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in relative_paths:
            if file_path.exists():
                zip_file.write(file_path, arcname=file_path.relative_to(run_rel_path))
                file_path.unlink()
    
    print_fs(f"zipped '{run_dir_name}' data.")

def unzip_run(run_dir_name):
    run_rel_path = YOLO_RUNS_REL_DIR / run_dir_name
    zip_file_path = run_rel_path / YOLO_ZIP_NAME

    if not zip_file_path.is_file():
        print_error("the specified run is already unzipped")
        exit(1)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(run_rel_path)

    zip_file_path.unlink()

    print_fs(f"unzipped '{run_dir_name}' data.")

__all__ = [
    'zip_run',
    'unzip_run'
]

def print_usage():
    print(
        gb("USAGE:") + "\n" +
            "  util.py zip <run-name>\n" +
            "  util.py unzip <run-name>\n" +
            "  util.py fix\n" +
            "  util.py zipall\n" +
            "  util.py list\n" +
            "  util.py help"
    )

if __name__ == "__main__":
    nargs = len(sys.argv)
    if nargs == 1:
        print_error("insufficient args")
        print_usage()
        exit(1)
    
    first = sys.argv[1]
    if first == "help":
        print_usage()
        exit(0)

    if nargs == 2:
        if first == "fix":
            for d in YOLO_RUNS_ABS_DIR.iterdir():
                run_name = d.name
                weights = d / "weights"
                best_pt = weights / "best.pt"
                print(yb(f"{run_name.upper()} FOLDER"))

                critical = False
                if not weights.is_dir():
                    critical = True
                    print(rb("███") + " " + gi(weights.name+"/") + " " + rb("CRITICAL"))
                else:
                    print(gb("███") + " " + gi(weights.name+"/"))

                if not best_pt.is_file():
                    if not critical:
                        print(rb("███") + " " + gi(best_pt.name) + " " + rb("CRITICAL"))
                    critical = True
                else:
                    print(gb("███") + " " + gi(best_pt.name))

                if critical:
                    print(yb("NOTE:") + f" '{run_name}' is not a run folder")
                    if input("delete the folder? (y/n): ") == "y":
                        try:
                            shutil.rmtree(d)
                            print_fs(f"succesfully removed 'runs/segment/{run_name}'")
                            print()
                        except Exception as e:
                            print_error(f"couldn't remove 'runs/segment/{run_name}', reason: {e}")
                            exit(1)
                    continue

                zip_file = d / YOLO_ZIP_NAME
                was_zipped = False
                if zip_file.is_file():
                    was_zipped = True
                    unzip_run(d.name)

                for f in RELATIVE_PATHS:
                    file_path = d / f
                    if not file_path.is_file():
                        print(yb("███") + " " + gi(f.name) + " " + yb("MISSING"))
                    else:
                        print(gb("███") + " " + gi(f.name))

                if was_zipped:
                    zip_run(d.name)
                
                print()
            exit(0)
        elif first == "list":
            print(gb("AVAILABLE RUNS") + ":")
            print(*YOLO_AVAILABLE_MODELS, sep="\n")
            exit(0)
        elif first == "zipall":
            for r in YOLO_AVAILABLE_MODELS:
                if not (YOLO_RUNS_ABS_DIR / r / YOLO_ZIP_NAME).is_file():
                    zip_run(r)
                else:
                    print_fs(f"run '{r}' is already zipped")
            exit(0)
        else:
            print_usage()
            exit(1)
    
    if nargs == 3:
        run_names = YOLO_AVAILABLE_MODELS
        second = sys.argv[2]
        
        if second not in run_names:
            print_error(f"the specified run name '{second}' does not exist")
            print_hint(f"here is a list of available runs: {','.join(YOLO_AVAILABLE_MODELS)}")
            exit(1)
        
        if first == "zip":
            zip_run(second)
        elif first == "unzip":
            unzip_run(second)
        else:
            print_usage()
            exit(1)