from pathlib import Path
import os
import sys
import shutil

FILE_PATH = Path(__file__).resolve()
LABELS_ABS_DIR = FILE_PATH.parent
os.chdir(LABELS_ABS_DIR)

def print_usage():
    print(f"\033[38;5;120m\033[1mUSAGE:\033[0m\n  fix_labels.py <input_dir> <output_dir> <specific|generic>\n  fix_labels.py help")

ALLUMINUM_CAN_ID = 8 # alluminum can
FIRST_SPECIFIC_CLASS = 27 # coca-cola

if __name__ == "__main__":
    nargs = len(sys.argv)

    if nargs == 2 and sys.argv[1] == "help":
        print_usage()
        exit(0)

    if nargs < 4:
        print_usage()
        exit(1)

    input_dir = LABELS_ABS_DIR / sys.argv[1]
    output_dir = LABELS_ABS_DIR / sys.argv[2]
    mode = sys.argv[3]

    if mode not in ["specific","generic"]:
        print(f"\033[38;5;210m\033[1mERROR:\033[0m expected mode either 'generic' or 'specific', got '{mode}'")
        exit(1)

    if not input_dir.is_dir():
        print(f"\033[38;5;210m\033[1mERROR:\033[0m the specified input folder '{sys.argv[1]}' does not exist in '{LABELS_ABS_DIR}'")
        exit(1)

    if output_dir.is_dir():
        print(f"\033[93m\033[1mWARNING:\033[0m the specified output folder '{sys.argv[2]}' already exists")
        if input("do you want to use it? (y/n): ") != "y":
            print("aborting...")
            exit(0)

    output_dir.mkdir(exist_ok=True)

    for input_file in input_dir.iterdir():
        fname = input_file.name
        output_file_path = output_dir / fname

        with open(input_file, 'r') as f:
            input_lines = f.readlines()

        output_lines = []
        for line in input_lines:
            parts = line.split()
            if not parts:
                continue

            line_class_id = int(parts[0]) 
            if mode == "specific" and line_class_id == ALLUMINUM_CAN_ID:
                continue
            elif mode == "specific" and line_class_id > ALLUMINUM_CAN_ID:
                parts[0] = str(line_class_id - 1)
            elif mode == "generic" and line_class_id >= FIRST_SPECIFIC_CLASS:
                parts[0] = str(ALLUMINUM_CAN_ID)
            else:
                output_lines.append(line)
                continue

            output_lines.append(" ".join(parts) + "\n")

        with open(output_file_path, "w") as f:
            f.writelines(output_lines)

            