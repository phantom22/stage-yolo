from pathlib import Path
import os
import sys
import shutil

FILE_PATH = Path(__file__).resolve()
LABELS_ABS_DIR = FILE_PATH.parent
os.chdir(LABELS_ABS_DIR)

def print_usage():
    print(f"\033[38;5;120m\033[1mUSAGE:\033[0m\n  fix_specific_labels.py <input_dir> <output_dir>\n  fix_specific_labels.py help")

REMOVED_CLASS = 8 # alluminum can

if __name__ == "__main__":
    nargs = len(sys.argv)

    if nargs == 2 and sys.argv[1] != "help" or nargs < 3:
        print_usage()
        exit(1)

    input_dir = LABELS_ABS_DIR / sys.argv[1]
    output_dir = LABELS_ABS_DIR / sys.argv[2]

    if not input_dir.is_dir():
        print(f"\033[38;5;210m\033[1mERROR:\033[0m the specified input folder '{sys.argv[1]}' does not exist in '{LABELS_ABS_DIR}'")
        exit(1)

    if output_dir.is_dir():
        print(f"\033[93m\033[1mWARNING:\033[0m the specified output folder '{sys.argv[2]}' already exists")
        if input("do you want to override it? (y/n): ") == "y":
            shutil.rmtree(output_dir)
        else:
            print("aborting...")
            exit(0)

    output_dir.mkdir()

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
            if line_class_id == REMOVED_CLASS:
                continue
            elif line_class_id > REMOVED_CLASS:
                parts[0] = str(line_class_id - 1)
                output_lines.append(" ".join(parts) + "\n")
            else:
                output_class_id = line_class_id
                output_lines.append(line)

        with open(output_file_path, "w") as f:
            f.writelines(output_lines)

            