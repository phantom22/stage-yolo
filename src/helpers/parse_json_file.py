import json
import os

def parse_json_file(from_file,filepath):
    from_dir = os.path.dirname(from_file)
    json_path = os.path.join(from_dir, filepath)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in file: {e}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None