import json

def parse_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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
