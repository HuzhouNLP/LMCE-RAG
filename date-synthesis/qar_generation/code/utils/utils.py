import sys
import json
import logging
from typing import List, Dict

def read_prompt(file_path: str = 'prompts/finance_zh.jsonl') -> List[Dict]:
    """Read prompts from a JSONL-like file (one or more JSON objects)."""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)

    decoder = json.JSONDecoder()
    idx = 0
    length = len(content)

    while idx < length:
        # Skip whitespace between JSON objects
        while idx < length and content[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            obj, next_idx = decoder.raw_decode(content, idx)
            prompts.append(obj)
            idx = next_idx
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error while reading prompts: {e} at index {idx} in {file_path}")
            sys.exit(1)

    return prompts

def read_config_json(json_path: str) -> Dict:
    """Read a JSON configuration file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.decoder.JSONDecodeError as e:
        logging.error(f'JSON Decode Error: {e} in {json_path}')
        sys.exit(1)
    except FileNotFoundError:
        logging.error(f"File not found: {json_path}")
        sys.exit(1)
    return config

def write_config_json(json_path: str, config: Dict) -> None:
    """Write configuration data to a JSON file."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
