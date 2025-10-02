import json
import csv
from pathlib import Path
from typing import List, Dict, Any

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Reads a .jsonl file and returns a list of dictionaries."""
    data = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Saves a list of dictionaries to a .jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def save_csv(data: List[Dict[str, Any]], path: Path) -> None:
    """Saves a list of dictionaries to a .csv file."""
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def append_csv(data: List[Dict[str, Any]], path: Path) -> None:
    """Appends a list of dictionaries to a .csv file."""
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        if is_new_file:
            writer.writeheader()
        writer.writerows(data)
