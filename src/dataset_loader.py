import json
from typing import List, Dict, Any

def load_evaluation_dataset(file_path: str = "data/dataset.json") -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)