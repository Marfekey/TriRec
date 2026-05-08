import json
import os
import sys
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import get_exposure_path

TRAIN_PATH = get_exposure_path("train")
SEMANTIC_PATH = get_exposure_path("test_semantic")
OUTPUT_PATH = get_exposure_path("overall")

def load_json_dict(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if not isinstance(obj, dict):
                raise ValueError(f"JSON content is not a dict: {path}")
            # Normalize keys to strings and values to float
            out = {}
            for k, v in obj.items():
                ks = str(k)
                try:
                    out[ks] = float(v)
                except Exception:
                    # If not numeric, keep the original value (overall values are expected to be numeric)
                    out[ks] = v
            return out
    except FileNotFoundError:
        print(f"[error] File not found: {path}")
        return {}
    except Exception as e:
        print(f"[error] Failed to read {path}: {e}")
        return {}

def merge_exposure(train: Dict[str, Any], semantic: Dict[str, Any]) -> Dict[str, Any]:
    train_keys = set(train.keys())
    sem_keys = set(semantic.keys())
    overlap_keys = train_keys & sem_keys
    train_only_keys = train_keys - sem_keys
    sem_only_keys = sem_keys - train_keys

    print(f"Overlapping item count: {len(overlap_keys)}")
    print(f"Train-only item count: {len(train_only_keys)}")
    print(f"Semantic-only item count: {len(sem_only_keys)}")
    print(f"Total merged item count: {len(train_keys | sem_keys)}")

    # Merge rule: for overlapping items use training value; for the rest use the value from their respective source
    combined = {}

    # All items from training (including overlapping and train-only)
    for k in train_keys:
        combined[k] = train[k]

    # Add semantic-only items
    for k in sem_only_keys:
        combined[k] = semantic[k]

    return combined

def save_json(path: str, obj: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    print(f"Overall exposure file written: {path} ({len(obj)} entries)")

def main():
    train = load_json_dict(TRAIN_PATH)
    semantic = load_json_dict(SEMANTIC_PATH)

    if not train and not semantic:
        print("[error] Both input files are empty or failed to load; aborting.")
        return

    combined = merge_exposure(train, semantic)
    save_json(OUTPUT_PATH, combined)

if __name__ == "__main__":
    main()