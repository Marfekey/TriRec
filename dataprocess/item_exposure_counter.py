import os
import sys
import json
from typing import Dict

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import createInterDF, inter_data_source, get_experiment_id, get_exposure_path


def compute_item_exposure(inter_df: pd.DataFrame) -> Dict[str, int]:
    """
    Count the exposure of each item (based on the parent_asin column).
    """
    if "parent_asin" not in inter_df.columns:
        raise KeyError("interDF is missing the 'parent_asin' column")
    return inter_df["parent_asin"].astype(str).value_counts().to_dict()


def save_exposure_json(exposures: Dict[str, int], mode: str = "train", exp_id: int = None) -> str:
    """
    Save exposure statistics as a JSON file.
    """
    out_path = get_exposure_path(mode)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(exposures, f, ensure_ascii=False, indent=2)
    return out_path


def run(mode: str = "train", exp_id: int = None) -> str:
    """
    Main pipeline: read interaction data -> count exposures -> save as JSON.
    """
    inter_df = createInterDF(inter_data_source(mode))
    exposures = compute_item_exposure(inter_df)
    return save_exposure_json(exposures, mode=mode, exp_id=exp_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count the exposure of each item in the training interaction file and save as JSON")
    parser.add_argument("--mode", type=str, default="train", help="dataset mode, e.g. 'train'")
    parser.add_argument("--exp_id", type=int, default=None, help="experiment ID, read from config by default")
    args = parser.parse_args()

    try:
        output = run(mode=args.mode, exp_id=args.exp_id)
        print(f"Saved item exposure statistics to: {output}")
    except Exception as e:
        print(f"Statistics failed: {e}")