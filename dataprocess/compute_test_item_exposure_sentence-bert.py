
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import (
    createInterDF,
    createItemDF,
    inter_data_source,
    item_data_source,
    get_exposure_path,
    get_similarity_log_path,
)

SBERT_MODEL = os.getenv("SBERT_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")


def _compose_item_text(row: pd.Series) -> str:
    """
    Compose the item text: title + subtitle + categories.
    Missing fields fall back to an empty string automatically.
    """
    title = str(row.get("title", "")).strip()
    subtitle = str(row.get("subtitle", "")).strip()
    categories = str(row.get("categories", "")).strip()
    return " ".join([title, subtitle, categories]).strip()


def _safe_load_json(path: str) -> dict:
    """
    Safely load a JSON file; return an empty dict if the file does not exist or fails to load.
    """
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"[Warn] Failed to load JSON {path}: {e}")
        return {}


def build_train_vectors(itemDF: pd.DataFrame, train_item_ids: list, device: str) -> tuple[list, SentenceTransformer, np.ndarray]:
    """
    Build sentence vectors for the training side using Sentence-BERT.
    Returns: list of training item IDs, SentenceTransformer model, and training vector matrix (normalized).
    """
    train_df = itemDF[itemDF["parent_asin"].isin(train_item_ids)].copy()
    if train_df.empty:
        print("[SemanticExposure] No items found in itemDF on the training side; vector construction failed")
        model = SentenceTransformer(SBERT_MODEL, device=device)
        return [], model, np.zeros((0, 0), dtype=float)

    corpus = train_df.apply(_compose_item_text, axis=1).tolist()
    training_ids = train_df["parent_asin"].tolist()

    # Load Sentence-BERT from the specified path/model name
    model = SentenceTransformer(SBERT_MODEL, device=device)
    train_matrix = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    return training_ids, model, train_matrix


def compute_semantic_exposure_for_test_items(
    itemDF: pd.DataFrame,
    test_item_ids: list,
    training_ids: list,
    train_matrix,
    model: SentenceTransformer,            
    train_exposure: dict,
    top_k: int,
    similarity_log_path: str,
) -> dict:
    """
    For each test item:
    - Compute semantic similarity with each training item
    - Take the Top-K most similar training items
    - Use normalized similarity weights to perform a weighted average (not weighted sum) to obtain the test item's "semantic exposure"
    - Normalization prevents the result from exceeding the upper bound of any training exposure
    Also write the similarity details (Top-K list) into a JSONL log file.
    """
    os.makedirs(os.path.dirname(similarity_log_path) or ".", exist_ok=True)
    exposures_result = {}

    item_map = itemDF.set_index("parent_asin")

    # If the training matrix is unavailable, return exposure 0 for everything
    if len(training_ids) == 0 or train_matrix.shape[0] == 0:
        print("[SemanticExposure] Training-side vectors are empty; all test item exposures set to 0")
        for tid in test_item_ids:
            exposures_result[tid] = 0.0
        return exposures_result

    for cid in test_item_ids:
        # Compose candidate item text + extract test item title
        if cid in item_map.index:
            row = item_map.loc[cid]
            cand_text = _compose_item_text(row)
            test_item_title = str(row.get("title", "")).strip() or "Unknown"
        else:
            cand_text = str(cid)
            test_item_title = "Unknown"

        try:
            # Encode the test item text with Sentence-BERT, keeping the same normalization as the training side
            cand_vec = model.encode([cand_text], convert_to_numpy=True, normalize_embeddings=True)
            sims = cosine_similarity(cand_vec, train_matrix).ravel()
        except Exception as e:
            print(f"[SemanticExposure] Failed to compute similarity for test item {cid}: {e}")
            exposures_result[cid] = 0.0
            continue

        if sims.size == 0:
            exposures_result[cid] = 0.0
            continue

        # Top-K most similar training items
        top_idx = np.argsort(-sims)[:top_k]
        top_entries = []
        weighted_sum = 0.0
        sim_sum = 0.0

        for i in top_idx:
            tid = training_ids[i]
            sim = float(sims[i])
            train_exp = float(train_exposure.get(tid, 0))
            # Extract training item title
            if tid in item_map.index:
                train_row = item_map.loc[tid]
                train_title = str(train_row.get("title", "")).strip() or "Unknown"
            else:
                train_title = "Unknown"

            weighted_sum += sim * train_exp
            sim_sum += sim
            top_entries.append({
                "train_item_id": tid,
                "train_item_title": train_title,  
                "similarity": sim,
                "train_exposure": train_exp,
                "norm_weight": None
            })

        # Normalize into a weighted average; set to 0 if the similarity sum is 0
        if sim_sum > 0:
            weighted_avg = float(weighted_sum / sim_sum)
            # Back-fill normalized weights into the log
            for j, i in enumerate(top_idx):
                top_entries[j]["norm_weight"] = float(sims[i] / sim_sum)
            exposures_result[cid] = weighted_avg
        else:
            exposures_result[cid] = 0.0

        # Append to the similarity log (JSONL, one line per test item's Top-K details)
        try:
            with open(similarity_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "test_item_id": cid,
                    "test_item_title": test_item_title,  
                    "weighted_exposure": exposures_result[cid],
                    "top_k": top_entries
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[SemanticExposure] Failed to write similarity log: {e}")

    return exposures_result



def parse_args():
    parser = argparse.ArgumentParser(description="Compute the semantic weighted exposure for each test item and save as JSON")
    parser.add_argument(
        "--train_mode", type=str, default="train",
        help="training data mode name (default: train)"
    )
    parser.add_argument(
        "--test_mode", type=str, default="test",
        help="test data mode name (default: test)"
    )
    parser.add_argument(
        "--train_exposure_json", type=str,
        default=get_exposure_path("train"),
        help="training data item exposure JSON path (used for weighting)"
    )
    parser.add_argument(
        "--output_exposure_json", type=str,
        default=get_exposure_path("test_semantic"),
        help="output test data item exposure JSON path"
    )
    parser.add_argument(
        "--output_similarity_jsonl", type=str,
        default=get_similarity_log_path(),
        help="output Top-10 similarity details (JSONL) path between test items and training items"
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="number of Top-K similar training items"
    )
    # Added: device selection (e.g. cuda:0, cuda:1 or cpu)
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="compute device: cuda:<idx> to specify a GPU, or cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device availability check: fall back to CPU if CUDA is unavailable
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Warn] CUDA is not available in the current environment (requested {device}); falling back to CPU")
        device = "cpu"

    # Load interaction and item information
    interDF_train = createInterDF(inter_data_source(args.train_mode))
    interDF_test = createInterDF(inter_data_source(args.test_mode))
    itemDF = createItemDF(item_data_source)

    # Unique sets of training/test items
    train_item_ids = sorted(set(interDF_train["parent_asin"].astype(str)))
    test_item_ids = sorted(set(interDF_test["parent_asin"].astype(str)))

    # Load training exposure (used only as a weight source, not directly for test items)
    train_exposure = _safe_load_json(args.train_exposure_json)
    if not train_exposure:
        print(f"[Warn] Failed to load training exposure file or it is empty: {args.train_exposure_json}; will use 0 as default exposure")

    # Build training-side Sentence-BERT sentence vectors (bound to the specified device)
    training_ids, model, train_matrix = build_train_vectors(itemDF, train_item_ids, device)
    print(f"[Info] Training-side vectors built: items={len(training_ids)}, dim={train_matrix.shape}, device={device}")

    # Clear or initialize the similarity log file
    try:
        with open(args.output_similarity_jsonl, "w", encoding="utf-8") as f:
            pass
    except Exception as e:
        print(f"[Warn] Failed to initialize the similarity log file: {e}")

    # Compute the semantic weighted exposure for test items (using Sentence-BERT similarity)
    exposures_result = compute_semantic_exposure_for_test_items(
        itemDF=itemDF,
        test_item_ids=test_item_ids,
        training_ids=training_ids,
        train_matrix=train_matrix,
        model=model,
        train_exposure=train_exposure,
        top_k=args.top_k,
        similarity_log_path=args.output_similarity_jsonl,
    )

    # Save test item exposure results
    os.makedirs(os.path.dirname(args.output_exposure_json) or ".", exist_ok=True)
    with open(args.output_exposure_json, "w", encoding="utf-8") as f:
        json.dump(exposures_result, f, ensure_ascii=False, indent=2)

    # Print summary
    zero_cnt = sum(1 for v in exposures_result.values() if v == 0)
    print(f"[Done] Number of test items: {len(exposures_result)} | Items with zero exposure: {zero_cnt}")
    print(f"[Path] Exposure JSON: {args.output_exposure_json}")
    print(f"[Path] Top-{args.top_k} similarity JSONL: {args.output_similarity_jsonl}")


if __name__ == "__main__":
    main()