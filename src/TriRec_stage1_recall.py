
import math
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import sys
import re
from collections import Counter
from datetime import datetime
from tqdm import tqdm

from config import (
    candidate_num,
    model,
    prompt_strategy,
    evaluation_times,
    inter_data_source,
    item_data_source,
    DOMAIN,
    get_experiment_id,
    num_users_to_sample,
    MEMORY_ROOT,
    LOG_ROOT,
    BASELINES_ROOT,
    DATASET_ROOT,
    createInterDF,
    createItemDF,
)
from request import parallel_get_responses, MAX_WORKERS
from fuzzywuzzy import fuzz

# Import SASRec-related modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'SASRec'))
from sasrec.model import SASREC
from sasrec.util import SASRecDataSet


def calculate_ndcg(relevance, k):
    """Compute NDCG@k, where relevance is a 0/1 list given in ranking order."""
    rel = relevance[:k]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


import tensorflow as tf

# Experiment configuration
exp_name = f"SASRec {DOMAIN}_{num_users_to_sample}"
memory_exp_name = f"{DOMAIN}_{num_users_to_sample}"
exp_id = get_experiment_id()
mode = "test"

# Parallel/batch parameters
API_BATCH = 32

# Candidate pool = 1 GT + N_NEG, defaults to 9; adjustable via the CAND_NUM environment variable
N_NEG = int(os.environ.get("CAND_NUM", 10)) - 1

# Log directory
log_dir = LOG_ROOT / f"exp_{exp_id}"
log_dir.mkdir(parents=True, exist_ok=True)

log_file_path = str(log_dir / "recommendation_process.jsonl")
first_stage_jsonl_path = str(log_dir / "first_stage_recommendations.jsonl")
two_stage_recommendations_path = str(log_dir / "two_stage_recommendations.jsonl")
recommendation_details_path = str(log_dir / "recommendation_details.jsonl")

# Clear log files
open(log_file_path, "w", encoding="utf-8").close()
open(recommendation_details_path, "w", encoding="utf-8").close()


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types to avoid JSON serialization issues."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif obj is None:
        return None
    else:
        return obj


def load_sasrec_model(model_path, maps_path, data_path, max_len=80, hidden_units=8):
    """Load a trained SASRec model."""
    print(f"Loading SASRec model from {model_path}")
    
    # 1. Read user & item maps
    with open(maps_path, 'rb') as f:
        user_map, item_map = pickle.load(f)
    print(f"Loaded user_map: {len(user_map)} users, item_map: {len(item_map)} items")

    # 2. Rebuild the model structure
    data = SASRecDataSet(data_path)  # use the correct path
    data.split()  # split the data
    
    model = SASREC(
        item_num=data.itemnum,
        seq_max_len=max_len,
        num_blocks=1,
        embedding_dim=hidden_units,
        attention_dim=hidden_units,
        attention_num_heads=1,
        dropout_rate=0.4,
        conv_dims=[hidden_units, hidden_units],
        l2_reg=0.00001
    )

    # build the model
    model.build((None, max_len))

    # 3. Load weights
    model.load_weights(model_path)
    print("Model loaded successfully ✅")

    # 4. Sample validation users
    print("Sampling validation users...")
    model.sample_val_users(data, 100)
    encoded_users = model.val_users
    print(f"Sampled {len(encoded_users) if encoded_users is not None else 0} validation users")

    return model, data, user_map, item_map

def item_agent_prompt(item_title, item_description, user_description): 
    """Generate the prompt for the item advertising agent."""
    return f"""You are a product promotion agent. Your task is to create a short but appealing ad copy for the following product, highlighting its strengths and features.

Product Title: {item_title}
Product Description: {item_description}

Target User Preferences: {user_description}

Please create an ad copy of no more than 50 words, focusing on how this product meets the user's preferences. 
Avoid exaggerated marketing language, and base the ad on the actual features of the product and the specific preferences of the user.

Output format: Only output the ad copy itself, without any additional explanation or formatting."""


def user_agent_prompt(user_description, item_ads_list):
    items_text = "\n".join([
        f"{i+1}. ID: {item['id']} | Title: {item['title']} | Ad: {item['ad']}"
        for i, item in enumerate(item_ads_list)
    ])

    return f"""You are an Amazon shopper with the following preferences and dislikes:

{user_description}

Here are several candidate items (each includes ID, Title, and Ad Copy):

{items_text}

Your task:
1. Rank **all** of them according to your preferences — **use every ID exactly once** (no omission, no repetition).
2. Rank them from most to least preferred.
3. Give each item a **Relevance Score** between **0 and 10**, using a **nonlinear, human-like scale**:
   - 9–10: Perfectly fits your preferences, you would almost certainly buy it.
   - 0: Completely irrelevant or opposite to your taste.
4. The score distribution should be **nonlinear**:
   - Only a few items should get scores above 8.
   - A few should be clearly low (0–2).
5. Output the result in **strict JSON** format only, no extra text or explanation.

JSON output format (very important):
{{
  "scores": [
    {{"id": "ITEM_ID_1", "score": FLOAT}},
    {{"id": "ITEM_ID_2", "score": FLOAT}},
    ...
  ],
  "reason": "Brief explanation why you gave these scores"
}}

Rules:
- Use every ID exactly once.
- Use only the IDs listed above (do not invent or modify any).
- The list must contain exactly {len(item_ads_list)} IDs.
- The 'score' field must be numeric (0–10).
- Do not output markdown or commentary outside the JSON.
- The output will be automatically validated — if any duplication or missing ID is found, your response will be considered invalid."""



if __name__ == "__main__":
    # Log run information
    start_time = datetime.now()
    exp_id = get_experiment_id()
    execution_log_path = str(log_dir / "execution_info.txt")
    memory_folder_name = f"{DOMAIN}_{num_users_to_sample}"
    with open(execution_log_path, "w", encoding="utf-8") as f:
        f.write("File Name: TriRec_stage1_recall.py\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Memory Folder: {memory_folder_name}\n")

    # Path setup
    domain = DOMAIN
    save_dir = BASELINES_ROOT / "SASRec" / f"{domain}_{num_users_to_sample}"
    maps_path = str(save_dir / "maps.pkl")
    data_path = str(save_dir / "sasrec_data.txt")
    weight_path = str(save_dir / "exp_example" / "exp_example_weights.weights.h5")
    
    # Check that files exist
    if not os.path.exists(maps_path):
        print(f"Error: mapping file {maps_path} not found")
        print("Please run the SASRec training script first to generate the mapping file")
        exit(1)
    
    if not os.path.exists(weight_path):
        print(f"Error: model weight file {weight_path} not found")
        print("Please run the SASRec training script first to generate the model weights")
        exit(1)

    # Load the SASRec model
    print("Loading SASRec model...")
    sasrec_model, sasrec_data, user_map, item_map = load_sasrec_model(weight_path, maps_path, data_path)
    
    # Build the dataset
    print("Building dataset...")
    interDF = createInterDF(inter_data_source(mode))
    itemDF = createItemDF(item_data_source)

    # Normalize ID types to strings to avoid type mismatches caused by numeric IDs (e.g. Steam)
    itemDF["parent_asin"] = itemDF["parent_asin"].astype(str)
    interDF["parent_asin"] = interDF["parent_asin"].astype(str)
    interDF["user_id"] = interDF["user_id"].astype(str)
    item_map = {str(k): v for k, v in item_map.items()}
    user_map = {str(k): v for k, v in user_map.items()}

    # Get the global item pool, but must be restricted to items existing in item_map (SASRec can only predict seen items)
    all_item_ids = [iid for iid in itemDF["parent_asin"].unique().tolist() if iid in item_map]
    print(f"Global item pool size: {len(all_item_ids)}")

    # --- Generate/load random.csv (SASRec recalls 100) ---
    random_csv_path = str(
        DATASET_ROOT / "user_item_data" / f"{domain}_{num_users_to_sample}" / "random.csv"
    )
    if not os.path.exists(random_csv_path):
        print(f"Generating recalled items to {random_csv_path}...")
        os.makedirs(os.path.dirname(random_csv_path), exist_ok=True)
        all_users = interDF["user_id"].unique().tolist()
        with open(random_csv_path, "w", encoding="utf-8") as f:
            for i in tqdm(range(0, len(all_users), 10), desc="Generating Recall"):
                batch_users = all_users[i:i+10]
                scores_df = sasrec_model.get_user_item_score(
                    sasrec_data, batch_users, all_item_ids, user_map, item_map, batch_size=len(batch_users)
                )
                for uid in batch_users:
                    # scores_df is a wide table: user_id, item_id1, item_id2, ...
                    user_row = scores_df[scores_df['user_id'] == uid].iloc[0]
                    # Drop user_id column; the rest are item_id: score
                    item_scores = user_row.drop('user_id')
                    # Sort and take the top 100 item IDs
                    top_100 = item_scores.sort_values(ascending=False).head(max(100, N_NEG + 1)).index.tolist()
                    f.write(f"{uid}\t{' '.join([str(i) for i in top_100])}\n")
        print("Generation complete.")
    
    user_to_recalled_items = {}
    with open(random_csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                user_to_recalled_items[parts[0]] = parts[1].split(" ")
    # ----------------------------------------------------

    print("SASRec model loaded; starting recommendation evaluation...")

    # Evaluation metrics
    ndcg_10_list_before = []
    ndcg_5_list_before = []
    ndcg_1_list_before = []
    mrr_list_before = []
    
    ndcg_10_list_after = []
    ndcg_5_list_after = []
    ndcg_1_list_after = []
    mrr_list_after = []
    
    # Store two-stage recommendation results
    two_stage_recommendations = []
    
    # Process interactions in batches
    all_inter_num = len(interDF)
    for start in range(0, all_inter_num, API_BATCH):
        end = min(start + API_BATCH, all_inter_num)
        batch = interDF.iloc[start:end]
        
        batch_data = []
        
        # 1. Prepare batch data
        user_id_list = batch["user_id"].tolist()
        
        # --- Optimization: fetch SASRec scores for the entire batch at once ---
        # batch_size here can be set to API_BATCH (10)
        all_scores_df = sasrec_model.get_user_item_score(
            sasrec_data, user_id_list, all_item_ids, user_map, item_map, batch_size=len(user_id_list)
        )
        # Convert results to a dict for easy lookup: {userId: {itemId: score}}
        scores_dict = all_scores_df.set_index('user_id').to_dict('index')

        for index, record in batch.iterrows():
            try:
                target_itemId = str(record["parent_asin"])
                userId = str(record["user_id"])
                target_item_title = str(itemDF[itemDF["parent_asin"] == target_itemId]["title"].values[0])
                main_kind = itemDF[itemDF["parent_asin"] == target_itemId]["main_category"].values[0]

                # Read user memory (single domain)
                one_domain_user_path = str(MEMORY_ROOT / memory_exp_name / "user" / f"user.{userId}")
                with open(one_domain_user_path, "r", encoding="utf-8") as file:
                    user_description = file.read()

                # Stage 1: fetch 100 candidates from the pre-generated random.csv, and sample 1 GT + 9 Random
                recalled_100 = user_to_recalled_items.get(userId, [])
                if not recalled_100:
                    # If not found, fall back to real-time SASRec computation
                    user_scores = scores_dict[userId]
                    scores = [(iid, user_scores[iid]) if iid in user_scores else (iid, 0) for iid in all_item_ids]
                    scores.sort(key=lambda x: x[1], reverse=True)
                    recalled_100 = [iid for iid, _ in scores[:100]]
                
                # 1 GT + N_NEG Random (N_NEG controlled by env var, default 9)
                recalled_100 = [str(x) for x in recalled_100]
                others = [iid for iid in recalled_100 if iid != target_itemId]
                sampled = random.sample(others, min(N_NEG, len(others)))
                top_k_items = [target_itemId] + sampled
                random.shuffle(top_k_items) # shuffle to obtain list0
                
                batch_data.append({
                    'userId': userId,
                    'target_itemId': target_itemId,
                    'target_item_title': target_item_title,
                    'main_kind': main_kind,
                    'user_description': user_description,
                    'top_k_items': top_k_items,
                    'index': index
                })
            except Exception as e:
                print(f"Error preparing batch data (Index {index}): {e}")
                continue

        if not batch_data:
            continue

        # 2. Multi-round evaluation
        for eval_round in range(evaluation_times):
            print(f"\n--- Batch {start//API_BATCH + 1}, eval round {eval_round + 1} ---")
            
            # A. Generate ads in parallel (Item Agent)
            ad_prompts = []
            ad_task_map = [] # record (user_idx, item_id)
            
            for i, d in enumerate(batch_data):
                for item_id in d['top_k_items']:
                    item_id = str(item_id)
                    try:
                        matched = itemDF[itemDF["parent_asin"] == item_id]["title"]
                        if matched.empty:
                            raise KeyError(f"item {item_id} not found in itemDF")
                        item_title = str(matched.values[0])
                        with open(str(MEMORY_ROOT / memory_exp_name / "item" / f"item.{item_id}"), "r", encoding="utf-8") as file:
                            item_description = file.read()
                        ad_prompts.append(item_agent_prompt(item_title, item_description, d['user_description']))
                        ad_task_map.append((i, item_id, item_title))
                    except Exception as e:
                        print(f"Failed to build ad prompt for {item_id}: {e}")

            ad_responses = parallel_get_responses(ad_prompts, model, max_workers=MAX_WORKERS)
            
            # Organize ad results
            for i in range(len(batch_data)):
                batch_data[i]['item_ads_list'] = []
            
            for resp, (user_idx, item_id, item_title) in zip(ad_responses, ad_task_map):
                if resp:
                    batch_data[user_idx]['item_ads_list'].append({
                        'id': item_id,
                        'title': item_title,
                        'ad': resp.strip()
                    })

            # B. User ranking in parallel (User Agent)
            user_prompts = []
            valid_user_indices = []
            
            for i, d in enumerate(batch_data):
                if d['item_ads_list']:
                    user_prompts.append(user_agent_prompt(d['user_description'], d['item_ads_list']))
                    valid_user_indices.append(i)

            user_responses = parallel_get_responses(user_prompts, model, max_workers=MAX_WORKERS)

            # C. Parse results and compute metrics
            for resp, user_idx in zip(user_responses, valid_user_indices):
                d = batch_data[user_idx]
                try:
                    # Stage 2 parsing: extract scores and ranking from JSON
                    final_item_ids, final_result_list, user_agent_scores = [], [], []
                    user_agent_reason = ""
                    if resp:
                        try:
                            # Try to match a JSON block
                            json_match = re.search(r'\{.*\}', resp, re.DOTALL)
                            if json_match:
                                data = json.loads(json_match.group())
                                scores_list = data.get("scores", [])
                                user_agent_reason = str(data.get("reason", ""))
                                
                                # Extract all valid candidate scores (compare as str to avoid numeric ID mismatch)
                                candidate_id_set = {str(it['id']) for it in d['item_ads_list']}
                                valid_scores = []
                                for item_info in scores_list:
                                    iid = str(item_info.get("id")).strip()
                                    score = float(item_info.get("score", 0))
                                    if iid in candidate_id_set:
                                        valid_scores.append({"id": iid, "score": score})
                                
                                # Sort by score descending as the final ranking
                                valid_scores.sort(key=lambda x: x["score"], reverse=True)
                                
                                for item_info in valid_scores:
                                    iid = item_info["id"]
                                    final_item_ids.append(iid)
                                    final_result_list.append(str(itemDF[itemDF["parent_asin"] == iid]["title"].values[0]))
                                    user_agent_scores.append(item_info)
                        except Exception as e:
                            print(f"JSON parsing failed for user {d['userId']}: {e}")
                    
                    # If parsing fails or result is empty, skip this record
                    if not final_item_ids:
                        print(f"User {d['userId']} result parsing failed or empty; skipping this sample.")
                        print(f"  item_ads_list size: {len(d['item_ads_list'])}")
                        print(f"  raw response (first 300 chars): {str(resp)[:300] if resp else 'None'}")
                        continue

                    # Stage 1 metrics (only computed when stage 2 succeeds, for a fair comparison)
                    result_list = [str(itemDF[itemDF["parent_asin"] == iid]["title"].values[0]) for iid in d['top_k_items']]
                    relevance_before = [1 if t.lower() == d['target_item_title'].lower() else 0 for t in result_list]
                    
                    if 1 in relevance_before:
                        rank_b = relevance_before.index(1) + 1
                        ndcg_10_b = calculate_ndcg(relevance_before, 10)
                        mrr_b = 1.0 / rank_b
                    else:
                        ndcg_10_b, mrr_b, rank_b = 0, 0, -1
                    
                    ndcg_10_list_before.append(ndcg_10_b)
                    mrr_list_before.append(mrr_b)

                    # Stage 2 metrics
                    relevance_after = [1 if t.lower() == d['target_item_title'].lower() else 0 for t in final_result_list]
                    if 1 in relevance_after:
                        rank_a = relevance_after.index(1) + 1
                        ndcg_10_a = calculate_ndcg(relevance_after, 10)
                        mrr_a = 1.0 / rank_a
                    else:
                        ndcg_10_a, mrr_a, rank_a = 0, 0, -1
                    
                    ndcg_10_list_after.append(ndcg_10_a)
                    mrr_list_after.append(mrr_a)

                    # Record the result
                    complete_rec = {
                        'user_id': d['userId'],
                        'target_item': d['target_itemId'],
                        'target_title': d['target_item_title'],
                        'original_candidates': d['top_k_items'],
                        'user_agent_scores': user_agent_scores,
                        'first_stage_metrics': {'ndcg_10': ndcg_10_b, 'mrr': mrr_b},
                        'second_stage_metrics': {'ndcg_10': ndcg_10_a, 'mrr': mrr_a},
                        'timestamp': datetime.now().isoformat()
                    }
                    two_stage_recommendations.append(complete_rec)
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(convert_numpy_types(complete_rec), ensure_ascii=False) + "\n")

                    # Record detailed recommendation info (user profile / target item details / all candidate ads / user agent reason)
                    try:
                        target_desc = ""
                        target_item_memory_path = str(MEMORY_ROOT / memory_exp_name / "item" / f"item.{d['target_itemId']}")
                        if os.path.isfile(target_item_memory_path):
                            with open(target_item_memory_path, "r", encoding="utf-8") as _tf:
                                target_desc = _tf.read()
                        detailed_rec = {
                            'user_id': d['userId'],
                            'user_description': d['user_description'],
                            'target_item': {
                                'id': d['target_itemId'],
                                'title': d['target_item_title'],
                                'description': target_desc,
                                'main_category': str(d['main_kind']),
                            },
                            'original_candidates': d['top_k_items'],
                            'candidate_ads': d['item_ads_list'],
                            'user_agent_scores': user_agent_scores,
                            'user_agent_reason': user_agent_reason,
                            'user_agent_raw_response': str(resp) if resp else "",
                            'final_ranked_ids': final_item_ids,
                            'final_ranked_titles': final_result_list,
                            'first_stage_metrics': {'ndcg_10': ndcg_10_b, 'mrr': mrr_b},
                            'second_stage_metrics': {'ndcg_10': ndcg_10_a, 'mrr': mrr_a},
                            'timestamp': datetime.now().isoformat(),
                        }
                        with open(recommendation_details_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(convert_numpy_types(detailed_rec), ensure_ascii=False) + "\n")
                    except Exception as _e:
                        print(f"Failed to write details for user={d['userId']}: {_e}")

                except Exception as e:
                    print(f"Error parsing result for user {d['userId']}: {e}")

        print(f"Current progress: {end}/{all_inter_num}")
        if ndcg_10_list_before:
            print(f"Stage 1 NDCG@10: {np.mean(ndcg_10_list_before):.4f}, MRR: {np.mean(mrr_list_before):.4f}")
            print(f"Stage 2 NDCG@10: {np.mean(ndcg_10_list_after):.4f}, MRR: {np.mean(mrr_list_after):.4f}")
        
        # Brief pause after each batch to avoid 502s caused by overly fast requests
        time.sleep(1)

    # Save final results
    with open(two_stage_recommendations_path, "w", encoding="utf-8") as f:
        for rec in two_stage_recommendations:
            f.write(json.dumps(convert_numpy_types(rec), ensure_ascii=False) + "\n")

    # Record end time
    end_time = datetime.now()
    with open(execution_log_path, "a", encoding="utf-8") as f:
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {end_time - start_time}\n")