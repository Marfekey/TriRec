
import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from request import get_embedding_from_openai, parallel_get_embeddings
from config import item_data_source, DOMAIN, inter_data_source, num_users_to_sample, DATASET_ROOT

def _build_item_text(row: pd.Series) -> str:
    title = str(row.get('title', '')).strip()
    subtitle = str(row.get('subtitle', '')).strip()
    author = str(row.get('author', '')).strip()
    main_category = str(row.get('main_category', '')).strip()
    categories = str(row.get('categories', '')).strip()
    parts = [
        f"title: {title}" if title else "",
        f"subtitle: {subtitle}" if subtitle else "",
        f"author: {author}" if author else "",
        f"main_category: {main_category}" if main_category else "",
        f"categories: {categories}" if categories else "",
    ]
    return ". ".join([p for p in parts if p])

def _load_train_test_inter_sets():
        train_path = inter_data_source('train')
        test_path = inter_data_source('test')
        train_df = pd.read_csv(train_path, dtype=str, low_memory=False).fillna('')
        train_inter_df = train_df[['user_id', 'parent_asin']].copy()

        test_df = pd.DataFrame(columns=['user_id', 'parent_asin'])
        try:
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path, dtype=str, low_memory=False).fillna('')
        except Exception:
            pass

        inter_df = pd.concat(
            [train_inter_df, test_df[['user_id', 'parent_asin']]],
            ignore_index=True
        )
        items_set = set(inter_df['parent_asin'].astype(str))
        users_set = set(inter_df['user_id'].astype(str))
        train_users_set = set(train_inter_df['user_id'].astype(str))
        print(f"Items seen in interactions: {len(items_set)}, users: {len(users_set)}")
        print(f"Train-only users: {len(train_users_set)}")
        return items_set, users_set, inter_df, train_inter_df, train_users_set

def generate_item_embeddings(output_dir: str, model: str = "text-embedding-ada-002", restrict_items: set | None = None):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "item_embeddings.json")

    # Load item metadata
    item_df = pd.read_csv(item_data_source, dtype=str, low_memory=False).fillna('')

    # If a cache exists, load it first to support resuming
    item_embeddings = {}
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            try:
                item_embeddings = json.load(f)
            except Exception:
                item_embeddings = {}

    # Only keep items that appear in interactions, then deduplicate by parent_asin
    if restrict_items is not None and len(restrict_items) > 0:
        item_df = item_df[item_df['parent_asin'].astype(str).isin(restrict_items)]
    item_df = item_df.drop_duplicates(subset=['parent_asin'])

    # Collect items that need embedding generation
    to_generate_asins = []
    to_generate_texts = []
    for _, row in item_df.iterrows():
        asin = str(row['parent_asin'])
        if asin in item_embeddings:
            continue
        text = _build_item_text(row)
        if not text:
            continue
        to_generate_asins.append(asin)
        to_generate_texts.append(text)

    # Increase batch_size and raise the number of parallel workers
    batch_size = 100
    for i in tqdm(range(0, len(to_generate_texts), batch_size), desc="Generating item embeddings (parallel)"):
        batch_texts = to_generate_texts[i:i+batch_size]
        batch_asins = to_generate_asins[i:i+batch_size]
        # Raise max_workers to 15 or more (you have 18 keys)
        batch_embs = parallel_get_embeddings(batch_texts, model=model, max_workers=15)
        
        for asin, emb in zip(batch_asins, batch_embs):
            if emb is not None:
                item_embeddings[asin] = emb

    # Save
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(item_embeddings, f)
    print(f"Item embeddings saved to {out_path}")

def generate_user_embeddings(
    output_dir: str,
    inter_df: pd.DataFrame,
    restrict_users: set | None = None,
    output_filename: str = "user_embeddings.json",
    split_desc: str = "train/test"
):
    # Load item embeddings
    item_path = os.path.join(output_dir, "item_embeddings.json")
    with open(item_path, 'r', encoding='utf-8') as f:
        item_embeddings = json.load(f)

    # Aggregate sequence per user
    user_items = defaultdict(list)
    for _, row in tqdm(inter_df.iterrows(), total=inter_df.shape[0], desc=f"Collect user sequences ({split_desc})"):
        uid = str(row['user_id'])
        asin = str(row['parent_asin'])
        user_items[uid].append(asin)

    # User vector: average its history item embeddings (only within the restrict_users set)
    user_embeddings = {}
    users_iter = user_items.items()
    if restrict_users is not None:
        users_iter = ((uid, asins) for uid, asins in user_items.items() if uid in restrict_users)

    for uid, asin_list in tqdm(list(users_iter), desc=f"Computing user embeddings ({split_desc} only)"):
        vecs = [item_embeddings[a] for a in asin_list if a in item_embeddings]
        if not vecs:
            continue
        dim = len(vecs[0])
        avg = [0.0] * dim
        for v in vecs:
            for i in range(dim):
                avg[i] += v[i]
        avg = [x / len(vecs) for x in avg]
        user_embeddings[uid] = avg

    # Save
    out_path = os.path.join(output_dir, output_filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(user_embeddings, f)
    print(f"User embeddings saved to {out_path}")

if __name__ == "__main__":
    """
    Inputs:
    - DOMAIN, num_users_to_sample, item_data_source, inter_data_source in config.py.
    - OpenAI API Key (used via request.py).

    Outputs:
    - item_embeddings.json: mapping from item ID to a 1536-dim vector.
    - user_embeddings.json: mapping from user ID to the average of its interacted item vectors.
    - Storage path: dataset/user_item_data/{DOMAIN}_{num_users}/semantic_embeddings/
    """
    # Semantic vector output directory (consistent with the data directory in config.py)
    domain_folder = f"{DOMAIN}_{num_users_to_sample}"
    output_dir = str(DATASET_ROOT / "user_item_data" / domain_folder / "semantic_embeddings")
    
    print(f"Output directory: {output_dir}")
    # Read train/test interaction sets
    items_set, users_set, inter_df, train_inter_df, train_users_set = _load_train_test_inter_sets()
    # Generate embeddings only for items seen in interactions
    generate_item_embeddings(output_dir, model="text-embedding-ada-002", restrict_items=items_set)
    # Generate embeddings only for users seen in interactions (based on merged train/test interactions)
    generate_user_embeddings(
        output_dir,
        inter_df,
        restrict_users=users_set,
        output_filename="user_embeddings.json",
        split_desc="train/test"
    )
    # Generate user embeddings based on training-set interactions only
    generate_user_embeddings(
        output_dir,
        train_inter_df,
        restrict_users=train_users_set,
        output_filename="user_embeddings_train.json",
        split_desc="train"
    )