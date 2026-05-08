
import os
import sys
import re
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import DATASET_ROOT, DOMAIN


ORG_INTER_DIR = DATASET_ROOT / "org_data" / "inter"
FILTERED_META_DIR = DATASET_ROOT / "filtered_data" / "meta"
FILTERED_INTER_DIR = DATASET_ROOT / "filtered_data" / "inter"

# "Source file name" and "field mapping" for each domain. To add a new domain, just extend this dict.
DOMAIN_INFO = {
    "goodreads_books_young_adult": {
        "source": "goodreads_interactions_young_adult.csv",
        "mapping": {"item": "book_id", "user": "user_id", "rating": "rating"},
    },
    "steam_games": {
        "source": "steam_reviews.csv",
        "mapping": {"item": "product_id", "user": "user_id", "rating": "rating"},
    },
    "CDs": {
        "source": "CDs_and_Vinyl.csv",
        "mapping": {"item": "parent_asin", "user": "user_id", "rating": "rating"},
    },
    "Movies_and_TV": {
        "source": "Movies_and_TV.csv",
        "mapping": {"item": "parent_asin", "user": "user_id", "rating": "rating"},
    },
}

# By default only process DOMAIN; change to a list of multiple for batch processing
FILTER_DOMAINS = [DOMAIN]


def sanitize_id(id_str):
    """Ensure the ID is safe to use as a filename."""
    if pd.isna(id_str):
        return "unknown"
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', str(id_str))


def _postprocess(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Domain-specific post-processing (timestamp generation, user_id cleaning, etc.)."""
    if key == "steam_games":
        if 'username' in df.columns:
            df['username'] = df['username'].astype(str)
        if 'user_id' not in df.columns:
            df['user_id'] = df['username'] if 'username' in df.columns else "unknown"
        else:
            df['user_id'] = df['user_id'].fillna(df['username'] if 'username' in df.columns else "unknown")
        df['user_id'] = df['user_id'].apply(sanitize_id)
        if 'found_funny' in df.columns:
            df['rating'] = df['found_funny'].apply(lambda x: 5 if pd.notnull(x) and x > 0 else 0)
        else:
            df['rating'] = 0
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], utc=True).astype('int64') // 10**9
    if key == "goodreads_books_young_adult":
        if 'date_added' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date_added'], utc=True, format='mixed').astype('int64') // 10**9
    return df


def filter_inter_data():
    FILTERED_INTER_DIR.mkdir(parents=True, exist_ok=True)

    for key in FILTER_DOMAINS:
        if key not in DOMAIN_INFO:
            print(f"[Warn] domain '{key}' is not registered in DOMAIN_INFO, skipping")
            continue
        info = DOMAIN_INFO[key]
        inter_path = ORG_INTER_DIR / info["source"]
        meta_path = FILTERED_META_DIR / f"meta_{key}.csv"
        out_path = FILTERED_INTER_DIR / f"inter_{key}.csv"

        print(f"[{key}] Reading interactions: {inter_path}")
        df = pd.read_csv(inter_path)
        mapping = info["mapping"]
        df = df.rename(columns={mapping['item']: 'parent_asin', mapping['user']: 'user_id'})
        df = _postprocess(df, key)

        print(f"[{key}] Raw interaction count: {df.shape[0]}")

        print(f"[{key}] Reading meta: {meta_path}")
        meta_df = pd.read_csv(meta_path)
        parent_asin_set = set(meta_df['parent_asin'].unique())

        df = df[df['parent_asin'].isin(parent_asin_set)]
        print(f"[{key}] Filtered interaction count: {df.shape[0]}")

        df.to_csv(out_path, index=False)
        print(f"[{key}] Saved: {out_path}")


if __name__ == "__main__":
    filter_inter_data()
