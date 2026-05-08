
import os
import sys
import json
import ast
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import DATASET_ROOT, DOMAIN


ORG_INTER_DIR = DATASET_ROOT / "org_data" / "inter"
ORG_META_DIR = DATASET_ROOT / "org_data" / "meta"
FILTERED_META_DIR = DATASET_ROOT / "filtered_data" / "meta"


# Raw file and field configuration for each domain.
# - inter_file: interaction source file (used to determine the set of item IDs to keep)
# - inter_id_field: column in the interaction table that represents the item ID
# - meta_file: raw meta file (jsonl)
# - meta_id_field: field in meta that represents the item ID (uniformly mapped to parent_asin)
# - main_category: main_category label written into filtered_data
# - wanted_keys: final set of fields to keep
# - extra: optional extra per-record preprocessing function
DOMAIN_CONFIG = {
    "Books": {
        "inter_file": "Books.csv",
        "inter_id_field": "parent_asin",
        "meta_file": "meta_Books.jsonl",
        "meta_id_field": "parent_asin",
        "main_category": "Books",
        "wanted_keys": {"parent_asin", "title", "main_category", "subtitle", "author",
                         "average_rating", "rating_number", "price", "store", "categories"},
    },
    "Movies_and_TV": {
        "inter_file": "Movies_and_TV.csv",
        "inter_id_field": "parent_asin",
        "meta_file": "meta_Movies_and_TV.jsonl",
        "meta_id_field": "parent_asin",
        "main_category": "Movies & TV",
        "wanted_keys": {"parent_asin", "title", "main_category", "subtitle",
                         "average_rating", "rating_number", "price", "store", "categories"},
    },
    "CDs": {
        "inter_file": "CDs_and_Vinyl.csv",
        "inter_id_field": "parent_asin",
        "meta_file": "meta_CDs_and_Vinyl.jsonl",
        "meta_id_field": "parent_asin",
        "main_category": "Digital Music",
        "wanted_keys": {"parent_asin", "title", "main_category", "subtitle",
                         "average_rating", "rating_number", "price", "store", "categories"},
    },
    "goodreads_books_young_adult": {
        "inter_file": "goodreads_interactions_young_adult.csv",
        "inter_id_field": "book_id",
        "meta_file": "goodreads_books_young_adult.json",
        "meta_id_field": "book_id",
        "main_category": "Goodreads_YA",
        "wanted_keys": {"parent_asin", "title", "main_category", "average_rating",
                         "ratings_count", "description", "publisher"},
    },
    "steam_games": {
        "inter_file": "steam_reviews.csv",
        "inter_id_field": "product_id",
        "meta_file": "steam_games.json",
        "meta_id_field": "id",
        "main_category": "Steam_Games",
        "wanted_keys": {"parent_asin", "title", "main_category", "price",
                         "developer", "publisher", "categories"},
    },
}

FILTER_DOMAINS = [DOMAIN]


def _normalize_record(domain: str, record: dict) -> dict | None:
    """
    Normalize raw records from different domains into a unified field form, including:
    parent_asin, title, categories, etc. If the record does not meet basic conditions
    (missing title/main_category, etc.), return None.
    """
    cfg = DOMAIN_CONFIG[domain]
    id_field = cfg["meta_id_field"]

    # Unify the ID field
    raw_id = record.get(id_field)
    if raw_id is None:
        return None
    record['parent_asin'] = str(raw_id)

    # Common basic validation
    if domain in ("Books", "Movies_and_TV", "CDs"):
        title = record.get("title")
        main_category = record.get("main_category")
        if str(title) == "nan" or not title:
            return None
        if str(main_category) == "nan" or not main_category:
            return None

    # Domain-specific field adjustments
    if domain == "Books":
        author = record.get("author")
        if author and isinstance(author, dict) and author.get("name"):
            record["author"] = author["name"]
        else:
            record["author"] = None
        if record.get("categories"):
            record["categories"] = ";".join(record["categories"])

    elif domain in ("Movies_and_TV", "CDs"):
        if record.get("categories"):
            record["categories"] = ";".join(record["categories"])

    elif domain == "goodreads_books_young_adult":
        if not record.get("title"):
            return None
        authors = record.get("authors")
        if authors:
            record["author"] = authors[0].get("author_id")
        else:
            record["author"] = None

    elif domain == "steam_games":
        title = record.get("title") or record.get("app_name")
        if not title:
            return None
        record["title"] = title
        if record.get("genres"):
            record["categories"] = ";".join(record["genres"])

    return record


def _load_meta_lines(meta_path):
    """Compatible with jsonl / cases where some steam lines use ast literal."""
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                try:
                    yield ast.literal_eval(line)
                except Exception:
                    continue


def filter_domain_meta(domain: str):
    cfg = DOMAIN_CONFIG[domain]
    inter_path = ORG_INTER_DIR / cfg["inter_file"]
    meta_path = ORG_META_DIR / cfg["meta_file"]
    out_path = FILTERED_META_DIR / f"meta_{domain}.csv"
    FILTERED_META_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{domain}] Reading interactions: {inter_path}")
    inter_df = pd.read_csv(inter_path, encoding="utf-8")
    id_set = set(inter_df[cfg["inter_id_field"]].astype(str).unique())

    print(f"[{domain}] Reading meta: {meta_path}")
    rows = []
    for record in _load_meta_lines(meta_path):
        if record is None:
            continue
        raw_id = str(record.get(cfg["meta_id_field"]))
        if raw_id not in id_set:
            continue
        normalized = _normalize_record(domain, record)
        if not normalized:
            continue
        filtered = {k: normalized.get(k) for k in cfg["wanted_keys"]}
        rows.append(filtered)

    df = pd.DataFrame(rows)
    df["main_category"] = cfg["main_category"]
    df.to_csv(out_path, index=False)
    print(f"[{domain}] Saved {len(df)} meta records to: {out_path}")


def filter_meta_data():
    for domain in FILTER_DOMAINS:
        if domain not in DOMAIN_CONFIG:
            print(f"[Warn] domain '{domain}' is not registered in DOMAIN_CONFIG, skipping")
            continue
        filter_domain_meta(domain)


if __name__ == "__main__":
    filter_meta_data()
