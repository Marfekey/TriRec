
import os
import sys
import gzip
import json
import ast
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import DATASET_ROOT, DOMAIN


DATA_DIR = DATASET_ROOT / "data"
ORG_INTER_DIR = DATASET_ROOT / "org_data" / "inter"

# domain -> (source gz filename, output csv filename)
DOMAIN_FILES = {
    "CDs": ("CDs_and_Vinyl.jsonl.gz", "CDs_and_Vinyl.csv"),
    "Movies_and_TV": ("Movies_and_TV.jsonl.gz", "Movies_and_TV.csv"),
    "goodreads_books_young_adult": (
        "goodreads_interactions_young_adult.json.gz",
        "goodreads_interactions_young_adult.csv",
    ),
    "steam_games": ("steam_reviews.json.gz", "steam_reviews.csv"),
}

# By default only process DOMAIN
FILES = [DOMAIN_FILES[DOMAIN]] if DOMAIN in DOMAIN_FILES else []


def trans():
    ORG_INTER_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, out_name in FILES:
        src = str(DATA_DIR / src_name)
        out_file = str(ORG_INTER_DIR / out_name)
        print(f"Processing {src}...")
        data = []
        with gzip.open(src, 'rt', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    if 'steam_reviews' in src:
                        data.append(ast.literal_eval(line))
                    else:
                        data.append(json.loads(line))
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue

        df = pd.json_normalize(data, sep='_')
        if 'images' in df.columns:
            df.drop('images', axis=1, inplace=True)

        df.to_csv(out_file, index=False, encoding='utf-8', escapechar='\\')
        print(f"Data successfully converted to {out_file}")


if __name__ == "__main__":
    trans()
