
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Project root and subdirectories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
MEMORY_ROOT = PROJECT_ROOT / "memory"
LOG_ROOT = PROJECT_ROOT / "log"
EXPOSURE_ROOT = PROJECT_ROOT / "exposure"
ANALYZE_ROOT = PROJECT_ROOT / "analyze"
LOG_VISUAL_ROOT = PROJECT_ROOT / "log_visual"
BASELINES_ROOT = PROJECT_ROOT / "baselines"

# ---------------------------------------------------------------------------
# 2. Experiment basic hyperparameters
# ---------------------------------------------------------------------------
DOMAIN = "CDs"
num_users_to_sample = 2000

domain_main_category_dict = {
    "goodreads_books_young_adult": "Goodreads_YA",
    "steam_games": "Steam_Games",
    "CDs": "CDs",
    "Movies_and_TV": "Movies & TV",
}

def get_main_kind(domain: str = DOMAIN) -> str:
    return domain_main_category_dict[domain]

# Model and recommendation related
candidate_num = 10
model = "gpt-4o-mini"
prompt_strategy = "B"
evaluation_times = 1

# ---------------------------------------------------------------------------
# 2.1 Stage-1 item promotion mode
# ---------------------------------------------------------------------------
#   full     : personalized self-promotion targeted at the user (default, main method)
#   grounded : promotion restricted to verifiable catalog attributes
#              (ablates factuality, keeps personalization)
#   generic  : non-personalized promotion, the user profile is replaced by a
#              constant (ablates the personalization condition)
#   none     : no promotion is generated; the raw item metadata text is used
#              (ablates the promotion mechanism itself, and skips the item-agent
#              LLM calls entirely)
PROMO_MODE = os.environ.get("PROMO_MODE", "full").strip().lower()
GENERIC_USER_DESC = (
    "a general audience with diverse tastes and no specific stated preferences"
)

def verifiable_attrs(row) -> str:
    """Build the "verifiable attributes" text used by ``PROMO_MODE=grounded``.

    Only objective catalog fields are taken; no LLM-generated content is
    included, so hallucinations cannot propagate from an earlier stage.
    """
    def g(key: str, default: str = "N/A") -> str:
        try:
            v = row.get(key)
            return default if v is None or str(v).strip() in ("", "nan", "None") else str(v).strip()
        except Exception:
            return default

    return (
        f"Categories: {g('categories')}\n"
        f"Store/Artist: {g('store')}\n"
        f"Subtitle: {g('subtitle')}\n"
        f"Price: {g('price')}\n"
        f"Average rating: {g('average_rating')} from {g('rating_number')} ratings"
    )

# ---------------------------------------------------------------------------
# 2.2 Promotion-feedback item memory update
# ---------------------------------------------------------------------------
# After each scoring round, an item can fold the user agent's feedback back into
# its own memory, so later rounds emphasize the angles that worked.  The signal
# is the relative tier assigned by the user agent only; no ground-truth label is
# used (the candidate set is 1 positive + N negatives, so using the label would
# leak the test signal).  Disabled by default: under the offline protocol the
# candidate pool is frozen and each item receives too few feedback events for the
# update to take effect.
MEMORY_UPDATE_ENABLED = os.environ.get("MEMORY_UPDATE", "0") == "1"
# Number of feedback entries buffered per item before one LLM integration fires.
MEMORY_UPDATE_BUFFER = int(os.environ.get("MEMORY_UPDATE_BUFFER", "3"))
# Integration results longer than this many words are treated as malformed and
# discarded, keeping the original memory.
MEMORY_UPDATE_MAX_WORDS = int(os.environ.get("MEMORY_UPDATE_MAX_WORDS", "80"))

# ---------------------------------------------------------------------------
# 3. Data paths (all based on DATASET_ROOT)
# ---------------------------------------------------------------------------
_USER_ITEM_ROOT = DATASET_ROOT / "user_item_data" / f"{DOMAIN}_{num_users_to_sample}"
_INITIAL_ROOT = DATASET_ROOT / "initial" / f"{DOMAIN}_{num_users_to_sample}"

def inter_data_source(mode: str) -> str:
    """Interaction data path: mode in {train, test, all}."""
    return str(_USER_ITEM_ROOT / "timesequence" / f"inter_timesequence_{mode}.csv")

item_data_source = str(_USER_ITEM_ROOT / "meta.csv")
random_source = str(_USER_ITEM_ROOT / "random" / f"random_{DOMAIN}.csv")

# ---------------------------------------------------------------------------
# 4. Experiment ID / memory directory / log directory / exposure directory
# ---------------------------------------------------------------------------
def get_experiment_id() -> str:
    """Read from the EXPERIMENT_ID environment variable first; fall back to a timestamp otherwise."""
    return os.getenv("EXPERIMENT_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))

def get_memory_dir_absolute(exp_name: str, exp_id: str | None = None) -> str:
    exp_id = exp_id or get_experiment_id()
    return str(MEMORY_ROOT / f"{exp_name}_{exp_id}")

def get_memory_dir_relative(exp_name: str, exp_id: str | None = None) -> str:
    exp_id = exp_id or get_experiment_id()
    return f"./memory/{exp_name}_{exp_id}"

def get_log_dir(exp_id: str | None = None) -> str:
    exp_id = exp_id or get_experiment_id()
    return str(LOG_ROOT / f"exp_{exp_id}")

def get_exposure_path(name: str) -> str:
    """Exposure JSON path with domain and sampled-user-count suffix."""
    EXPOSURE_ROOT.mkdir(parents=True, exist_ok=True)
    return str(EXPOSURE_ROOT / f"item_exposure_{name}_{DOMAIN}_{num_users_to_sample}.json")

def get_similarity_log_path() -> str:
    """Top-K similarity JSONL log path for test items."""
    EXPOSURE_ROOT.mkdir(parents=True, exist_ok=True)
    return str(EXPOSURE_ROOT / f"item_similarity_test_top10_{DOMAIN}_{num_users_to_sample}.jsonl")

# ---------------------------------------------------------------------------
# 5. DataFrame loading (replaces the old dataPrepare module)
# ---------------------------------------------------------------------------
def createItemDF(file_path: str) -> pd.DataFrame:
    """Build the complete item information table."""
    return pd.read_csv(file_path, low_memory=False)

def createInterDF(file_path: str) -> pd.DataFrame:
    """Build the interaction dataset."""
    return pd.read_csv(file_path, low_memory=False)

def createRandomDF(file_path: str) -> pd.DataFrame:
    """Read random samples (keep IDs as strings to avoid losing leading zeros)."""
    return pd.read_csv(file_path, dtype=str)

# ---------------------------------------------------------------------------
# 6. Initialize user / item / user-long memory based on interDF
# ---------------------------------------------------------------------------
def prepare_data_from_interDF(mode: str = "train") -> None:
    """
    Build user/item initialization memory from interaction data.
    Output location: dataset/initial/{DOMAIN}_{num_users_to_sample}/{item,user,user-long}/
    """
    interDF = createInterDF(inter_data_source(mode))
    itemDF = createItemDF(item_data_source)

    # Collect items that appear in interactions, preserving order
    item_ids, seen = [], set()
    for v in interDF["parent_asin"].astype(str).tolist():
        if v not in seen:
            item_ids.append(v)
            seen.add(v)

    # Initialize item memory
    item_dir = _INITIAL_ROOT / "item"
    item_dir.mkdir(parents=True, exist_ok=True)
    for item_id in item_ids:
        row = itemDF[itemDF["parent_asin"].astype(str) == item_id]
        if row.empty:
            continue
        r = row.iloc[0]
        payload = (
            f"'main_category':{r.get('main_category', '')}, "
            f"'item_title': '{r.get('title', '')}', "
            f"'item_subtitle': '{r.get('subtitle', '')}', "
            f"'item_class': '{r.get('categories', '')}', "
            f"'item_price': '{r.get('price', '')}'"
        )
        (item_dir / f"item.{item_id}").write_text(payload, encoding="utf-8")

    # Initialize user memory
    user_dir = _INITIAL_ROOT / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    for uid in interDF["user_id"].astype(str).unique().tolist():
        (user_dir / f"user.{uid}").write_text(
            f"I enjoy {DOMAIN} very much.", encoding="utf-8"
        )

    # Copy user-long (clear it first if it exists)
    long_dir = _INITIAL_ROOT / "user-long"
    if long_dir.exists():
        shutil.rmtree(long_dir)
    shutil.copytree(user_dir, long_dir)
