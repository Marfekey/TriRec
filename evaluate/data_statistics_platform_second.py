
import math
import json
import numpy as np
from collections import Counter, defaultdict
import re
import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import config
from config import get_exposure_path, ANALYZE_ROOT, DATASET_ROOT, DOMAIN, num_users_to_sample

import json
import numpy as np

# ======== Logging utilities ========

class Logger:
    """Logger that writes output to a txt file."""
    def __init__(self, log_file=None):
        if log_file is None:
            ANALYZE_ROOT.mkdir(parents=True, exist_ok=True)
            log_file = str(ANALYZE_ROOT / "data_statistics_log_second.txt")
        # If the file already exists, automatically find a new filename (appending _k suffix)
        if os.path.exists(log_file):
            base, ext = os.path.splitext(log_file)
            k = 1
            while os.path.exists(f"{base}_{k}{ext}"):
                k += 1
            log_file = f"{base}_{k}{ext}"
            
        self.log_file = log_file
        # Create the log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Data statistics analysis log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        """Record a message to the log file."""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_separator(self, char="=", length=80):
        """Record a separator line."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(char * length + "\n")

class NullLogger:
    def log(self, message):
        pass
    def log_separator(self, char="=", length=80):
        pass

# Global logger
# Enable file logging only when the script is run as the main program; use a null logger when imported to avoid overwriting files
if __name__ == "__main__":
    logger = Logger()
else:
    logger = NullLogger()

# ======== JSON serialization helpers ========

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ======== Basic functions ========

def dcg_at_k(r, k):
    """DCG@k"""
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    """NDCG@k"""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def mrr(r):
    """MRR"""
    for i, rel in enumerate(r):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0

# ======== Data loading functions ========

def load_recommendation_process_jsonl(file_path):
    """Load the recommendation-process data in JSONL format."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# ======== Evaluation functions ========
SEM_EMB_DIR = str(
    DATASET_ROOT / "user_item_data" / f"{DOMAIN}_{num_users_to_sample}" / "semantic_embeddings"
)
ITEM_EMB_PATH = os.path.join(SEM_EMB_DIR, "item_embeddings.json")
USER_EMB_PATH = os.path.join(SEM_EMB_DIR, "user_embeddings.json")

# Global cache to avoid repeated loading
_EMBEDDING_CACHE = {}
_EXPOSURE_CACHE = {}

def _safe_load_json(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.log(f"Failed to load JSON: {path} | {e}")
        return {}

def load_user_item_embeddings(user_path=USER_EMB_PATH, item_path=ITEM_EMB_PATH) -> tuple[dict, dict]:
    global _EMBEDDING_CACHE
    cache_key = (user_path, item_path)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    user_embs = _safe_load_json(user_path)
    item_embs = _safe_load_json(item_path)
    # Unify keys as strings
    user_embs = {str(k): np.asarray(v, dtype=np.float32) for k, v in user_embs.items()} if user_embs else {}
    item_embs = {str(k): np.asarray(v, dtype=np.float32) for k, v in item_embs.items()} if item_embs else {}
    logger.log(f"Loaded user embeddings: {user_path} | users={len(user_embs)}")
    logger.log(f"Loaded item embeddings: {item_path} | items={len(item_embs)}")
    
    _EMBEDDING_CACHE[cache_key] = (user_embs, item_embs)
    return user_embs, item_embs

def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x = x.astype(np.float32)
    m = np.max(x)
    ex = np.exp(x - m)
    s = np.sum(ex)
    return ex / (s if s > 0 else 1.0)

def compute_ctr_probs(user_id: str, candidates: list[str], user_embs: dict, item_embs: dict) -> list[float]:
    """
    CTR(u,i) = sigmoid( cosine(u, i) / tau )
    - tau defaults to the standard deviation of candidate cosine scores; falls back to a constant (default 0.5) when too small or non-finite
    - Missing user vector: return a constant small probability 0.01
    - Missing / invalid item vector: heavy down-weight 0.001
    - Dimension mismatch / zero norm / non-finite: heavy down-weight 0.001
    """
    if not candidates:
        return []

    uvec = user_embs.get(str(user_id))
    if uvec is None:
        return [0.01] * len(candidates)

    u = np.asarray(uvec, dtype=float)
    unorm = np.linalg.norm(u)
    if not np.isfinite(unorm) or unorm == 0.0:
        return [0.01] * len(candidates)

    # First pass: compute cosine scores for valid candidates, used to estimate the temperature tau
    cos_scores = []
    valid_flags = []
    for iid in candidates:
        ivec = item_embs.get(str(iid))
        if ivec is None:
            valid_flags.append(False)
            cos_scores.append(None)
            continue

        v = np.asarray(ivec, dtype=float)
        vnorm = np.linalg.norm(v)
        if not np.isfinite(vnorm) or vnorm == 0.0 or v.shape != u.shape:
            valid_flags.append(False)
            cos_scores.append(None)
            continue

        cos = float(np.dot(u, v) / (unorm * vnorm + 1e-8))
        cos = max(min(cos, 1.0), -1.0) 
        valid_flags.append(True)
        cos_scores.append(cos)

    # Dynamic temperature estimation: stddev of candidate cosine scores; fall back to constant if too small or non-finite
    valid_values = [c for c in cos_scores if c is not None]
    tau = float(np.std(valid_values)) if valid_values else 0.0
    if not np.isfinite(tau) or tau < 1e-3:
        tau = 0.5  

    # Second pass: apply Sigmoid with temperature scaling to compute CTR
    probs = []
    for flag, cos in zip(valid_flags, cos_scores):
        if not flag:
            probs.append(0.001)
            continue
        z = cos / tau
        z = max(min(z, 20.0), -20.0)  # Numerical stability, avoid exp overflow
        ctr = 1.0 / (1.0 + math.exp(-z))
        probs.append(ctr)

    return probs

# Position exposure function (exposure decreases from top to bottom)
def position_bias(rank):
    return 1.0 / np.log2(rank + 1)




def compute_eiu_values(candidates: list[str], ctr_probs: list[float]) -> list[float]:
    """
    Correct EIU definition: EIU(i) = position_bias(rank_i) * CTR(i)
    - rank is 1-based
    - Does not depend on training exposure or total_users
    """
    if not candidates or not ctr_probs:
        return [0.0] * len(candidates)

    eiu_vals = []
    for r, (iid, ctr) in enumerate(zip(candidates, ctr_probs), start=1):
        pos_weight = position_bias(r)
        eiu_vals.append(float(pos_weight) * float(ctr))
    return eiu_vals

def extract_alpha_from_filename(file_path):
    base = os.path.basename(file_path)
    match = re.search(r'_alpha([0-9]*\.?[0-9]+)', base)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None


def calculate_metrics(relevance, ranked_items, target_id, eiu_values=None):
    """Compute various evaluation metrics."""
    if not ranked_items:
        return {
            'NDCG@1': 0, 'NDCG@5': 0, 'NDCG@10': 0,
            'MRR': 0,
            'target_rank': None,
            'ranked_items': []
        }
    
    # Compute target item rank
    target_rank = None
    if target_id in ranked_items:
        target_rank = ranked_items.index(target_id) + 1
    
    # EIU aggregated metrics (aligned with candidate list positions)
    eiu1 = eiu5 = eiu10 = None
    target_eiu = None
    if eiu_values is not None and len(eiu_values) == len(ranked_items):
        eiu1 = float(np.sum(eiu_values[:1])) if eiu_values else 0.0
        eiu5 = float(np.sum(eiu_values[:5])) if eiu_values else 0.0
        eiu10 = float(np.sum(eiu_values[:10])) if eiu_values else 0.0
        # Target item EIU (if present in candidates)
        if target_id in ranked_items:
            idx = ranked_items.index(target_id)
            target_eiu = float(eiu_values[idx])

    return {
        'NDCG@1': ndcg_at_k(relevance, 1),
        'NDCG@5': ndcg_at_k(relevance, 5),
        'NDCG@10': ndcg_at_k(relevance, 10),
        'MRR': mrr(relevance),
        'target_rank': target_rank,
        'ranked_items': ranked_items,
        # EIU metrics
        'EIU@1_sum': eiu1,
        'EIU@5_sum': eiu5,
        'EIU@10_sum': eiu10,
        'target_eiu': target_eiu,
    }

# ======== Main functions ========

def evaluate_recommendation_process_file(file_path, exposure_path=None, user_path=None, item_path=None):
    """Evaluate recommendation_process_alphaXX.jsonl: compute metrics for candidates_before and final_candidates."""
    try:
        data = load_recommendation_process_jsonl(file_path)
    except Exception as e:
        logger.log(f"Failed to load file: {file_path} | {e}")
        return None

    # For more accurate Gini computation, load training exposure item set to determine the full item universe
    exposure_map = load_item_exposure_train(get_exposure_path("train"))
    all_known_items = set(exposure_map.keys())

    before_metrics_list = []
    after_metrics_list = []
    
    before_item_counts = Counter()
    after_item_counts = Counter()

    # Load exposure and embeddings, as well as total user count estimation
    if user_path and item_path:
        user_embs, item_embs = load_user_item_embeddings(user_path=user_path, item_path=item_path)
    else:
        user_embs, item_embs = load_user_item_embeddings()

    for rec in data:
        target_item = rec.get('target_item', {})
        target_id = target_item.get('id') if isinstance(target_item, dict) else str(target_item)

        candidates_before = [str(x) for x in rec.get('candidates_before', [])]
        final_candidates = [str(x) for x in rec.get('final_candidates', [])]

        # Record items in the universe
        all_known_items.update(candidates_before)
        all_known_items.update(final_candidates)

        # Parse user ID (fallback across multiple field names)
        user_id = rec.get('user_id') or rec.get('uid') or rec.get('user') or rec.get('customer_id')
        user_id = str(user_id) if user_id is not None else None

        if not target_id or (not candidates_before and not final_candidates):
            continue


        if candidates_before:
            # Use position-weighted exposure statistics
            for r, iid in enumerate(candidates_before, start=1):
                before_item_counts[iid] += position_bias(r)
            relevance_b = [1 if item == target_id else 0 for item in candidates_before]
            ctr_b = compute_ctr_probs(user_id, candidates_before, user_embs, item_embs) if user_id else [1.0/len(candidates_before)]*len(candidates_before)
            eiu_b = compute_eiu_values(candidates_before, ctr_b)
            EIU_before_sum = float(np.sum(eiu_b))

            mb = calculate_metrics(relevance_b, candidates_before, target_id, eiu_values=eiu_b)
            # Record EIU total gain (before)
            mb['EIU_sum'] = EIU_before_sum
            # Single item (target) gain (before)
            if target_id in candidates_before:
                idx_b = candidates_before.index(target_id)
                mb['target_eiu'] = float(eiu_b[idx_b])
            else:
                mb['target_eiu'] = 0.0
            before_metrics_list.append(mb)

        if final_candidates:
            # Use position-weighted exposure statistics
            for r, iid in enumerate(final_candidates, start=1):
                after_item_counts[iid] += position_bias(r)
            relevance_a = [1 if item == target_id else 0 for item in final_candidates]
            ctr_a = compute_ctr_probs(user_id, final_candidates, user_embs, item_embs) if user_id else [1.0/len(final_candidates)]*len(final_candidates)
            eiu_a = compute_eiu_values(final_candidates, ctr_a)
            EIU_after_sum = float(np.sum(eiu_a))

            ma = calculate_metrics(relevance_a, final_candidates, target_id, eiu_values=eiu_a)
            # Record EIU total gain (after)
            ma['EIU_sum'] = EIU_after_sum
            # Single item (target) gain (after)
            if target_id in final_candidates:
                idx_a = final_candidates.index(target_id)
                ma['target_eiu'] = float(eiu_a[idx_a])
            else:
                ma['target_eiu'] = 0.0

            # Record total gain change before/after reranking
            if candidates_before:
                ma['delta_EIU'] = EIU_after_sum - EIU_before_sum
            else:
                ma['delta_EIU'] = None
            after_metrics_list.append(ma)

    def summarize_metrics(metrics_list):
        if not metrics_list:
            return {
                'summary_metrics': {},
                'target_rank_stats': {},
                'processed_records': 0,
            }

        summary = {}
        for metric in ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR',
                       'EIU@1_sum', 'EIU@5_sum', 'EIU@10_sum', 'target_eiu',
                       'EIU_sum', 'delta_EIU']:
            vals = [m[metric] for m in metrics_list if m.get(metric) is not None]
            if vals:
                summary[f'{metric}_mean'] = float(np.mean(vals))
                summary[f'{metric}_std'] = float(np.std(vals))

        target_ranks = [m['target_rank'] for m in metrics_list if m['target_rank'] is not None]
        target_rank_stats = {}
        if target_ranks:
            target_rank_stats = {
                'mean_rank': float(np.mean(target_ranks)),
                'median_rank': float(np.median(target_ranks)),
                'min_rank': int(min(target_ranks)),
                'max_rank': int(max(target_ranks)),
            }

        return {
            'summary_metrics': summary,
            'target_rank_stats': target_rank_stats,
            'processed_records': len(metrics_list),
        }

    before_summary = summarize_metrics(before_metrics_list)
    after_summary = summarize_metrics(after_metrics_list)
    
    total_distinct_items = len(all_known_items)
    
    return {
        'file': os.path.basename(file_path),
        'total_records': len(data),
        'before': before_summary,
        'after': after_summary,
    }

def evaluate_recommendation_process_dir(dir_path, sample_size=None):
    """Scan .jsonl files matching recommendation_process_alpha in the directory and evaluate them."""
    files = []
    for fname in os.listdir(dir_path):
        # Exclude intermediate shard files (files containing _part)
        if fname.endswith(".jsonl") and "recommendation_process_alpha" in fname and "_part" not in fname:
            files.append(os.path.join(dir_path, fname))
    if not files:
        logger.log(f"No matching .jsonl files found in directory: {dir_path}")
        return {}

    items = []
    for f in files:
        alpha = extract_alpha_from_filename(f)
        items.append((alpha, f, os.path.basename(f)))
    items.sort(key=lambda x: (x[0] if x[0] is not None else float('inf'), x[1]))

    results = {}
    for alpha, fpath, fname in items:
        try:
            # Optional sampling
            if sample_size:
                # Simply read the first sample_size lines
                subset = []
                with open(fpath, 'r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh):
                        if i >= sample_size:
                            break
                        if line.strip():
                            subset.append(json.loads(line))
                # Write to a temporary file for evaluation
                tmp_path = fpath + ".subset.tmp"
                with open(tmp_path, 'w', encoding='utf-8') as th:
                    for obj in subset:
                        th.write(json.dumps(obj, ensure_ascii=False) + "\n")
                res = evaluate_recommendation_process_file(tmp_path)
                try:
                    os.remove(tmp_path)
                except:
                    pass
            else:
                res = evaluate_recommendation_process_file(fpath)
            # Fairness evaluation
            fairness = evaluate_fairness_for_file(fpath, exposure_path=get_exposure_path("train"), num_groups=8)
            res['fairness'] = fairness

            res['alpha'] = alpha
            results[fname] = res
            logger.log(f"Evaluated file: {fname} (alpha={alpha}) | records: {res['after']['processed_records']}/{res['total_records']}")
            # Log fairness metrics (per K)
            if fairness:
                ks = fairness.get('ks', [1, 3, 5, 7])
                fb = fairness.get('before', {})
                fa = fairness.get('after', {})
                for k in ks:
                    d_b = fb.get('DGU', {}).get(k, None)
                    d_a = fa.get('DGU', {}).get(k, None)  
                    m_b = fb.get('MGU', {}).get(k, None)
                    m_a = fa.get('MGU', {}).get(k, None)
                    g_b = fb.get('Gini', {}).get(k, None)
                    g_a = fa.get('Gini', {}).get(k, None)
                    if None not in (d_b, d_a, m_b, m_a, g_b, g_a):
                        logger.log(f"Fairness@{k}: DGU={d_b:.4f}->{d_a:.4f}, MGU={m_b:.4f}->{m_a:.4f}, Gini={g_b:.4f}->{g_a:.4f}, delta_DGU={d_a-d_b:.4f}, delta_Gini={g_a-g_b:.4f}")
        except Exception as e:
            logger.log(f"Failed to evaluate file {fname}: {e}")
            continue

    return results

def print_recommendation_process_report(results):
    """Print the evaluation report for recommendation_process_alpha*.jsonl files (before/after comparison and lift)."""
    if not results:
        logger.log("No recommendation_process_alpha*.jsonl files found for evaluation.")
        return

    sorted_items = sorted(
        ((res.get('alpha', None), fname, res) for fname, res in results.items()),
        key=lambda x: (x[0] if x[0] is not None else float('inf'), x[1])
    )

    for alpha, fname, res in sorted_items:
        before = res.get('before', {})
        after = res.get('after', {})
        bsum = before.get('summary_metrics', {})
        asum = after.get('summary_metrics', {})
        fairness = res.get('fairness', None)

        logger.log(f"\n[ALPHA] file: {fname} | alpha={alpha}")
        logger.log(f"Processed records: after={after.get('processed_records', 0)} / total={res.get('total_records', 0)}")

        for metric in ['NDCG@10', 'NDCG@5', 'NDCG@1', 'MRR']:
            b_mean = bsum.get(f"{metric}_mean", None)
            a_mean = asum.get(f"{metric}_mean", None)
            if b_mean is not None and a_mean is not None:
                lift = a_mean - b_mean
                logger.log(f"{metric}: before={b_mean:.4f}, after={a_mean:.4f}, lift={lift:.4f}")
        
        # Print before/after comparison of EIU metrics
        for eiu_metric in ['EIU@10_sum', 'EIU@5_sum', 'EIU@1_sum', 'target_eiu']:
            b_mean = bsum.get(f"{eiu_metric}_mean", None)
            a_mean = asum.get(f"{eiu_metric}_mean", None)
            if b_mean is not None and a_mean is not None:
                lift = a_mean - b_mean
                logger.log(f"{eiu_metric}: before={b_mean:.6f}, after={a_mean:.6f}, lift={lift:.6f}")

        # Print fairness metrics DGU, MGU
        if fairness:
            ks = fairness.get('ks', [1, 3, 5, 7, 10])
            fb = fairness.get('before', {})
            fa = fairness.get('after', {})
            for k in ks:
                d_b = fb.get('DGU', {}).get(k, None)
                d_a = fa.get('DGU', {}).get(k, None)
                m_b = fb.get('MGU', {}).get(k, None)
                m_a = fa.get('MGU', {}).get(k, None)
                if None not in (d_b, d_a, m_b, m_a):
                    logger.log(f"Fairness@{k}: DGU={d_b:.4f}->{d_a:.4f} (lift={d_a-d_b:.4f}), MGU={m_b:.4f}->{m_a:.4f} (lift={m_a-m_b:.4f})")


def load_item_exposure_train(exposure_path=None) -> dict:
    """Load each item's exposure from training interactions, returning {item_id(str): exposure(float)}."""
    global _EXPOSURE_CACHE
    if exposure_path is None:
        exposure_path = get_exposure_path("train")
    
    if exposure_path in _EXPOSURE_CACHE:
        return _EXPOSURE_CACHE[exposure_path]

    try:
        with open(exposure_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Normalize key type to str
        res = {str(k): float(v) for k, v in data.items()}
        _EXPOSURE_CACHE[exposure_path] = res
        return res
    except Exception as e:
        logger.log(f"Failed to load training exposure: {e}")
        return {}

def build_popularity_groups(exposure_map: dict, num_groups: int = 8) -> list:
    """
    Split items into popularity groups from low to high exposure (approximate quantile grouping).
    Returns: list[set[str]], each group contains item ids.
    """
    # Items with no exposure are treated as 0
    items = sorted([(str(i), float(exp)) for i, exp in exposure_map.items()], key=lambda x: x[1])
    n = len(items)
    if n == 0:
        return [set() for _ in range(num_groups)]
    # Equal-size grouping
    chunk = max(1, n // num_groups)
    groups = []
    for g in range(num_groups):
        start = g * chunk
        end = (g + 1) * chunk if g < num_groups - 1 else n
        group_ids = {iid for iid, _ in items[start:end]}
        groups.append(group_ids)
    return groups

def train_dist_from_exposure(exposure_map: dict, group_boundaries: list) -> np.ndarray:
    """
    Compute the training distribution p_train(g) based on exposure:
    p_train(g) = sum of exposure in the group / total exposure
    """
    total_exp = float(sum(exposure_map.values()))
    if total_exp <= 0:
        return np.zeros(len(group_boundaries), dtype=float)
    group_probs = []
    for grp in group_boundaries:
        s = sum(exposure_map.get(item_id, 0.0) for item_id in grp)
        group_probs.append(float(s) / total_exp)
    return np.asarray(group_probs, dtype=float)

def compute_group_distribution(items, group_boundaries):
    """
    Group items by popularity (by count proportion).
    items: list[str] list of recommended item ids
    group_boundaries: list[set[str]] item id set of each group
    Returns: array of per-group proportions (length = number of groups)
    """
    items = [str(x) for x in items]
    group_counts = np.zeros(len(group_boundaries), dtype=float)
    for i, group in enumerate(group_boundaries):
        group_counts[i] = sum(item in group for item in items)
    denom = max(len(items), 1)
    return group_counts / denom

def evaluate_fairness_for_file(file_path, exposure_path=None, num_groups=8):
    if exposure_path is None:
        exposure_path = get_exposure_path("train")
    # Load recommendation process data
    try:
        data = load_recommendation_process_jsonl(file_path)
    except Exception as e:
        logger.log(f"Failed to load file (fairness): {file_path} | {e}")
        return None

    # Collect all items appearing in this file
    all_items_in_file = set()
    for rec in data:
        all_items_in_file.update(map(str, rec.get('candidates_before', [])))
        all_items_in_file.update(map(str, rec.get('final_candidates', [])))

    # Load training exposure, and add items not seen in training with exposure set to 0
    exposure_map = load_item_exposure_train(exposure_path)
    for iid in all_items_in_file:
        if iid not in exposure_map:
            exposure_map[iid] = 0.0

    # Build groups and compute training distribution based on the extended exposure
    group_boundaries = build_popularity_groups(exposure_map, num_groups=num_groups)
    p_train = train_dist_from_exposure(exposure_map, group_boundaries)

    # Build item_id -> group_index mapping
    def build_group_index_map(groups):
        mapping = {}
        for gi, grp in enumerate(groups):
            for iid in grp:
                mapping[str(iid)] = gi
        return mapping

    group_index_map = build_group_index_map(group_boundaries)
    G = len(group_boundaries)

    # K values to compute
    ks = [1, 3, 5, 7, 10]

    # Cumulative per-K group counts and item-level weighted exposure (for Gini)
    before_counts = {k: np.zeros(G, dtype=float) for k in ks}
    before_item_exposures = {k: Counter() for k in ks}
    before_totals = {k: 0.0 for k in ks}
    
    after_counts = {k: np.zeros(G, dtype=float) for k in ks}
    after_item_exposures = {k: Counter() for k in ks}
    after_totals = {k: 0.0 for k in ks}

    for rec in data:
        candidates_before = [str(x) for x in rec.get('candidates_before', [])]
        final_candidates = [str(x) for x in rec.get('final_candidates', [])]

        for k in ks:
            if candidates_before:
                topk = candidates_before[:k]
                for r, iid in enumerate(topk, start=1):
                    # Use position bias for weighted exposure statistics
                    weight = position_bias(r)
                    before_item_exposures[k][iid] += weight
                    gi = group_index_map.get(str(iid))
                    if gi is not None:
                        before_counts[k][gi] += weight
                        before_totals[k] += weight
            if final_candidates:
                topk = final_candidates[:k]
                for r, iid in enumerate(topk, start=1):
                    weight = position_bias(r)
                    after_item_exposures[k][iid] += weight
                    gi = group_index_map.get(str(iid))
                    if gi is not None:
                        after_counts[k][gi] += weight
                        after_totals[k] += weight

    def safe_dist(counts: np.ndarray, total: float, G: int):
        if total <= 0:
            return np.zeros(G, dtype=float)
        return counts / total

    before_dist = {k: safe_dist(before_counts[k], before_totals[k], G) for k in ks}
    after_dist = {k: safe_dist(after_counts[k], after_totals[k], G) for k in ks}

    def dgu_mgu(p_rec: np.ndarray, p_train: np.ndarray):
        diff = np.abs(p_rec - p_train)
        dgu = float(diff.sum() / 2.0)
        mgu = float(diff.max() if diff.size else 0.0)
        return dgu, mgu

    before_dgu, before_mgu = {}, {}
    after_dgu, after_mgu = {}, {}

    for k in ks:
        dgu_b, mgu_b = dgu_mgu(before_dist[k], p_train)
        dgu_a, mgu_a = dgu_mgu(after_dist[k], p_train)
        before_dgu[k], before_mgu[k] = dgu_b, mgu_b
        after_dgu[k], after_mgu[k] = dgu_a, mgu_a
        
    return {
        "ks": ks,
        "train_distribution": p_train.tolist(),
        "before": {
            "DGU": {k: float(before_dgu[k]) for k in ks},
            "MGU": {k: float(before_mgu[k]) for k in ks},
            "dist": {k: before_dist[k].tolist() for k in ks},
        },
        "after": {
            "DGU": {k: float(after_dgu[k]) for k in ks},
            "MGU": {k: float(after_mgu[k]) for k in ks},
            "dist": {k: after_dist[k].tolist() for k in ks},
        },
    }
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data statistics and fairness evaluation for stage-2 platform reranking")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Directory of rerank results to evaluate (containing recommendation_process_alphaXX.jsonl)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file path (optional)")
    args = parser.parse_args()

    logger = Logger(log_file=args.log_file)
    target_dir = args.target_dir

    logger.log("Recommender system data statistics analysis tool")
    logger.log(f"Start evaluating directory: {target_dir} for recommendation_process_alphaXX.jsonl files")

    results = evaluate_recommendation_process_dir(target_dir)
    print_recommendation_process_report(results)

    logger.log_separator()
    logger.log("Evaluation finished.")
