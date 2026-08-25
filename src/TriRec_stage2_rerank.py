
import json
import os
import shutil
import numpy as np
import argparse
import datetime
from collections import Counter

from config import (
    DOMAIN,
    num_users_to_sample,
    get_experiment_id,
    get_exposure_path,
    EXPOSURE_ROOT,
    LOG_ROOT,
    LOG_VISUAL_ROOT,
    DATASET_ROOT,
)


exp_name = f"{DOMAIN}"
exp_id = get_experiment_id()
mode = "test"
max_retries = 3

item_exposure_before = Counter()
item_exposure_before_top3 = Counter()
# item_exposure_after = Counter()
item_exposure_after_all = Counter()   # Statistics of overall recommendation exposure
item_exposure_after_top3 = Counter()  # Statistics of user top-3 exposure

# File-level new functions: load_exposure_dict and re_rank_candidates
class Logger:
    """Logger that writes output to a txt file."""
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Create the log file if it does not exist
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Rerank processing log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        """Record a message to the log file."""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

def load_exposure_dict(path: str):
    """
    Load the item exposure dictionary from the fixed path.
    Return an empty dict if not found.
    """
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"Exposure statistics file not found: {path}, using empty dict.")
            return {}
    except Exception as e:
        print(f"Failed to read exposure statistics: {e}, using empty dict.")
        return {}

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types to resolve JSON serialization issues.
    """
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

def log_run_params(output_dir: str, params: dict):
    """
    Append the parameters of this run to {output_dir}/run_params.txt.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "run_params.txt")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("=== RUN PARAMS ===\n")
            for k, v in params.items():
                try:
                    if isinstance(v, (dict, list, tuple, np.ndarray)):
                        v = json.dumps(convert_numpy_types(v), ensure_ascii=False)
                except Exception:
                    pass
                f.write(f"{k}: {v}\n")
            f.write("\n")
    except Exception as e:
        print(f"[warn] Failed to write parameter log: {e}")

def soft_normalize_rel(rel_scores, beta=1.0):
    """
    Apply soft normalization (Sigmoid) to rel_scores.
    beta: controls compression strength (larger -> closer to min-max).
    """
    rel_scores = np.array(rel_scores, dtype=float)
    mu = np.mean(rel_scores)
    norm_rel = 1 / (1 + np.exp(-beta * (rel_scores - mu)))
    return norm_rel

# ==========================================================================
# Interchangeable forms of the core functions (position weight / relevance fusion / long-tail decay)
# ==========================================================================
def position_weight(pos: int, fn: str = "log2", K: int = 10,
                    power_p: float = 1.0, exp_k: float = 0.5) -> float:
    """
    Position weight w(pos); defaults to log2 (classic NDCG form).
    - log2  : 1 / log2(pos + 2)           (baseline)
    - power : 1 / (pos + 1)^power_p        (power law, steeper than log)
    - exp   : exp(-exp_k * pos)            (exponential decay, head-sensitive)
    - linear: max(0, 1 - pos / K)          (linear, smoothest)
    """
    if fn == "log2":
        return float(1.0 / np.log2(pos + 2))
    if fn == "power":
        return float(1.0 / (pos + 1.0) ** power_p)
    if fn == "exp":
        return float(np.exp(-exp_k * pos))
    if fn == "linear":
        return float(max(0.0, 1.0 - pos / float(max(K, 1))))
    raise ValueError(f"unknown position_weight fn: {fn}")

def apply_phi(sim_emb, fn: str = "sigmoid", alpha: float = 1.0):
    """
    Map cosine similarity sim_emb (in [-1, 1]) to a non-negative benefit/CTR baseline.
    - exp     : phi(x) = exp(x)                (default)
    - sigmoid : phi(x) = 1 / (1 + exp(-alpha*x))
    - linear  : phi(x) = (x + 1) / 2            (map to [0, 1])
    - relu    : phi(x) = max(x, 0)
    """
    arr = np.asarray([float(x) for x in sim_emb], dtype=float)
    if fn == "exp":
        return np.exp(arr)
    if fn == "sigmoid":
        return 1.0 / (1.0 + np.exp(-alpha * arr))
    if fn == "linear":
        return (arr + 1.0) / 2.0
    if fn == "relu":
        return np.maximum(arr, 0.0)
    raise ValueError(f"unknown phi fn: {fn}")



def build_popularity_groups(exposure_map: dict, num_groups: int = 8):
    """
    Divide items evenly into num_groups groups in ascending exposure order; return list[set[str]].
    """
    items = sorted([(str(i), float(exp)) for i, exp in exposure_map.items()], key=lambda x: x[1])
    n = len(items)
    if n == 0:
        return [set() for _ in range(num_groups)]
    chunk = max(1, n // num_groups)
    groups = []
    for g in range(num_groups):
        start = g * chunk
        end = (g + 1) * chunk if g < num_groups - 1 else n
        groups.append({iid for iid, _ in items[start:end]})
    return groups

def train_dist_from_exposure(exposure_map: dict, group_boundaries: list):
    """
    p_train[g] = sum of exposures of this group / sum of all exposures.
    """
    total = float(sum(exposure_map.values()))
    if total <= 0:
        return np.zeros(len(group_boundaries), dtype=float)
    probs = []
    for grp in group_boundaries:
        s = sum(exposure_map.get(item_id, 0.0) for item_id in grp)
        probs.append(float(s) / total)
    return np.asarray(probs, dtype=float)

def build_group_index_map(groups):
    """
    item_id -> group_index
    """
    mapping = {}
    for gi, grp in enumerate(groups):
        for iid in grp:
            mapping[str(iid)] = gi
    return mapping

def compute_embedding_relevance(user_id, item_ids, user_embeddings, item_embeddings):
    """
    Compute cosine similarity between user and item embeddings.
    """
    u_id = str(user_id)
    if u_id not in user_embeddings:
        return [0.0] * len(item_ids)
    
    u_vec = np.array(user_embeddings[u_id])
    u_norm = np.linalg.norm(u_vec)
    
    scores = []
    for iid in item_ids:
        i_id = str(iid)
        if i_id in item_embeddings:
            i_vec = np.array(item_embeddings[i_id])
            i_norm = np.linalg.norm(i_vec)
            if u_norm > 0 and i_norm > 0:
                # Cosine similarity
                score = np.dot(u_vec, i_vec) / (u_norm * i_norm)
                scores.append(float(score))
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return scores


def re_rank_candidates_greedy(
    candidates,
    rel_scores,
    ctr_scores,
    alpha: float, # here alpha becomes alpha_max
    alpha_min: float, # new parameter
    alpha_p: float, # new parameter
    lambda1: float,
    lambda2: float,
    lambda_item: float,
    K: int,
    group_of: dict,
    p_train: np.ndarray,
    global_counts: np.ndarray,
    global_N: float,
    rank_weights: np.ndarray,
    pos_fn: str = "log2",
):
    """
    Greedy sequential action generation (Eq. 16): the list is built top-down and
    the exposure state is carried across rounds.
    - Greedy is applied only to the top-K positions; the rest keeps the original order.
    - Updates global_counts / global_N in real time.
    """

    # Normalize relevance
    rel_norm = soft_normalize_rel(rel_scores, beta=1.0)
    G = len(p_train)
    eps = 1e-12

    def get_pos_weight(pos: int) -> float:
        # Prefer the provided rank_weights; if out of range, fall back to the chosen function form
        if pos < len(rank_weights):
            return float(rank_weights[pos])
        return position_weight(pos, fn=pos_fn, K=K)


    def minmax_norm(values: list) -> list:
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if abs(vmax - vmin) < 1e-12:
            return [0.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    chosen = []
    combined_scores = []

    # Iterate over positions and perform the overall greedy
    for pos in range(len(candidates)):
        w = get_pos_weight(pos)

        # Compute dynamic alpha_pos
        w_norm = float(w / rank_weights[0]) # normalized position weight; rank_weights[0] is the max weight
        alpha_pos = float(alpha_min + (alpha - alpha_min) * (w_norm ** alpha_p))

        # Global ratio at the current position and per-group gaps
        p_rec_now = global_counts / max(global_N, eps)
        gaps_now = np.abs(p_rec_now - p_train)
        max_gap_now = float(gaps_now.max()) if gaps_now.size > 0 else 0.0

        # Pre-collect three types of marginal gains (user + platform group + Item benefit) for within-step normalization
        delta_acc_list = []
        delta_dgu_list = []
        delta_mgu_list = []
        delta_eiu_list = []
        cand_indices = []

        for j, cid in enumerate(candidates):
            if cid in chosen:
                continue
            cand_indices.append(j)

            # Delta_acc: position-related user gain
            delta_acc = float(rel_norm[j]) * float(w)

            # EIU = position_weight * CTR
            item_eiu_gain = float(ctr_scores[j]) * float(w)
            delta_eiu_list.append(item_eiu_gain)

            # Get group info and current/after-insertion ratio for this group
            g = group_of.get(str(cid))
            if g is None or g < 0 or g >= G:
                # Missing group: no fairness gain
                delta_dgu = 0.0
                delta_mgu = 0.0
            else:
                p_g = float(global_counts[g] / max(global_N, eps))
                p_g_new = float((global_counts[g] + w) / (global_N + w))

                # Delta_DGU (reduction of the group's own gap; non-negative means improvement)
                gap_g = abs(p_g - p_train[g])
                gap_g_new = abs(p_g_new - p_train[g])
                delta_dgu = float(max(0.0, gap_g - gap_g_new))

                # Delta_MGU (existing logic: only consider reduction when this group is the current max gap)
                if gap_g >= max_gap_now - 1e-12:
                    delta_mgu = float(max(0.0, gap_g - gap_g_new))
                else:
                    delta_mgu = 0.0

            delta_acc_list.append(delta_acc)
            delta_dgu_list.append(delta_dgu)
            delta_mgu_list.append(delta_mgu)

        # Within-step min-max normalization (four dimensions normalized separately)
        acc_norm = minmax_norm(delta_acc_list)
        dgu_norm = minmax_norm(delta_dgu_list)
        mgu_norm = minmax_norm(delta_mgu_list)
        eiu_norm = minmax_norm(delta_eiu_list) # Normalize the benefit gain

        # All candidates have been placed
        if not cand_indices:
            break

        # Joint utility (Eq. 9): relevance-fairness gain * exposure-aware item utility
        best_score = -1e18
        best_idx_in_cands = None
        for k, j in enumerate(cand_indices):
            # Relevance-fairness gain g(.) (Eq. 10 with the platform term of Eq. 11):
            # a position-aware convex combination of the normalized user utility and
            # the normalized marginal fairness gains on DGU / MGU.
            total_marginal_gain = float(
                alpha_pos * acc_norm[k]
                + (1.0 - alpha_pos) * (lambda1 * dgu_norm[k] + lambda2 * mgu_norm[k])
            )
            # Exposure-aware item utility modulator (Eq. 15).  Under-exposure is not
            # handled here; it enters only through the group-fairness term above.
            score = total_marginal_gain * (eiu_norm[k]) ** lambda_item

            if score > best_score:
                best_score = score
                best_idx_in_cands = j

        # commit the best item for the current position
        best_item = candidates[best_idx_in_cands]
        chosen.append(best_item)
        combined_scores.append(best_score)

        # Update global counts (fairness state)
        g_best = group_of.get(str(best_item))
        if isinstance(g_best, int) and 0 <= g_best < G:
            global_counts[g_best] += w
        global_N += w

    # Return the full greedy sequence and the corresponding combined scores
    return chosen, combined_scores, global_counts, global_N


def load_results_jsonl(path: str):
    """
    Read the results_*.jsonl file and parse each line into a dict list.
    """
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except Exception as e:
                    print(f"[warn] Failed to parse JSONL line: {e}")
    except Exception as e:
        print(f"[error] Failed to open file: {e}")
    return records

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-two rerank script")
    parser.add_argument("--input_path", type=str, required=True, help="Path of the JSONL file to be reranked")
    parser.add_argument("--output_dir", type=str, default=str(LOG_VISUAL_ROOT), help="Directory to save rerank results")

    # Re-ranking hyperparameters
    parser.add_argument("--K", type=int, default=int(os.environ.get("CAND_NUM", 10)), help="Top-K for rerank, defaults to CAND_NUM")
    parser.add_argument("--lambda1", type=float, default=0.5, help="DGU weight")
    parser.add_argument("--lambda2", type=float, default=0.5, help="MGU weight")
    parser.add_argument("--lambda_item", type=float, default=10.0, help="Exponent lambda_item of the exposure-aware item utility (Eq. 15)")
    parser.add_argument("--alpha_min", type=float, default=0.1, help="Dynamic alpha lower bound")
    parser.add_argument("--alpha_p", type=float, default=0.1, help="Power exponent of alpha curve")

    # alpha_max sweep range
    parser.add_argument("--alpha_start", type=float, default=0.1)
    parser.add_argument("--alpha_end", type=float, default=1.01)
    parser.add_argument("--alpha_step", type=float, default=0.1)

    # Functional forms of the core terms
    parser.add_argument("--pos_fn", type=str, default="log2", choices=["log2", "power", "exp", "linear"], help="Position weight function")
    parser.add_argument("--phi_fn", type=str, default="sigmoid", choices=["exp", "sigmoid", "linear", "relu"], help="CTR/benefit baseline mapping phi(sim_emb)")
    parser.add_argument("--phi_alpha", type=float, default=1.0, help="Slope alpha when phi=sigmoid")

    args = parser.parse_args()

    results_jsonl_path = args.input_path
    
    # Extract the experiment ID and create the corresponding result subfolder
    input_dir = os.path.dirname(results_jsonl_path)
    exp_folder_name = os.path.basename(input_dir)
    base_output_dir = os.path.join(args.output_dir, exp_folder_name)
    output_dir = base_output_dir
    
    # Append a _k suffix when the target folder already exists
    k = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_output_dir}_{k}"
        k += 1
        
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the logger
    summary_log_path = os.path.join(output_dir, "rerank_summary.txt")
    logger = Logger(summary_log_path)

    # Retrieve experiment metadata (refer to evaluate_two_stage_logs.py)
    exec_info_path = os.path.join(input_dir, "execution_info.txt")
    
    # Copy execution_info.txt to the result directory
    if os.path.exists(exec_info_path):
        shutil.copy2(exec_info_path, os.path.join(output_dir, "execution_info.txt"))
        logger.log(f"Copied execution_info.txt to {output_dir}")

    script_name = "Unknown"
    domain_info = "Unknown"
    if os.path.exists(exec_info_path):
        with open(exec_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "File Name:" in line:
                    script_name = line.split(":")[-1].strip()
                elif "Memory Folder:" in line:
                    domain_info = line.split(":")[-1].strip()

    logger.log(f"Evaluation directory: {input_dir}")
    logger.log(f"Script name: {script_name}")
    logger.log(f"Domain info: {domain_info}")

    # --- Dynamically load exposure data ---
    # Extract the domain string from domain_info (e.g. "CDs_100" -> "CDs_100")
    domain_str = domain_info.strip()
    first_domain = domain_str.split(" ")[0]
    
    exposure_folder = str(EXPOSURE_ROOT)
    import glob
    # Try to find the training exposure file matching this domain (supports different user-count suffixes)
    train_matches = glob.glob(os.path.join(exposure_folder, f"item_exposure_train_{first_domain}*.json"))
        
    if train_matches:
        # Prefer exact match, otherwise pick the first one
        exact_match = [m for m in train_matches if domain_str in m]
        train_exposure_path = exact_match[0] if exact_match else train_matches[0]
            
        # Extract the actual filename suffix (also used to locate the embedding folder)
        actual_suffix = os.path.basename(train_exposure_path).replace("item_exposure_train_", "").replace(".json", "")

        logger.log(f"Dynamically located exposure files: domain={first_domain}, suffix={actual_suffix}")
    else:
        # Final fallback: fall back to the configuration from config
        fallback_domain = f"{DOMAIN}_{num_users_to_sample}"
        actual_suffix = fallback_domain
        train_exposure_path = os.path.join(exposure_folder, f"item_exposure_train_{fallback_domain}.json")
        logger.log(f"Warning: exposure file matching {first_domain} not found, falling back to config: {fallback_domain}")
    
    logger.log(f"Loading training exposure data: {train_exposure_path}")
    exposure_dict = load_exposure_dict(train_exposure_path)

    # --- Load user and item embeddings ---
    embedding_dir = str(
        DATASET_ROOT / "user_item_data" / actual_suffix / "semantic_embeddings"
    )
    user_emb_path = os.path.join(embedding_dir, "user_embeddings_train.json")
    item_emb_path = os.path.join(embedding_dir, "item_embeddings.json")
    
    user_embeddings = {}
    item_embeddings = {}
    if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
        logger.log(f"Loading embedding data from: {embedding_dir}")
        with open(user_emb_path, "r", encoding="utf-8") as f:
            user_embeddings = json.load(f)
        with open(item_emb_path, "r", encoding="utf-8") as f:
            item_embeddings = json.load(f)
    else:
        logger.log(f"Warning: embedding files not found at {embedding_dir}")

    logger.log(f"Reading data from the result file: {results_jsonl_path}")
    records = load_results_jsonl(results_jsonl_path)
    logger.log(f"Total records: {len(records)}")

    # For grouping: items appearing in the log but missing from the exposure dict are set to 0 exposure
    all_item_ids = set()
    for rec in records:
        cands = rec.get("original_candidates") or rec.get("originalCandidates") or rec.get("ranked_ids") or []
        if cands and isinstance(cands[0], dict) and "id" in cands[0]:
            cands = [str(x.get("id")) for x in cands]
        else:
            cands = [str(x) for x in cands]
        for cid in cands:
            all_item_ids.add(str(cid))
    for iid in all_item_ids:
        if iid not in exposure_dict:
            exposure_dict[iid] = 0.0

    # Build 8 groups and compute training distribution
    groups = build_popularity_groups(exposure_dict, num_groups=8)
    group_of = build_group_index_map(groups)
    p_train = train_dist_from_exposure(exposure_dict, groups)

    # Hyperparameters (controlled via CLI for convenient OFAT sweep)
    K = int(args.K)
    lambda1 = float(args.lambda1)
    lambda2 = float(args.lambda2)
    lambda_item = float(args.lambda_item)
    alpha_min = float(args.alpha_min)
    alpha_p = float(args.alpha_p)
    pos_fn = str(args.pos_fn)
    phi_fn = str(args.phi_fn)
    phi_alpha = float(args.phi_alpha)
    rank_weights = np.asarray([position_weight(r, fn=pos_fn, K=K) for r in range(K)], dtype=float)

    log_run_params(output_dir, {
        "exp_name": "full",
        "exp_id": exp_id,
        "load_jsonl_path": results_jsonl_path,
        "K": K,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda_item": lambda_item,
        "alpha_min": alpha_min,
        "alpha_p": alpha_p,
        "pos_fn": pos_fn,
        "phi_fn": phi_fn,
        "phi_alpha": phi_alpha,
        "num_groups": len(groups),
        "rank_weights": rank_weights.tolist()
    })

    # Iterate over alpha values (range controlled via CLI)
    for alpha_val in np.arange(args.alpha_start, args.alpha_end, args.alpha_step):
        alpha = round(alpha_val, 2)
        print(f"\n--- Start processing alpha = {alpha} ---")

        # Define an independent log file for each alpha, saved under the specified output_dir
        # Extract the base part of the input filename and append the alpha suffix
        input_base = os.path.basename(results_jsonl_path).replace(".jsonl", "")
        # If the input filename is recommendation_process, try to extract exp_id from the path to avoid conflicts
        if input_base == "recommendation_process":
            parent_dir = os.path.basename(os.path.dirname(results_jsonl_path))
            input_base = f"{parent_dir}_{input_base}"
            
        log_file_path = os.path.join(output_dir, f"{input_base}_alpha{alpha}.jsonl")
        
        with open(log_file_path, "w", encoding="utf-8") as f:
            pass  # Make sure the file is cleared


        # Initialize the global fairness state (counts/N)
        global_counts = np.zeros(len(groups), dtype=float)
        global_N = 1e-6  # Prevent division by zero

        # Iterate over the JSONL records and perform the platform-implicit intervention (Method-A greedy)
        for idx, rec in enumerate(records):
            # Extract fields
            userId = rec.get("user_id") or rec.get("userId")
            target_item = rec.get("target_item") or rec.get("targetItem")

            # Compatible with the two-stage log format: prefer second_stage_items or final_candidates
            original_candidates = rec.get("second_stage_items") or rec.get("final_candidates") or rec.get("original_candidates") or rec.get("originalCandidates") or rec.get("ranked_ids") or []
            
        # Extract and convert IDs
            if original_candidates and isinstance(original_candidates[0], dict) and "id" in original_candidates[0]:
                original_candidates = [str(x.get("id")) for x in original_candidates]
            else:
                original_candidates = [str(x) for x in original_candidates]

            # Deduplicate while preserving order
            seen = set()
            dedup_candidates = []
            for cid in original_candidates:
                if cid not in seen:
                    dedup_candidates.append(cid)
                    seen.add(cid)
            original_candidates = dedup_candidates

            # User-agent relevance scores r_LLM; when the Stage-1 log carries no
            # scores, fall back to a rank-derived monotone surrogate
            user_agent_scores = rec.get("user_agent_scores") or rec.get("scores") or []
            if user_agent_scores:
                id2score = {str(e["id"]): float(e["score"]) for e in user_agent_scores if isinstance(e, dict) and "id" in e and "score" in e}
                score_llm = [id2score.get(cid, 0.0) for cid in original_candidates]
            else:
                # Rank-derived surrogate: 1.0, 0.9, 0.8 ...
                score_llm = [max(0.0, 1.0 - 0.1 * i) for i in range(len(original_candidates))]

            # Compute embedding similarity
            sim_emb = compute_embedding_relevance(userId, original_candidates, user_embeddings, item_embeddings)
            
            rel_scores = [float(score_llm[i]) for i in range(len(original_candidates))]
            ctr_scores = apply_phi(sim_emb, fn=phi_fn, alpha=phi_alpha)  # phi(sim_emb) as the CTR score
            
            # Greedy rerank (Top-K), and update global count/N in real time
            reranked_items, combined_scores, global_counts, global_N = re_rank_candidates_greedy(
                candidates=original_candidates,
                rel_scores=rel_scores,
                ctr_scores=ctr_scores,  # phi(sim_emb), the CTR proxy of Eq. 14
                alpha=alpha, # alpha here corresponds to alpha_max
                alpha_min=alpha_min,
                alpha_p=alpha_p,
                lambda1=lambda1,
                lambda2=lambda2,
                lambda_item=lambda_item,
                K=K,
                group_of=group_of,
                p_train=p_train,
                global_counts=global_counts,
                global_N=global_N,
                rank_weights=rank_weights,
                pos_fn=pos_fn
            )

            # Record post-improvement exposure
            for item_id in reranked_items:
                item_exposure_after_all[item_id] += 1
            for item_id in reranked_items[:3]:
                item_exposure_after_top3[item_id] += 1

            # Save to the log file corresponding to alpha
            log_entry = {
                "user_id": userId,
                "target_item": target_item,
                "candidates_before": original_candidates,
                "final_candidates": reranked_items,
                "rel_scores": rel_scores,
                "combined_scores": combined_scores,
            }
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(convert_numpy_types(log_entry), ensure_ascii=False) + "\n")
