#!/usr/bin/env bash
# TriRec Stage 1: generative item self-promotion + LLM ranking over the
# hard-negative candidate set (ground truth + n negatives sampled from SASRec top-100).
# Output: log/exp_<EXPERIMENT_ID>/recommendation_process.jsonl
#         log/exp_<EXPERIMENT_ID>/recommendation_details.jsonl  (per-candidate promotions)
# Required / optional environment variables:
#   OPENAI_API_KEY
#   OPENAI_BASE_URL      # (optional)
#   MAX_WORKERS          # (optional)
#   CAND_NUM             # (optional) candidate set size |C_u|, default 10
#   EXPERIMENT_ID        # (optional)
#   PROMO_MODE           # (optional) full | grounded | generic | none, default full
#                        #   the three arms of the controlled ablation are
#                        #   none / generic / full; grounded is the fourth arm
#   MEMORY_UPDATE        # (optional) 1 to enable the item memory update
set -e
cd "$(dirname "$0")/.."

python src/TriRec_stage1_recall.py
