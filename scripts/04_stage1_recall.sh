#!/usr/bin/env bash
# TriRec Stage 1: generative item self-promotion + LLM ranking over the
# hard-negative candidate set (ground truth + n negatives sampled from SASRec top-100).
# Output: log/exp_<EXPERIMENT_ID>/recommendation_process.jsonl
# Required / optional environment variables:
#   OPENAI_API_KEY
#   OPENAI_BASE_URL      # (optional)
#   MAX_WORKERS          # (optional)
#   CAND_NUM             # (optional) candidate set size |C_u|, default 10
#   EXPERIMENT_ID        # (optional)
set -e
cd "$(dirname "$0")/.."

python src/TriRec_stage1_recall.py
