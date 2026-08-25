#!/usr/bin/env bash
# TriRec Stage 2: Platform sequential re-ranking.
# Ranks candidates sequentially (one position at a time), balancing immediate
# user relevance, platform-level fairness, and expected item utility.
#
# Usage:
#   bash scripts/05_stage2_rerank.sh <stage1_output.jsonl> [extra args...]
#   e.g. bash scripts/05_stage2_rerank.sh log/exp_run01/recommendation_process.jsonl \
#               --lambda1 0.5 --lambda2 0.5 --lambda_item 10
#
# Optional environment variables:
#   CAND_NUM             # |C_u|, default 10
set -e
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/05_stage2_rerank.sh <stage1_output.jsonl> [args...]"
    exit 1
fi

INPUT_JSONL="$1"
shift

python src/TriRec_stage2_rerank.py --input_path "$INPUT_JSONL" "$@"
