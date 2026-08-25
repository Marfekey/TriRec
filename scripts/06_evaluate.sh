#!/usr/bin/env bash

# Usage:
#   bash scripts/06_evaluate.sh <stage1_jsonl> <stage2_output_dir>
#   e.g. bash scripts/06_evaluate.sh \
#               log/exp_run01/recommendation_process.jsonl \
#               log_visual/exp_run01
#
# <stage2_output_dir> is the directory written by 05_stage2_rerank.sh; it holds
# one *_recommendation_process_alpha<a>.jsonl file per alpha_max value.
set -e
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/06_evaluate.sh <stage1_jsonl> <stage2_output_dir>"
    exit 1
fi

STAGE1_JSONL="$1"
STAGE2_DIR="$2"

python evaluate/data_statistics_platform_first.py  --target_file "$STAGE1_JSONL"
python evaluate/data_statistics_platform_second.py --target_dir  "$STAGE2_DIR"
