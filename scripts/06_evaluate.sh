#!/usr/bin/env bash

# Usage:
#   bash scripts/06_evaluate.sh <stage1_jsonl> <stage2_output_dir>
#   e.g. bash scripts/06_evaluate.sh \
#               log/exp_run01/recommendation_process.jsonl \
#               log_visual/exp_run01/manual_default
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
