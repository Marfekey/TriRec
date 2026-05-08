#!/usr/bin/env bash
# Pre-train the SASRec sequential model. Used as the upstream retriever to sample
# hard negatives (top-100) for the candidate construction in Stage 1.
# Optional environment variables:
#   CUDA_VISIBLE_DEVICES  # GPU index
set -e
cd "$(dirname "$0")/.."

python src/SASRec/train.py
