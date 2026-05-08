#!/usr/bin/env bash
# Build item semantic embeddings, training/test exposure JSONs, and similarity Top-K logs
# Required / optional environment variables:
#   OPENAI_API_KEY        # OpenAI API key
#   OPENAI_BASE_URL       # (optional) custom gateway
#   SBERT_MODEL_PATH      # (optional) local Sentence-BERT path; default: all-MiniLM-L6-v2
set -e
cd "$(dirname "$0")/.."

python dataprocess/generate_embeddings.py
python dataprocess/item_exposure_counter.py
python dataprocess/compute_test_item_exposure_sentence-bert.py
python dataprocess/merge_exposure_json.py
