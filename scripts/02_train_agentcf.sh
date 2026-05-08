#!/usr/bin/env bash
# Train user / item agent memories via LLM-driven iterative refinement (AgentCF backbone).
# Required / optional environment variables:
#   OPENAI_API_KEY
#   OPENAI_BASE_URL      # (optional) custom gateway
#   MAX_WORKERS          # (optional) LLM concurrency, default 16
#   EXPERIMENT_ID        # (optional) run tag, default timestamp
set -e
cd "$(dirname "$0")/.."

python src/AgentCF.py
