#!/usr/bin/env bash
# Data preprocessing: raw JSONL -> CSV -> filter meta -> filter inter -> init user/item memory
# Prerequisite: place Amazon Reviews raw files under dataset/org_data/{inter,meta}/
set -e
cd "$(dirname "$0")/.."

python dataprocess/trans.py
python dataprocess/filter_meta_data.py
python dataprocess/filter_inter_data.py
python dataprocess/DataPrepare.py
