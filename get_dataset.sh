#!/bin/bash
# Download XinnanZhang/DAPO-17k-AIME24-26 from HuggingFace and save as parquet.
#
# Usage:
#   bash get_dataset.sh                        # saves to ~/data/dapo_17k_aime2426/
#   LOCAL_DIR=/your/path bash get_dataset.sh   # custom save path

set -e
# HOME=/projects/standard/mhong/zhan9359
HF_DATASET="XinnanZhang/DAPO-17k-AIME24-26-suffix"
LOCAL_DIR="${LOCAL_DIR:-${HOME}/data/dapo_17k_aime2426-suffix}"
BASE_URL="https://huggingface.co/datasets/${HF_DATASET}/resolve/main"

mkdir -p "${LOCAL_DIR}"

echo "==> Downloading '${HF_DATASET}' → ${LOCAL_DIR}"

wget -nv -O "${LOCAL_DIR}/train.parquet" "${BASE_URL}/data/train-00000-of-00001.parquet"
wget -nv -O "${LOCAL_DIR}/test.parquet"  "${BASE_URL}/data/test-00000-of-00001.parquet"

echo "==> Done."
ls -lh "${LOCAL_DIR}"
