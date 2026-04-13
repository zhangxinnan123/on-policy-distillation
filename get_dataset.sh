#!/bin/bash
# Download XinnanZhang/DAPO-17k-AIME24-26 from HuggingFace and save as parquet.
#
# Usage:
#   bash get_dataset.sh                        # saves to ~/data/dapo_17k_aime2426/
#   LOCAL_DIR=/your/path bash get_dataset.sh   # custom save path

set -e

HF_DATASET="XinnanZhang/DAPO-17k-AIME24-26"
LOCAL_DIR="${LOCAL_DIR:-${HOME}/data/dapo_17k_aime2426}"
BASE_URL="https://huggingface.co/datasets/${HF_DATASET}/resolve/main"

mkdir -p "${LOCAL_DIR}"

echo "==> Downloading '${HF_DATASET}' → ${LOCAL_DIR}"

wget -nv -O "${LOCAL_DIR}/train.parquet" "${BASE_URL}/train.parquet"
wget -nv -O "${LOCAL_DIR}/test.parquet"  "${BASE_URL}/test.parquet"

echo "==> Done."
ls -lh "${LOCAL_DIR}"
