#!/usr/bin/env bash
# Download + preprocess XinnanZhang/deepmath-diff6to8-verified into the verl
# GRPO parquet schema. Output: <LOCAL_DIR>/train.parquet (plus extra columns
# `generations_wo_think` carrying the privileged expert trajectories used by
# SDPO).
#
# Usage:
#   bash examples/data_preprocess/download_deepmath_diff6to8.sh [local_dir]
#
# Default local_dir = ~/data/deepmath_diff6to8 (matches TRAIN_DATA_DIR default
# in sdpo_experiment/5_26/*.sh).
set -euo pipefail

LOCAL_DIR="${1:-${HOME}/data/deepmath_diff6to8}"

mkdir -p "$LOCAL_DIR"

python examples/data_preprocess/deepmath_diff6to8.py \
    --local_save_dir "$LOCAL_DIR"

echo "Preprocessed parquet written to: $LOCAL_DIR"
ls -lh "$LOCAL_DIR"
