#!/bin/bash
set -e

MODEL_PATH="ckpt"
REPO_ID="XinnanZhang/Qwen3-1.7B-Base-Openthought400K-SFT"

python push_model_to_hf.py \
    --model_path "$MODEL_PATH" \
    --repo_id "$REPO_ID"
