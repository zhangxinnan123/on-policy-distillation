#!/bin/bash
set -euo pipefail

# Generate Qwen3-4B trajectories with and without the privileged "expert trajectory"
# prompt augmentation from run_teacher_inference_v2.py, then score with math_boxed.

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
HF_DATASET="${HF_DATASET:-XinnanZhang/deepmath-diff6to8-verified}"
HF_SPLIT="${HF_SPLIT:-train}"
LIMIT="${LIMIT:-50}"
EXPERT_INDEX="${EXPERT_INDEX:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TP="${TP:-0}"
OUT_DIR="${OUT_DIR:-./sdpo_result/compare_privileged}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python sdpo_inference/run_compare_privileged.py \
  --hf_dataset "${HF_DATASET}" \
  --hf_split   "${HF_SPLIT}" \
  --model      "${MODEL}" \
  --limit      "${LIMIT}" \
  --expert_index "${EXPERT_INDEX}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --tp "${TP}" \
  --out_dir "${OUT_DIR}" \
  --enforce_eager
