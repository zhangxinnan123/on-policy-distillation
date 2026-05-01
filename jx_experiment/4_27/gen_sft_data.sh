#!/usr/bin/env bash
set -xeuo pipefail
export CUDA_VISIBLE_DEVICES=6
DATA_PATH="${HOME}/data/dapo_17k_aime2426-suffix"
DAPO_TRAIN_PATH="${DATA_PATH}/train.parquet"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-${HOME}/data/sft_gen_output}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16382}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
N="${N:-1}"
LIMIT="${LIMIT:-0}"          # 0 = all
ENABLE_THINKING="${ENABLE_THINKING:-false}"   # match training: enable_thinking=False

SCRIPT_DIR="$(cd "$(dirname "$0")/../../opd_inference" && pwd)"


python "${SCRIPT_DIR}/run_generate_parquet.py" \
    --parquet "${DAPO_TRAIN_PATH}" \
    --model   "${MODEL}" \
    --out_dir "${OUT_DIR}/raw" \
    --max_model_len   "${MAX_MODEL_LEN}" \
    --max_new_tokens  "${MAX_NEW_TOKENS}" \
    --temperature     "${TEMPERATURE}" \
    --top_p           "${TOP_P}" \
    --n               "${N}" \
    --enable_thinking "${ENABLE_THINKING}" \
    $([ "${LIMIT}" -gt 0 ] && echo "--limit ${LIMIT}" || true)

echo "[OK] generation done → ${OUT_DIR}/raw"

# Convert JSONL → SFT parquet
JSONL=$(ls -t "${OUT_DIR}/raw/train/"*_results.jsonl 2>/dev/null | head -1)
if [[ -z "${JSONL}" ]]; then
    # fallback: search one level deeper (model tag subdir)
    JSONL=$(ls -t "${OUT_DIR}/raw/train/"*/*_results.jsonl 2>/dev/null | head -1)
fi

echo "[INFO] converting ${JSONL} → SFT parquet"
python "${SCRIPT_DIR}/convert_to_sft_parquet.py" \
    --input  "${JSONL}" \
    --output "${OUT_DIR}/sft_train.parquet"

echo "[OK] SFT parquet → ${OUT_DIR}/sft_train.parquet"
