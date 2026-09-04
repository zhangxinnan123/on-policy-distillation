#!/usr/bin/env bash
# Generate SFT training data with Qwen3-8B (thinking mode) on OpenThoughts3-1.2M.
#
# Uses the project's preferred chat template (Qwen3 with enable_thinking=True).

set -xeuo pipefail

# ---------------- Config ----------------
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
HF_DATASET="${HF_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
SPLIT="${SPLIT:-train}"
OUT_DIR="${OUT_DIR:-/fsx/xinnanzh/data/sft_gen}"

# Qwen3 thinking-mode recommended sampling
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
N_PER_PROMPT="${N_PER_PROMPT:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16382}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"   # prompt(~2k) + response(16382) with headroom

START="${START:-0}"
# 10k SFT samples: LIMIT prompts × N_PER_PROMPT rollouts = 1250 × 8 = 10000
LIMIT="${LIMIT:-1250}"
DP="${DP:-8}"
TP="${TP:-1}"

# Qwen3 thinking mode (CoT)
ENABLE_THINKING="${ENABLE_THINKING:-true}"

# Suffix appended to user prompt (matches train.parquet format).
# Single-quote the default so bash doesn't eat the '{}'.
_DEFAULT_SUFFIX='Please reason step by step, and put your final answer within \boxed{}.'
USER_SUFFIX="${USER_SUFFIX:-$_DEFAULT_SUFFIX}"

# Keep only rows whose 'domain' == this value (OpenThoughts3 has math/code/science/etc.)
DOMAIN_FILTER="${DOMAIN_FILTER:-math}"

# ---------------- Env ----------------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------- Args ----------------
LIMIT_ARG=""
if [[ -n "${LIMIT}" ]]; then LIMIT_ARG="--limit ${LIMIT}"; fi

THINKING_ARG=""
if [[ -n "${ENABLE_THINKING}" ]]; then THINKING_ARG="--enable_thinking ${ENABLE_THINKING}"; fi

# ---------------- Run ----------------
echo "================== SFT data generation =================="
echo "Teacher model : ${TEACHER_MODEL}"
echo "HF dataset    : ${HF_DATASET} (split=${SPLIT})"
echo "Output dir    : ${OUT_DIR}"
echo "Sampling      : T=${TEMPERATURE}, top_p=${TOP_P}, top_k=${TOP_K}, min_p=${MIN_P}, n=${N_PER_PROMPT}"
echo "Length        : max_new=${MAX_NEW_TOKENS}, max_model_len=${MAX_MODEL_LEN}"
echo "Slice         : start=${START}, limit=${LIMIT:-<all>}"
echo "Parallel      : dp=${DP}, tp=${TP}"
echo "Thinking mode : ${ENABLE_THINKING}"
echo "========================================================="

cd "${REPO_ROOT}"

python opd_inference/run_generate_hf.py \
    --hf_dataset "${HF_DATASET}" \
    --split "${SPLIT}" \
    --model "${TEACHER_MODEL}" \
    --out_dir "${OUT_DIR}" \
    --start "${START}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --min_p "${MIN_P}" \
    --n "${N_PER_PROMPT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --tp "${TP}" \
    --dp "${DP}" \
    --dtype bfloat16 \
    --user_suffix "${USER_SUFFIX}" \
    --domain_filter "${DOMAIN_FILTER}" \
    ${THINKING_ARG} \
    ${LIMIT_ARG}

echo "================== Done =================="
