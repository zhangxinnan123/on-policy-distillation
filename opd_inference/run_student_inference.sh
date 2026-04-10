#!/bin/bash
set -euo pipefail

# cd /home/li003968/data/lzy_eval
# python -m py_compile run_generate_aime24_with_logpi.py
# echo "[OK] py_compile passed: run_generate_aime24_with_logpi.py"
# python -m py_compile run_generate_parquet_with_logpi.py
# echo "[OK] py_compile passed: run_generate_parquet_with_logpi.py"

DEFAULT_BASE_DIR="/mnt/data1/li003968/onpolicy_verl/qwen3_4bSFT_32b_opd_8192_no_whiten"
ANALYSE_DIR="${ANALYSE_DIR:-/mnt/data1/li003968/onpolicydistillation_analyse}"
OUT_DIR="${OUT_DIR:-${ANALYSE_DIR}/parquet_logpi_results}"
DATASET_PATH="${DATASET_PATH:-/home/li003968/data/2080/math__combined.parquet}"

# mkdir -p "${ANALYSE_DIR}"
# mkdir -p "${OUT_DIR}"

RUN_ALL_MERGED="${RUN_ALL_MERGED:-1}"
BASE_DIR="${BASE_DIR:-${DEFAULT_BASE_DIR}}"
MODEL_NAME="${MODEL_NAME:-lzy337/lzy-qwen3-4b-base-sft-openthoughts3}"
MAX_LEN="${MAX_LEN:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
LIMIT="${LIMIT:-1000000}"
LOGPROBS_K="${LOGPROBS_K:-20}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
TP="${TP:-4}"

# if [[ "${RUN_ALL_MERGED}" == "1" || "${RUN_ALL_MERGED,,}" == "true" || "${RUN_ALL_MERGED,,}" == "yes" ]]; then
#   if [[ ! -d "${BASE_DIR}" ]]; then
#     echo "[ERROR] BASE_DIR not found: ${BASE_DIR}" >&2
#     exit 1
#   fi

#   # 按 AIME 那套循环结构：每隔 STEP_INTERVAL 跑一次
#   START_STEP="${START_STEP:-250}"
#   END_STEP="${END_STEP:-1200}"
#   STEP_INTERVAL="${STEP_INTERVAL:-50}"

#   for ((STEP=${START_STEP}; STEP<=${END_STEP}; STEP+=${STEP_INTERVAL})); do
#     STEP_DIR="global_step_${STEP}"
#     STEP_PATH="${BASE_DIR}/${STEP_DIR}"

#     echo "============================================================"
#     echo "==> Processing checkpoint: ${STEP_DIR}"
#     echo "============================================================"

#     if [[ ! -d "${STEP_PATH}" ]]; then
#       echo "[WARNING] step dir missing, skip: ${STEP_PATH}"
#       continue
#     fi

#     # 兼容两种 merge 输出结构：
#     # - ${BASE_DIR}/global_step_XXX/hf_merged
#     # - ${BASE_DIR}/global_step_XXX/actor/hf_merged
#     MERGED_DIR=""
#     if [[ -f "${STEP_PATH}/hf_merged/config.json" ]]; then
#       MERGED_DIR="${STEP_PATH}/hf_merged"
#     elif [[ -f "${STEP_PATH}/actor/hf_merged/config.json" ]]; then
#       MERGED_DIR="${STEP_PATH}/actor/hf_merged"
#     fi

#     if [[ -z "${MERGED_DIR}" ]]; then
#       echo "[WARNING] merged HF model not found, skip: ${STEP_PATH}"
#       continue
#     fi

#     STEP_OUT_DIR="${OUT_DIR}/${STEP_DIR}"

#     echo "============================================================"
#     echo "==> Parquet logp/rank for: ${STEP_DIR}"
#     echo "==> model: ${MERGED_DIR}"
#     echo "==> out:   ${STEP_OUT_DIR}"
#     echo "============================================================"
#     mkdir -p "${STEP_OUT_DIR}"

#     CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python run_generate_parquet_with_logpi.py \
#       --parquet "${DATASET_PATH}" \
#       --model "${MERGED_DIR}" \
#       --out_dir "${STEP_OUT_DIR}" \
#       --max_model_len "${MAX_LEN}" \
#       --logprobs_k "${LOGPROBS_K}" \
#       --tp "${TP}" \
#       --limit "${LIMIT}" \
#       --batch
#   done
# else
#   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python run_generate_parquet_with_logpi.py \
#     --parquet "${DATASET_PATH}" \
#     --model "${MODEL_NAME}" \
#     --out_dir "${OUT_DIR}" \
#     --max_model_len "${MAX_LEN}" \
#     --logprobs_k "${LOGPROBS_K}" \
#     --tp "${TP}" \
#     --limit "${LIMIT}" \
#     --batch
# fi

# echo "[OK] parquet logp eval finished"

###############################################################################
# SFT-only: 只用你自己的 SFT 模型生成一次，并保存每个 token 的 logpi + rank
# 用法示例：
#   RUN_SFT_ONLY=1 \
#   SFT_MODEL_PATH="/path/to/your/sft_model_or_hf_merged" \
#   DATASET_PATH="/home/li003968/data/2080/math__combined.parquet" \
#   OUT_DIR="/mnt/data1/li003968/onpolicydistillation_analyse/parquet_logpi_results_sft" \
#   CUDA_VISIBLE_DEVICES=6 \
#   bash /home/li003968/data/stastic_test.sh
###############################################################################

# DATASET_PATH=/projects/standard/mhong/zhan9359/verl/data/aime-eval.parquet
DATASET_PATH=/projects/standard/mhong/zhan9359/verl/data/aime-eval-problem-only.parquet 
OUT_DIR="/projects/standard/mhong/zhan9359/work/on-policy-distillation/verl/lzy_eval_result/SFT-4B"
# OUT_DIR="/home/li003968/data/lzy_eval_result/SFT-4B"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "==> SFT-only parquet logp/rank"
echo "==> model: lzy337/lzy-qwen3-4b-base-sft-openthoughts3"
echo "==> out:   ${OUT_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_generate_parquet_with_logpi.py \
  --parquet "${DATASET_PATH}" \
  --model "lzy337/lzy-qwen3-4b-base-sft-openthoughts3" \
  --out_dir "${OUT_DIR}" \
  --max_model_len "${MAX_LEN}" \
  --logprobs_k "${LOGPROBS_K}" \
  --tp "${TP}" \
  --limit "${LIMIT}" \
  --batch \
  --n 32

echo "[OK] SFT-only parquet logp eval finished"
exit 0