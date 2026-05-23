#!/bin/bash
set -euo pipefail

# cd /home/li003968/data/lzy_eval



INPUT_JSONL="${INPUT_JSONL:-sdpo_result/Qwen3-4B/deepmath_diff6to8_verified/Qwen3-4B/20260417-013031_results.jsonl}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B}"
OUT_DIR="${OUT_DIR:-lzy_eval_result/teacher_inference}"
LIMIT="${LIMIT:-0}"          # 0=全量
MAX_LEN="${MAX_LEN:-18000}"
LOGPROBS_K="${LOGPROBS_K:-5}"
# 省事且最稳：默认单卡（即使 CUDA_VISIBLE_DEVICES 给了多卡，也强制 TP=1）
TP_TEACHER="${TP_TEACHER:-1}"
# 省显存：默认 batch_size 小一点，避免一次塞太多序列导致 KV cache/fragmentation OOM
BATCH_SIZE="${BATCH_SIZE:-1}"

if [[ -z "${INPUT_JSONL}" ]]; then
  echo "[ERROR] 请设置 INPUT_JSONL=... (已保存的 prompt/response jsonl)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "==> run_teacher_inference.py (teacher_logp only)"
echo "==> input_jsonl:  ${INPUT_JSONL}"
echo "==> teacher_model:${TEACHER_MODEL}"
echo "==> out_dir:      ${OUT_DIR}"
echo "==> limit:        ${LIMIT}"
echo "==> max_model_len:${MAX_LEN}"
echo "==> tp_teacher:   ${TP_TEACHER}"
echo "==> batch_size:   ${BATCH_SIZE}"
echo "============================================================"

# 给 vLLM 的 KV cache 预留比例降一点，避免“初始化吃满显存 -> 运行时再申请 workspace OOM”
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
# 降低单次 prefill 的 token 数，避免 prompt_logprobs 触发超大 log_softmax 临时张量导致 OOM
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" python sdpo_inference/run_teacher_inference.py \
  --input_jsonl "${INPUT_JSONL}" \
  --teacher_model "${TEACHER_MODEL}" \
  --out_dir "${OUT_DIR}" \
  --limit "${LIMIT}" \
  --max_model_len "${MAX_LEN}" \
  --tp_teacher "${TP_TEACHER}" \
  --batch_size "${BATCH_SIZE}" \
  --logprobs_k "${LOGPROBS_K}" \
  --expert_index 2

echo "[OK] teacher_logp inference finished"

