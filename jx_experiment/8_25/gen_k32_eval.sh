#!/usr/bin/env bash
# Generate K candidates per prompt on the 4-benchmark eval set (aime24 / aime25 /
# aime26 / hmmt_feb_2025), then group into one record per prompt with all K
# candidates and score them.
#
# Sampling: Qwen3 thinking-mode recommended settings
#   temperature=0.6, top_p=0.95, top_k=20, min_p=0
#
# Env vars:
#   MODEL            (required)  HF id or local checkpoint dir
#   K                (default 32)
#   MAX_NEW_TOKENS   (default 16382)
#   PARQUET          (default test_extended_no_amc.parquet — the 4 benchmarks)
#   DP               (default 8) data-parallel vLLM workers, 1 GPU each
#   ENABLE_THINKING  (default true)  true -> thinking mode, false -> non-think
#   OUT_ROOT         (default /fsx/xinnanzh/eval_out/k${K}_${think|nonthink})

set -xeuo pipefail

MODEL="${MODEL:?set MODEL to an HF id or checkpoint dir}"
K="${K:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16382}"
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_MODEL_LEN=$(( MAX_PROMPT + MAX_NEW_TOKENS ))
DP="${DP:-8}"

# aime24 (labelled `aime`, 30) + aime25 (30) + aime26 (30) + hmmt_feb_2025 (30)
PARQUET="${PARQUET:-${HOME}/data/dapo_17k_aime2426-suffix/test_extended_no_amc.parquet}"

ENABLE_THINKING="${ENABLE_THINKING:-true}"
# Keep think / non-think runs in separate trees so they never collide.
if [[ "$ENABLE_THINKING" == "true" ]]; then MODE=think; else MODE=nonthink; fi
OUT_ROOT="${OUT_ROOT:-/fsx/xinnanzh/eval_out/k${K}_${MODE}}"

# Qwen3 thinking-mode sampling
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MIN_P=0.0

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "================== K=${K} generation =================="
echo "MODEL:          $MODEL"
echo "PARQUET:        $PARQUET"
echo "K:              $K"
echo "max_new_tokens: $MAX_NEW_TOKENS  (max_model_len=$MAX_MODEL_LEN)"
echo "sampling:       T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P"
echo "enable_thinking: $ENABLE_THINKING  (mode=$MODE)"
echo "======================================================="

python3 opd_inference/run_generate_parquet.py \
    --parquet "$PARQUET" \
    --model "$MODEL" \
    --out_dir "$OUT_ROOT" \
    --n "$K" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --min_p "$MIN_P" \
    --dp "$DP" \
    --dtype bfloat16 \
    --enable_thinking "$ENABLE_THINKING"

# run_generate_parquet.py writes to OUT_ROOT/<parquet stem>/<model tag>/<stamp>_..._results.jsonl
GEN_DIR="${OUT_ROOT}/$(basename "${PARQUET%.parquet}")/$(basename "${MODEL%/}")"
FLAT_JSONL="$(ls -t "${GEN_DIR}"/*_results.jsonl | head -1)"
echo "[INFO] flat results: $FLAT_JSONL"

# One record per prompt with all K candidates + per-candidate scores, pass@K.
python3 opd_inference/group_candidates.py \
    --input "$FLAT_JSONL" \
    --k "$K" \
    --score \
    --math_verify
