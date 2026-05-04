#!/bin/bash
set -euo pipefail

###############################################################################
# Two-step pipeline:
#   1) prepare_partials_file.py  -> bundle JSON (prompt + truncated partials
#                                    with original_continuation for diffing)
#   2) run_continue_from_partial.py --input_file BUNDLE -> continuation JSONL
#
# Override any variable from the env, e.g.:
#   ROW_IDX=8 TRUNCATE_FRACS="0,0.25,0.5,0.75" bash opd_inference/run_continue_from_partial.sh
###############################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# --- Inputs --------------------------------------------------------------------
SOURCE_JSONL="${SOURCE_JSONL:-lzy_eval_result/teacher_inference/20260419-214728_results/20260419-220251_Qwen3-4B_scored.jsonl}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-4B}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"

# Row selection: pick exactly one of these two paths.
# - If PICK_LONGEST=1: auto-select the row with the most response tokens
#   (filtered by MIN_TOKENS).
# - Otherwise use ROW_IDX.
PICK_LONGEST="${PICK_LONGEST:-0}"
ROW_IDX="${ROW_IDX:-9}"
MIN_TOKENS="${MIN_TOKENS:-0}"

# Truncation cutoffs. Default uses paragraph boundaries only (snap to "\n\n").
# Set TRUNCATE_TOKENS / TRUNCATE_FRACS to mix in token-count or fractional cutoffs.
# Paragraphs are 1-based indices of "\n\n" boundaries; "all" picks every boundary.
TRUNCATE_TOKENS="${TRUNCATE_TOKENS:-}"
TRUNCATE_FRACS="${TRUNCATE_FRACS:-}"
TRUNCATE_PARAGRAPHS="${TRUNCATE_PARAGRAPHS:-60,80,120,140,160,180}"

# --- Output paths --------------------------------------------------------------
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
BUNDLE_DIR="${BUNDLE_DIR:-opd_inference/continue_inputs}"
BUNDLE_FILE="${BUNDLE_FILE:-${BUNDLE_DIR}/bundle_${RUN_TAG}.json}"
OUT_DIR="${OUT_DIR:-opd_inference/continue_results}"

mkdir -p "${BUNDLE_DIR}" "${OUT_DIR}"

# --- Generation params ---------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
N_SAMPLES="${N_SAMPLES:-1}"
LOGPROBS_K="${LOGPROBS_K:-0}"
SEED="${SEED:-}"
TP="${TP:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Optional: list rows then exit --------------------------------------------
# LIST=1 SOURCE_JSONL=... bash opd_inference/run_continue_from_partial.sh
if [[ "${LIST:-0}" == "1" ]]; then
  echo "[INFO] listing rows in ${SOURCE_JSONL} (longest first)"
  python opd_inference/prepare_partials_file.py \
    --source_jsonl "${SOURCE_JSONL}" \
    --list \
    --list_top "${LIST_TOP:-20}" \
    ${MIN_TOKENS:+--min_tokens "${MIN_TOKENS}"}
  exit 0
fi

# --- Optional: list "\n\n" paragraph boundaries for the chosen row, then exit -
# LIST_PARAGRAPHS=1 ROW_IDX=7 bash opd_inference/run_continue_from_partial.sh
if [[ "${LIST_PARAGRAPHS:-0}" == "1" ]]; then
  echo "[INFO] listing paragraph boundaries for row_idx=${ROW_IDX}"
  python opd_inference/prepare_partials_file.py \
    --source_jsonl "${SOURCE_JSONL}" \
    --tokenizer "${TOKENIZER}" \
    --row_idx "${ROW_IDX}" \
    --list_paragraphs
  exit 0
fi

###############################################################################
# Step 1: build the bundle (prompt + partials + original continuations).
###############################################################################
echo "============================================================"
echo "==> Step 1: prepare bundle"
echo "==> source : ${SOURCE_JSONL}"
echo "==> picker : $([[ "${PICK_LONGEST}" == "1" ]] && echo "longest (min_tokens=${MIN_TOKENS})" || echo "row_idx=${ROW_IDX}")"
echo "==> bundle : ${BUNDLE_FILE}"
echo "============================================================"

PREP_ARGS=(
  --source_jsonl "${SOURCE_JSONL}"
  --tokenizer "${TOKENIZER}"
  --out "${BUNDLE_FILE}"
)
if [[ "${PICK_LONGEST}" == "1" ]]; then
  PREP_ARGS+=( --pick_longest )
  if [[ "${MIN_TOKENS}" -gt 0 ]]; then
    PREP_ARGS+=( --min_tokens "${MIN_TOKENS}" )
  fi
else
  PREP_ARGS+=( --row_idx "${ROW_IDX}" )
fi
if [[ -n "${TRUNCATE_TOKENS}" ]]; then
  PREP_ARGS+=( --truncate_tokens "${TRUNCATE_TOKENS}" )
fi
if [[ -n "${TRUNCATE_FRACS}" ]]; then
  PREP_ARGS+=( --truncate_fracs "${TRUNCATE_FRACS}" )
fi
if [[ -n "${TRUNCATE_PARAGRAPHS}" ]]; then
  PREP_ARGS+=( --truncate_paragraphs "${TRUNCATE_PARAGRAPHS}" )
fi

python opd_inference/prepare_partials_file.py "${PREP_ARGS[@]}"

###############################################################################
# Step 2: continue generation from each partial.
###############################################################################
echo "============================================================"
echo "==> Step 2: continue from partials"
echo "==> model        : ${MODEL}"
echo "==> bundle       : ${BUNDLE_FILE}"
echo "==> out_dir      : ${OUT_DIR}"
echo "==> CUDA devices : ${CUDA_VISIBLE_DEVICES}  (TP=${TP})"
echo "==> sampling     : T=${TEMPERATURE} top_p=${TOP_P} n=${N_SAMPLES} max_new=${MAX_NEW_TOKENS}"
echo "============================================================"

GEN_ARGS=(
  --model "${MODEL}"
  --input_file "${BUNDLE_FILE}"
  --out_dir "${OUT_DIR}"
  --max_model_len "${MAX_MODEL_LEN}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --n "${N_SAMPLES}"
  --logprobs_k "${LOGPROBS_K}"
  --tp "${TP}"
)
if [[ -n "${SEED}" ]]; then
  GEN_ARGS+=( --seed "${SEED}" )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python opd_inference/run_continue_from_partial.py "${GEN_ARGS[@]}"

###############################################################################
# Step 3 (optional): build a self-contained HTML viewer that interleaves the
# original response with inline buttons that toggle each model continuation.
# Set VISUALIZE=0 to skip.
###############################################################################
if [[ "${VISUALIZE:-1}" == "1" ]]; then
  MODEL_TAG="$(basename "${MODEL%/}")"
  CONT_DIR="${OUT_DIR}/${MODEL_TAG}"
  CONT_FILE="${CONT_FILE:-$(ls -t ${CONT_DIR}/*_continue.jsonl 2>/dev/null | head -n 1)}"
  HTML_OUT="${HTML_OUT:-${OUT_DIR}/${MODEL_TAG}/viewer_${RUN_TAG}.html}"
  if [[ -z "${CONT_FILE}" || ! -f "${CONT_FILE}" ]]; then
    echo "[WARN] no continuation JSONL found under ${CONT_DIR}; skipping HTML."
  else
    echo "============================================================"
    echo "==> Step 3: build HTML viewer"
    echo "==> bundle       : ${BUNDLE_FILE}"
    echo "==> continuations: ${CONT_FILE}"
    echo "==> html         : ${HTML_OUT}"
    echo "============================================================"
    python opd_inference/visualize_partials.py \
      --bundle "${BUNDLE_FILE}" \
      --continuations "${CONT_FILE}" \
      --out "${HTML_OUT}" \
      --title "row=${ROW_IDX} · model=${MODEL_TAG}"
  fi
fi

echo "[OK] done"
echo "     bundle : ${BUNDLE_FILE}"
echo "     out_dir: ${OUT_DIR}"
