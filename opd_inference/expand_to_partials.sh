#!/bin/bash
set -euo pipefail

###############################################################################
# Expand the DAPO train parquet into a partials parquet using a teacher-
# generated full-response JSONL.
#
# Each source row -> NUM_PARTIALS output rows. partial_idx=0 is the empty
# (vanilla) row; partial_idx>=1 carries a token-aligned prefix of the teacher
# response and routes to the partial-continuation agent loop.
#
# Override any variable from the env, e.g.:
#   NUM_PARTIALS=8 bash opd_inference/expand_to_partials.sh
###############################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Activate the project conda env (mirrors jx_experiment/config.sh layout).
if [[ -f "${REPO_ROOT}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/config.env"
fi
if [[ -f "${REPO_ROOT}/config.local.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/config.local.env"
fi
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-sdpo}"
fi

# --- Inputs ------------------------------------------------------------------
GEN_JSONL="${GEN_JSONL:-/projects/standard/mhong/zhan9359/data/sft_gen_output/raw/train/Qwen3-8B/20260502-004952_t0.7_p0.9_results.jsonl}"
BASE_PARQUET="${BASE_PARQUET:-/projects/standard/mhong/zhan9359/verl/data/train-suffix.parquet}"
TEST_IN="${TEST_IN:-/projects/standard/mhong/zhan9359/verl/data/test-suffix.parquet}"

TOKENIZER="${TOKENIZER:-Qwen/Qwen3-1.7B-Base}"
NUM_PARTIALS="${NUM_PARTIALS:-4}"
LIMIT="${LIMIT:-0}"

# --- Output ------------------------------------------------------------------
OUT_DIR="${OUT_DIR:-${HOME}/data/dapo_17k_aime2426-suffix-partial${NUM_PARTIALS}}"
OUT_PARQUET="${OUT_PARQUET:-${OUT_DIR}/train.parquet}"
TEST_OUT="${TEST_OUT:-${OUT_DIR}/test.parquet}"

mkdir -p "${OUT_DIR}"

# --- Run ---------------------------------------------------------------------
echo "============================================================"
echo "==> Expand to partials"
echo "==> gen_jsonl    : ${GEN_JSONL}"
echo "==> base_parquet : ${BASE_PARQUET}"
echo "==> tokenizer    : ${TOKENIZER}"
echo "==> num_partials : ${NUM_PARTIALS}"
echo "==> out_parquet  : ${OUT_PARQUET}"
echo "==> test_in/out  : ${TEST_IN} -> ${TEST_OUT}"
echo "============================================================"

ARGS=(
  --gen_jsonl "${GEN_JSONL}"
  --base_parquet "${BASE_PARQUET}"
  --out_parquet "${OUT_PARQUET}"
  --tokenizer "${TOKENIZER}"
  --num_partials "${NUM_PARTIALS}"
  --test_in "${TEST_IN}"
  --test_out "${TEST_OUT}"
)
if [[ "${LIMIT}" -gt 0 ]]; then
  ARGS+=( --limit "${LIMIT}" )
fi

python opd_inference/expand_to_partials.py "${ARGS[@]}"

echo "[OK] done"
echo "     train: ${OUT_PARQUET}"
echo "     test : ${TEST_OUT}"
