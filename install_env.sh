#!/bin/bash
# Run this script from the repo root after cloning:
#   git clone <your-repo-url>
#   cd verl
#   bash install_env.sh

set -e

CONDA_ENV_NAME="opd"

# ── 1. Create and activate conda environment ──────────────────────────────────
echo "==> Creating conda environment: ${CONDA_ENV_NAME} (python 3.12)"
conda create -y -n "${CONDA_ENV_NAME}" python=3.12

# Source conda so 'conda activate' works inside a script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "==> Active Python: $(which python) ($(python --version))"

# ── 2. Install inference frameworks, pytorch, and all dependencies ────────────
# Set USE_MEGATRON=1 to also install TransformerEngine + Megatron-LM
USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-0}
export USE_MEGATRON USE_SGLANG

echo "==> Running scripts/install_vllm_sglang_mcore.sh (USE_MEGATRON=${USE_MEGATRON}, USE_SGLANG=${USE_SGLANG})"
bash scripts/install_vllm_sglang_mcore.sh

# ── 3. Install verl itself (editable, no extra deps to avoid conflicts) ────────
echo "==> Installing verl in editable mode"
pip install --no-deps -e .

echo ""
echo "==> Setup complete. Activate the environment with:"
echo "      conda activate ${CONDA_ENV_NAME}"
