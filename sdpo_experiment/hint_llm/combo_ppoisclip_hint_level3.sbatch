#!/bin/bash
#SBATCH --job-name=seller
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/combo_ppoisclip_l3_%j.out
#SBATCH --error=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/combo_ppoisclip_l3_%j.err

# ============================================================================
# Full SDPO + PPO min-clip IS combo.
#   sign-flip λ=0.1 on Regime B, PPO min-clip [0.8, 1.28] on Regime C hint rows,
#   regime_c_scale=0.25.
# ============================================================================

set -euo pipefail

PROJECT_DIR="/fsx/xinnanzh/sdpo"
CONDA_ENV="${CONDA_ENV:-opd}"

cd "${PROJECT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "========== ENV =========="
echo "job_id=${SLURM_JOB_ID:-?} node=$(hostname) partition=${SLURM_JOB_PARTITION:-?}"
echo "conda_env=${CONDA_ENV}"
echo "python=$(which python) ($(python --version 2>&1))"
nvidia-smi -L
echo "========================="

unset ROCR_VISIBLE_DEVICES
export HINT_LEVEL="${HINT_LEVEL:-level_3}"
export SDPO_SIGN_FLIP_LAMBDA_POS="${SDPO_SIGN_FLIP_LAMBDA_POS:-0.1}"
export SDPO_SIGN_FLIP_LAMBDA_NEG="${SDPO_SIGN_FLIP_LAMBDA_NEG:-0.1}"
export PPO_CLIP_LOW="${PPO_CLIP_LOW:-0.2}"
export PPO_CLIP_HIGH="${PPO_CLIP_HIGH:-0.28}"

bash sdpo_experiment/hint_llm/run_qwen_combo_ppoisclip_hint_level3.sh "$@"
