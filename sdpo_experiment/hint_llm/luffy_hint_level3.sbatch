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
#SBATCH --output=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/luffy_l3_%j.out
#SBATCH --error=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/luffy_l3_%j.err

# ============================================================================
# LUFFY-only ablation: p/(p+c) reshape on hint rows, NO SDPO advantage sign-flip.
#   Wraps sdpo_experiment/hint_llm/run_qwen_luffy_hint_level3.sh.
#
#   Override LUFFY c:
#     LUFFY_C=0.3 sbatch sdpo_experiment/hint_llm/luffy_hint_level3.sbatch
#
#   Override hint level:
#     HINT_LEVEL=level_2 sbatch sdpo_experiment/hint_llm/luffy_hint_level3.sbatch
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
export LUFFY_C="${LUFFY_C:-0.1}"

bash sdpo_experiment/hint_llm/run_qwen_luffy_hint_level3.sh "$@"
