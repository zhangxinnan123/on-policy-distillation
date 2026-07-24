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
#SBATCH --output=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/hintRegRawP_l3_%j.out
#SBATCH --error=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/hintRegRawP_l3_%j.err

# LUFFY-style auxiliary hint_reg loss (gradient through shape).

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HINT_LEVEL="${HINT_LEVEL:-level_3}"
export HINT_REG_GAMMA="${HINT_REG_GAMMA:-0.1}"
export HINT_REG_COEF="${HINT_REG_COEF:-1.0}"

bash sdpo_experiment/hint_llm/run_qwen_hintRegRawP_hint_level3.sh "$@"
