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
#SBATCH --output=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/isclip_l3_%j.out
#SBATCH --error=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/isclip_l3_%j.err

# ============================================================================
# IS-clip only (Setting A): hint rows weighted by clip(w, lo, hi), no LUFFY
# shaping, no sign-flip. Compare against run_qwen_luffy_* (Setting B).
#
# Override clip bounds:
#   IS_CLIP_LOW=0.05 IS_CLIP_HIGH=20 sbatch ...
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
export IS_CLIP_LOW="${IS_CLIP_LOW:-0.8}"
export IS_CLIP_HIGH="${IS_CLIP_HIGH:-1.28}"

bash sdpo_experiment/hint_llm/run_qwen_isclip_hint_level3.sh "$@"
