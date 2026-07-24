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
#SBATCH --output=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/sdpo_hint_l3_%j.out
#SBATCH --error=/fsx/xinnanzh/sdpo/sdpo_experiment/hint_llm/logs/sdpo_hint_l3_%j.err

# ============================================================================
# SDPO+GRPO with Claude-extracted hints (level_3) on deepmath_diff6to8.
#   Wraps sdpo_experiment/hint_llm/run_qwen_sdpo_grpo_merge_is_hint_level3.sh.
#   Override the hint level or any script env var:
#     HINT_LEVEL=level_2 sbatch sdpo_experiment/hint_llm/sdpo_grpo_merge_is_hint_level3.sbatch
# ============================================================================

set -euo pipefail

PROJECT_DIR="/fsx/xinnanzh/sdpo"
CONDA_ENV="${CONDA_ENV:-opd}"

cd "${PROJECT_DIR}"

# Activate conda env (install_env.sh -> opd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# Show environment for debugging.
echo "========== ENV =========="
echo "job_id=${SLURM_JOB_ID:-?} node=$(hostname) partition=${SLURM_JOB_PARTITION:-?}"
echo "conda_env=${CONDA_ENV}"
echo "python=$(which python) ($(python --version 2>&1))"
nvidia-smi -L
echo "========================="

# Ensure export vars visible to the launched script (STUDENT_WORLD_SIZE etc.
# stay at the script's defaults unless the user overrides).
unset ROCR_VISIBLE_DEVICES
export HINT_LEVEL="${HINT_LEVEL:-level_3}"

# Launch. All positional args passed through as extra Hydra overrides.
bash sdpo_experiment/hint_llm/run_qwen_sdpo_grpo_merge_is_hint_level3.sh "$@"
