#!/bin/bash
# K=32 candidate generation on aime24 / aime25 / aime26 / hmmt_feb_2025.
# Array task 0 -> Qwen3-1.7B (thinking), task 1 -> Qwen3-8B (thinking).
#SBATCH --job-name=gen_k32
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/fsx/xinnanzh/logs/%x_%A_%a.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%A_%a.err

set -xeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /fsx/xinnanzh/on-policy-distillation

MODELS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

export K=32
export MAX_NEW_TOKENS=16382
export DP=8

echo "ARRAY TASK ${SLURM_ARRAY_TASK_ID}: MODEL=${MODEL}"

bash jx_experiment/8_25/gen_k32_eval.sh
