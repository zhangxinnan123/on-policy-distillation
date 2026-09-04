#!/bin/bash
# 16k eval of the 2-epoch 8B SFT checkpoint (100k n=8 dataset), thinking mode.
# Array 0 -> test.parquet         (aime24/25/26 + amc23)  — comparable to prior 16k evals
# Array 1 -> test_extended_no_amc (aime24/25/26 + hmmt)   — the 4-benchmark set
#SBATCH --job-name=eval_8b_2ep_16k
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=/fsx/xinnanzh/logs/%x_%A_%a.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%A_%a.err

set -xeu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

cd /fsx/xinnanzh/on-policy-distillation

PARQUETS=(test.parquet test_extended_no_amc.parquet)
export TEST_PARQUET="${PARQUETS[$SLURM_ARRAY_TASK_ID]}"

export VAL_MAX_RESPONSE_LENGTH=16384
export STUDENT_MODEL=/fsx/xinnanzh/checkpoints/sft_qwen3_8b_thinking_100k_2ep

echo "ARRAY TASK ${SLURM_ARRAY_TASK_ID}: TEST_PARQUET=${TEST_PARQUET}"

bash jx_experiment/8_23/eval_qwen3_8b_think.sh
