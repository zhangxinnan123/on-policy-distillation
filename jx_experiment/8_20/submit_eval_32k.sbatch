#!/bin/bash
#SBATCH --job-name=eval_32k
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/xinnanzh/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%j.err

set -xeu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

cd /fsx/xinnanzh/on-policy-distillation

# Qwen3-1.7B max_position_embeddings=32768; leave headroom for prompt+special tokens.
# max_model_len = MAX_PROMPT(2048) + VAL_MAX_RESPONSE_LENGTH + 1
# so cap VAL_MAX_RESPONSE_LENGTH at 32768 - 2048 - 1 = 30719
export VAL_MAX_RESPONSE_LENGTH=30719
bash jx_experiment/8_20/eval_sft_1p7b.sh
