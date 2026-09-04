#!/bin/bash
# 16k eval of the 8B SFT checkpoint trained on the 100k n=1 dataset (1 epoch).
# Controlled comparison against the n=8 1ep checkpoint (same steps, same schedule).
#SBATCH --job-name=eval_8b_n1_16k
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=6:00:00
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

export VAL_MAX_RESPONSE_LENGTH=16384
export STUDENT_MODEL=/fsx/xinnanzh/checkpoints/sft_qwen3_8b_thinking_100k_n1

bash jx_experiment/8_23/eval_qwen3_8b_think.sh
