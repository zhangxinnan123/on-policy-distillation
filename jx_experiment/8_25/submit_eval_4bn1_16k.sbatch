#!/bin/bash
#SBATCH --job-name=eval_4bn1
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
unset ROCR_VISIBLE_DEVICES; unset HIP_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
cd /fsx/xinnanzh/on-policy-distillation
export VAL_MAX_RESPONSE_LENGTH=16384
export STUDENT_MODEL=/fsx/xinnanzh/checkpoints/sft_qwen3_4b_thinking_100k_n1
bash jx_experiment/8_20/eval_sft_1p7b.sh
