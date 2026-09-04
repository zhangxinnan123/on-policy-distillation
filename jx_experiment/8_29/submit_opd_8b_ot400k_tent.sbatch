#!/bin/bash
# OPD tent — student = sft_qwen3_8b_ot400k (8B SFT on openthoughts3 400k), teacher = Qwen3-8B.
# Student and teacher are the same size here, so SP=2 and gpu_mem_util=0.4.
#SBATCH --job-name=opd8b_tent
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
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

# Saved by transformers 5.2.0 -> rope_theta is nested and this env would fall back
# to 10000 instead of 1000000. Idempotent.
python opd_inference/fix_llamafactory_tokenizer_config.py \
    /fsx/xinnanzh/checkpoints/sft_qwen3_8b_ot400k --also_generation_config

bash jx_experiment/8_29/run_8b_ot400k_8b_tent.sh
