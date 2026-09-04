#!/bin/bash
#SBATCH --job-name=sft_gen_90k
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/xinnanzh/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%j.err

set -xeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /fsx/xinnanzh/on-policy-distillation

# 90k samples: skip first 1250 prompts (covered by 14719 → 10k), take next 11250
# 11250 × 8 rollouts = 90000
export START=1250
export LIMIT=11250

bash jx_experiment/8_10/gen_sft_data.sh
