#!/bin/bash
#SBATCH --job-name=seller
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

# verl rejects concurrent ROCR/CUDA vars; strip AMD leftovers.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Cluster has no internet — force HF to use local cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /fsx/xinnanzh/on-policy-distillation

bash jx_experiment/8_9/run_8b_base_8b_tent_1p6.sh
