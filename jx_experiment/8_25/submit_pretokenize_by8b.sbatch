#!/bin/bash
#SBATCH --job-name=pretok_8b
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=1:30:00
#SBATCH --output=/fsx/xinnanzh/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%j.err
set -xeu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_factory
unset ROCR_VISIBLE_DEVICES; unset HIP_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/fsx/xinnanzh/.tmp
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export WANDB_MODE=disabled
cd /fsx/xinnanzh/on-policy-distillation/LlamaFactory
llamafactory-cli train /fsx/xinnanzh/on-policy-distillation/jx_experiment/8_25/pretokenize_n8_by8b.yaml
