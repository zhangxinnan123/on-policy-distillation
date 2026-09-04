#!/bin/bash
# Retry of the 8B SFT on the 100k n=1 dataset (8 nodes x 8 H100).
# The earlier attempt died at step 125/734 before save_steps=200, leaving no
# checkpoint; this config saves every 100 steps.
#SBATCH --job-name=sft_8b_n1
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/xinnanzh/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%j.err

set -xeu

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=8

echo "MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST)"

srun --label bash -c '
set -xeu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_factory

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/fsx/xinnanzh/.tmp
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export WANDB_API_KEY=c4b67c713ad88ef65b62908bcaa8b5c5cb72d1a9
export WANDB_ENTITY=rl_agent
export WANDB_PROJECT=opd_sft
export WANDB_MODE=offline

export NODE_RANK=$SLURM_NODEID
export FORCE_TORCHRUN=1

cd /fsx/xinnanzh/on-policy-distillation/LlamaFactory

echo "[NODE $SLURM_NODEID] MASTER=$MASTER_ADDR:$MASTER_PORT NNODES=$NNODES"

llamafactory-cli train /fsx/xinnanzh/on-policy-distillation/jx_experiment/8_25/sft_qwen3_8b_thinking_100k_n1_retry.yaml
'
