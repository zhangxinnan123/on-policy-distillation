#!/bin/bash
# OPD aopd with the REAL hybrid loss (k1_pg_fkl_topk): PG=k1@sampled + supervised=FKL(top-k),
# partitioned per-token by hybrid_mask_strategy. The earlier runs used
# k1_topk_overlap, which ignores hybrid_mask_strategy entirely (mask is
# diagnostic-only there), so they were silently identical to the baseline.
#SBATCH --job-name=opd1p7b_aopdfkl
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
# verl's colocated OPD runs vLLM with sleep_mode=True, whose CuMemAllocator uses
# the same CUDA virtual-memory APIs (cuMemCreate/cuMemMap). The two conflict and
# vLLM's WorkerProc fails to initialize.
cd /fsx/xinnanzh/on-policy-distillation

python opd_inference/fix_llamafactory_tokenizer_config.py /fsx/xinnanzh/checkpoints/xinnan_400k_1ep --also_generation_config

bash jx_experiment/8_29/run_1p7b_400k1ep_8b_aopd_fkl.sh
