#!/bin/bash
# OPD aopd with the REAL hybrid loss (k1_pg_fkl_topk): PG=k1@sampled + supervised=FKL(top-k),
# partitioned per-token by hybrid_mask_strategy. The earlier runs used
# k1_topk_overlap, which ignores hybrid_mask_strategy entirely (mask is
# diagnostic-only there), so they were silently identical to the baseline.
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

set -xeu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Log straight to wandb.ai. Verified reachable from a compute node
# (curl https://api.wandb.ai/ -> HTTP 404, that endpoint's normal response).
export WANDB_MODE=online
export WANDB_ENTITY=rl_agent

# Ray caches the last cluster it joined under /tmp/ray on each node, so leftover state from a
# previous multi-node job can make this single-node launch try to rejoin a dead head (job
# 15997). Start clean.
unset RAY_ADDRESS
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray/ray_current_cluster
# verl's colocated OPD runs vLLM with sleep_mode=True, whose CuMemAllocator uses
# the same CUDA virtual-memory APIs (cuMemCreate/cuMemMap). The two conflict and
# vLLM's WorkerProc fails to initialize.
cd /fsx/xinnanzh/on-policy-distillation

python opd_inference/fix_llamafactory_tokenizer_config.py /fsx/xinnanzh/checkpoints/sft_qwen3_4b_ot400k --also_generation_config

bash jx_experiment/8_31/run_4b_ot400k_8b_opdt4_vote0p5_epslow5_cov0p1.sh
