#!/bin/bash
#SBATCH --job-name=sft_gen_100k_n1
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=/fsx/xinnanzh/logs/%x_%A_%a.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%A_%a.err

set -xeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /fsx/xinnanzh/on-policy-distillation

# 100k unique prompts, n=1 rollout each, sharded to 8 nodes.
# Each shard: 12500 prompts starting at start_offset = i * 12500
SHARD_ID=${SLURM_ARRAY_TASK_ID}
SHARD_SIZE=12500
export START=$(( SHARD_ID * SHARD_SIZE ))
export LIMIT=${SHARD_SIZE}

# One rollout per prompt
export N_PER_PROMPT=1

# Distinct output dir per shard so files don't collide
# (run_generate_hf.py writes to OUT_DIR/<hf_dataset>/<model>/<timestamp>_...jsonl,
#  but timestamp granularity is minutes so shards launched together might share
#  a filename — write to per-shard subdir to be safe)
export OUT_DIR=/fsx/xinnanzh/data/sft_gen_100k_n1/shard_${SHARD_ID}

echo "SHARD ${SHARD_ID}: START=${START}, LIMIT=${LIMIT}, N=${N_PER_PROMPT}, OUT_DIR=${OUT_DIR}"

bash jx_experiment/8_10/gen_sft_data.sh
