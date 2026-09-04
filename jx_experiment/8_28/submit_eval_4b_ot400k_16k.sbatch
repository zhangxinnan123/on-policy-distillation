#!/bin/bash
# 16k eval of the 4b SFT checkpoint trained on openthoughts3-math-50k8 (400k +suffix).
# Applies the LlamaFactory tokenizer_config fix first: the training env writes
# extra_special_tokens as a list (transformers 5.2.0) but the inference env
# (4.55.2) needs a dict, otherwise AutoTokenizer raises AttributeError.
#SBATCH --job-name=eval_4b_ot400k_16k
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

CKPT=/fsx/xinnanzh/checkpoints/sft_qwen3_4b_ot400k

# Idempotent; no-op if the checkpoint is already correct.
python opd_inference/fix_llamafactory_tokenizer_config.py "$CKPT" --also_generation_config

export VAL_MAX_RESPONSE_LENGTH=16384
export STUDENT_MODEL="$CKPT"

bash jx_experiment/8_23/eval_qwen3_8b_think.sh
