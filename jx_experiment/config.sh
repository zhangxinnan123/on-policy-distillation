#!/usr/bin/env bash
# Unified config for jx_experiment scripts.
# Edit this file to set your credentials and project settings.

WANDB_API_KEY="c4b67c713ad88ef65b62908bcaa8b5c5cb72d1a9"
export WANDB_ENTITY="${WANDB_ENTITY:-rl_agent}"
PROJECT_NAME="verl_opd_dapo"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate opd
