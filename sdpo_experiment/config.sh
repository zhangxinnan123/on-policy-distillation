#!/usr/bin/env bash
# Shared config for sdpo_experiment scripts. Sourced by each run_*.sh so the
# wandb credentials and project name live in one place.

# Override via env (e.g. `WANDB_API_KEY=... bash run_all.sh`) if you have your own.
export WANDB_API_KEY="${WANDB_API_KEY:-c4b67c713ad88ef65b62908bcaa8b5c5cb72d1a9}"
export WANDB_ENTITY="${WANDB_ENTITY:-rl_agent}"
export PROJECT_NAME="${PROJECT_NAME:-verl_grpo_sdpo}"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate opd