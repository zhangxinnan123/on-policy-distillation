#!/usr/bin/env bash
# Run all 5_26 ablation scripts sequentially. Each leg writes its own wandb
# run via the auto-derived EXP_NAME; shared DATE_TAG + ORIGIN_KL_COEF keep
# the names comparable.
#
# Usage:
#   bash sdpo_experiment/5_26/run_all.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 DATE_TAG=5_27 bash sdpo_experiment/5_26/run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# No shared knob exports — each script honors its own defaults:
#   baseline: pure GRPO, no KL
#   sdpo*:    KL=0.001 origin anchor

# bash "${HERE}/run_qwen_grpo_baseline.sh"
bash "${HERE}/run_qwen_sdpo_grpo_signflip_origin_kl2.sh"
bash "${HERE}/run_qwen_sdpo_grpo_merge_origin_kl.sh"
