#!/usr/bin/env bash
# Reward-correctness check, no training: validate Qwen3-1.7B (opd-test-time-scaling's
# code student) on the TTS-format eval subset and dump every generation, so the same
# generations can be re-judged with their EvalPlus / lcb_runner judges
# (compare_with_tts_judge.py). Teacher is loaded only because distillation is on.
STUDENT_MODEL=Qwen/Qwen3-1.7B
SIZE_TAG=1p7b_instruct
METHOD_TAG=valonly
DISTILLATION_LOSS_MODE="k1_topk_overlap"
HYBRID_MASK_STRATEGY="all"
MASK_KWARGS=()
source "$(dirname "${BASH_SOURCE[0]}")/_common_code_p4.sh" trainer.val_only=True "$@"
