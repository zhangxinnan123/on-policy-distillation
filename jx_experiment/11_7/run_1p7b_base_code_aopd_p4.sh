#!/usr/bin/env bash
############################ AOPD -- advantage-guided routing ############################
# Qwen3-1.7B-Base student <- Qwen3-8B teacher, CodeR1-12k, response length 4096, p4d single node.
#
# Routes on the sign of the per-token advantage A = log pi_T(u) - log pi_S(u):
#   A <  0 -> FKL   the student already assigns the sampled token MORE mass than
#                   the teacher does -- the "overconfident" regime where a
#                   REINFORCE-RKL update pushes it further from the teacher
#   A >= 0 -> PG
# See hybrid_masks.py:_mask_aopd.
#
# Why this arm matters: 8_29/RESULTS.md records aopd as the ONLY mask that
# produced an effect above any noise floor on math (+2.14pp at 1.7B) -- but with
# an unstable sign across student size (-4.01pp at 4B, +0.65pp at 8B). Whether
# that instability reproduces on code is one of the questions this sweep answers.
#
# Shared config (placement, lengths, optimizer, eval wiring) lives in
# _common_code_p4.sh. This file sets ONLY what distinguishes this arm, so the
# eight arms of this sweep are identical by construction except the routing.
#
# Submit:
#   RUN=jx_experiment/11_7/run_1p7b_base_code_aopd_p4.sh \
#     sbatch --export=ALL,RUN=$RUN --job-name=aopd_1p7b jx_experiment/11_7/submit_code_p4.sbatch

STUDENT_MODEL=Qwen/Qwen3-1.7B-Base
SIZE_TAG=1p7b_base
METHOD_TAG=aopd

DISTILLATION_LOSS_MODE="k1_pg_fkl_topk"
HYBRID_MASK_STRATEGY="aopd"

MASK_KWARGS=(
    +distillation.distillation_loss.hybrid_mask_kwargs.threshold=0.0
)

source "$(dirname "${BASH_SOURCE[0]}")/_common_code_p4.sh" "$@"
