#!/usr/bin/env bash
############################ OPD baseline -- all-PG reverse KL ############################
# Qwen3-1.7B-Base student <- Qwen3-8B teacher, CodeR1-12k, response length 4096, p4d single node.
#
# The baseline every other arm is measured against: plain k1 policy gradient on
# EVERY response token, i.e. single-sample reverse KL with no supervised arm.
# This is jx_experiment/8_9/run_8b_base_8b_baseline_all_pg.sh's configuration,
# the baseline all the math comparisons in 8_29/RESULTS.md are made against.
#
# loss_mode is k1_topk_overlap and mask is "all". Those two agree here -- under a
# non-hybrid loss mode the mask does not route (losses.py only consults it via
# _compute_pg_diag_for_non_hybrid), and "all" would send every token to PG
# anyway, so the mask is redundant rather than inert-and-wrong. It is set
# explicitly so the arm reads as a deliberate all-PG run.
#
# NOTE on reading wandb: actor/distillation/pg_token_ratio is a diagnostic over
# an unset mask for this loss mode and will read ~0, NOT 100. Do not report this
# arm as "0% PG"; it is 100% PG by construction.
#
# Shared config (placement, lengths, optimizer, eval wiring) lives in
# _common_code_p4.sh. This file sets ONLY what distinguishes this arm, so the
# eight arms of this sweep are identical by construction except the routing.
#
# Submit:
#   RUN=jx_experiment/11_7/run_1p7b_base_code_opd_p4.sh \
#     sbatch --export=ALL,RUN=$RUN --job-name=opd_1p7b jx_experiment/11_7/submit_code_p4.sbatch

STUDENT_MODEL=Qwen/Qwen3-1.7B-Base
SIZE_TAG=1p7b_base
METHOD_TAG=opd

DISTILLATION_LOSS_MODE="k1_topk_overlap"
HYBRID_MASK_STRATEGY="all"

MASK_KWARGS=(
    # this mask takes no kwargs
)

source "$(dirname "${BASH_SOURCE[0]}")/_common_code_p4.sh" "$@"
