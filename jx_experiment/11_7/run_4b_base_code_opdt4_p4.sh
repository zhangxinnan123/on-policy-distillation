#!/usr/bin/env bash
############################ opd_theory_guided4 -- two-rule theory-guided routing (our method) ############################
# Qwen3-4B-Base student <- Qwen3-8B teacher, CodeR1-12k, response length 4096, p4d single node.
#
# Two rules, both asking whether a single-sample RKL/PG update can actually
# deliver the mass the teacher wants (the PG lift on candidate c is O(pi_S(c)) by
# the softmax Jacobian, so mass the student does not already hold is unreachable
# by RKL and needs a dense FKL signal):
#
#   R1 coverage: sum_{c in teacher top-p} pi_S(c) < coverage_threshold  -> FKL
#   R2 M3 vote : teacher-prob-weighted vote of [pi_T(c)/pi_S(c) > 1+eps_low]
#                over the teacher nucleus > fkl_vote_threshold          -> FKL
#   otherwise                                                          -> PG (k1)
#
# See hybrid_masks.py:_mask_opd_theory_guided4 for the full derivation.
#
# Hyperparameters are the best configuration found in the math sweep: run
# f11inhls (teacher_top_p=0.9, coverage_threshold=0.2, eps_low=2 i.e. a pi_T/pi_S
# ratio of 3, fkl_vote_threshold=0.3), which was the best opdt4 point at 16.93
# avg4 @180. Carried over unchanged rather than re-tuned, so this arm tests
# whether the math-selected setting transfers to code.
#
# HONEST FRAMING, from 8_29/RESULTS.md: on math the opd_theory_guided4 family did
# NOT work -- 11 runs, FKL share 0.6%-22.5%, none cleared its size's noise floor,
# and its best settings were the ones that barely routed at all. This sweep is a
# genuine test on a new domain, not a confirmation run. Watch
# actor/distillation/pg_token_ratio: if the FKL share is near 0 again, the arm is
# a baseline repeat and its score difference is noise regardless of sign.
#
# Shared config (placement, lengths, optimizer, eval wiring) lives in
# _common_code_p4.sh. This file sets ONLY what distinguishes this arm, so the
# eight arms of this sweep are identical by construction except the routing.
#
# Submit:
#   RUN=jx_experiment/11_7/run_4b_base_code_opdt4_p4.sh \
#     sbatch --export=ALL,RUN=$RUN --job-name=opdt4_4b jx_experiment/11_7/submit_code_p4.sbatch

STUDENT_MODEL=Qwen/Qwen3-4B-Base
SIZE_TAG=4b_base
METHOD_TAG=opdt4

DISTILLATION_LOSS_MODE="k1_pg_fkl_topk"
HYBRID_MASK_STRATEGY="opd_theory_guided4"

MASK_KWARGS=(
    +distillation.distillation_loss.hybrid_mask_kwargs.coverage_threshold=0.2
    +distillation.distillation_loss.hybrid_mask_kwargs.fkl_vote_threshold=0.3
    +distillation.distillation_loss.hybrid_mask_kwargs.eps_low=2
    +distillation.distillation_loss.hybrid_mask_kwargs.teacher_top_p=0.9
    +distillation.distillation_loss.hybrid_mask_kwargs.prob_floor=1e-6
    +distillation.distillation_loss.hybrid_mask_kwargs.use_coverage_rule=True
    +distillation.distillation_loss.hybrid_mask_kwargs.use_low_coverage_rule=True
)

source "$(dirname "${BASH_SOURCE[0]}")/_common_code_p4.sh" "$@"
