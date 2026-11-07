#!/usr/bin/env bash
############################ EOPD -- entropy-guided routing ############################
# Qwen3-4B-Base student <- Qwen3-8B teacher, CodeR1-12k, response length 4096, p4d single node.
#
# Routes on the TEACHER's top-k entropy alone, using no student-side quantity:
#   H = -sum_c p_hat(c) log p_hat(c),  p_hat = teacher top-k renormalized (nats)
#   H >= 1.0 -> FKL   (teacher diffuse: the dense top-k signal carries more than
#                      a single-sample k1 update at the sampled token)
#   H <  1.0 -> PG    (teacher near-deterministic: k1 at the sampled token suffices)
# See hybrid_masks.py:_mask_teacher_topk_entropy.
#
# threshold=1.0 is the registered default and the value used throughout the
# routing-probe analysis. H is bounded by log(16)=2.77 for topk=16, so 1.0 sits
# near the midpoint. eopd has no budget parameter -- the routed fraction is the
# mask's own behaviour. On math it routed 20.7%-26.7% of tokens.
#
# Shared config (placement, lengths, optimizer, eval wiring) lives in
# _common_code_p4.sh. This file sets ONLY what distinguishes this arm, so the
# eight arms of this sweep are identical by construction except the routing.
#
# Submit:
#   RUN=jx_experiment/11_7/run_4b_base_code_eopd_p4.sh \
#     sbatch --export=ALL,RUN=$RUN --job-name=eopd_4b jx_experiment/11_7/submit_code_p4.sbatch

STUDENT_MODEL=Qwen/Qwen3-4B-Base
SIZE_TAG=4b_base
METHOD_TAG=eopd

DISTILLATION_LOSS_MODE="k1_pg_fkl_topk"
HYBRID_MASK_STRATEGY="teacher_topk_entropy"

MASK_KWARGS=(
    +distillation.distillation_loss.hybrid_mask_kwargs.threshold=1.0
)

source "$(dirname "${BASH_SOURCE[0]}")/_common_code_p4.sh" "$@"
