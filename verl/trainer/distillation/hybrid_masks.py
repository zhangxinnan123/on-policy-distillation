# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-token mask strategies for hybrid PG + supervised distillation losses.

Each strategy returns a boolean tensor of shape (bsz, seqlen) where True means
"use the PG arm at this position" and False means "use the supervised arm".
The framework ANDs the returned mask with `response_mask` before use, and
derives `sup_mask = response_mask & ~pg_mask`, guaranteeing the two sub-masks
are disjoint and together cover exactly the valid response tokens.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch


@dataclass
class HybridMaskContext:
    """Inputs available to a mask strategy.

    All tensors are (bsz, seqlen[, ...]) padded — strategies don't need to
    handle nested/packed layouts. Tensors unrelated to a particular strategy
    are still populated; strategies just ignore what they don't use.
    """

    response_mask: torch.Tensor               # (B, T) bool
    student_sampled_ids: torch.Tensor         # (B, T) int
    student_log_prob_at_sampled: torch.Tensor # (B, T) float
    teacher_log_prob_at_sampled: torch.Tensor # (B, T) float — from teacher_next_token_logprobs
    teacher_topk_ids: torch.Tensor            # (B, T, K) int, sorted descending by logprob
    teacher_topk_logprobs: torch.Tensor       # (B, T, K) float
    k1_per_token: torch.Tensor                # (B, T) float = student_lp - teacher_lp at sampled
    is_argmax: Optional[torch.Tensor] = None  # (B, T) bool, may be missing w/ fused kernels
    # (B, T) precomputed student-mass coverage on teacher's top-p prefix,
    # surfaced by the FSDP compute_forward_kl_topk logit processor. top_p is
    # read from hybrid_mask_kwargs at compute time. None under megatron.
    coverage_scores: Optional[torch.Tensor] = None
    # (B, T, K) student probs at teacher top-k positions — required by opd_theory_guided.
    # Computed in fsdp/losses.py as student_topk_log_probs.exp(); no full-vocab materialization.
    student_topk_probs: Optional[torch.Tensor] = None
    # (B, T) Σ_b pi_S(b)² — required by opd_theory_guided.
    # Computed in fsdp/losses.py via the logsumexp identity exp(logsumexp(2x) − 2·log_Z).
    student_s2: Optional[torch.Tensor] = None
    mask_kwargs: dict = field(default_factory=dict)


MaskFn = Callable[[HybridMaskContext], torch.Tensor]
MASK_REGISTRY: dict[str, MaskFn] = {}


def register_mask(name: str) -> Callable[[MaskFn], MaskFn]:
    """Register a mask strategy under `name`."""

    def decorator(fn: MaskFn) -> MaskFn:
        if name in MASK_REGISTRY:
            raise ValueError(f"Hybrid mask strategy '{name}' is already registered.")
        MASK_REGISTRY[name] = fn
        return fn

    return decorator


def get_mask_fn(name: str) -> MaskFn:
    if name not in MASK_REGISTRY:
        raise ValueError(
            f"Unknown hybrid mask strategy '{name}'. Available: {sorted(MASK_REGISTRY.keys())}"
        )
    return MASK_REGISTRY[name]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@register_mask("oot")
def _mask_oot(ctx: HybridMaskContext) -> torch.Tensor:
    """PG where the student-sampled token is outside teacher's top-k.

    Rationale: supervised FKL-topk cannot push OOT token probability down
    (teacher gives no top-k signal on OOT positions), so fall back to k1 PG
    there. Positions inside teacher's top-k keep the dense supervised signal.
    """
    sampled = ctx.student_sampled_ids.unsqueeze(-1)  # (B, T, 1)
    in_topk = (ctx.teacher_topk_ids == sampled).any(dim=-1)  # (B, T)
    return ~in_topk


@register_mask("adv_threshold")
def _mask_adv_threshold(ctx: HybridMaskContext) -> torch.Tensor:
    """PG where |k1_advantage| exceeds a threshold.

    Strategy kwargs:
      threshold (float, default 0.5): minimum |k1| to route a token to PG.
    """
    threshold = float(ctx.mask_kwargs.get("threshold", 0.5))
    return ctx.k1_per_token.abs() > threshold


@register_mask("teacher_conf")
def _mask_teacher_conf(ctx: HybridMaskContext) -> torch.Tensor:
    """PG where teacher's top-1 probability is below a threshold (low confidence).

    Strategy kwargs:
      threshold (float, default 0.5): if teacher_top1_prob < threshold → PG.

    When teacher is diffuse, supervised FKL overfits noisy top-k; use PG instead.
    When teacher is confident, supervised FKL provides a clean dense signal.
    """
    threshold = float(ctx.mask_kwargs.get("threshold", 0.5))
    teacher_top1_prob = ctx.teacher_topk_logprobs[..., 0].exp()  # (B, T)
    return teacher_top1_prob < threshold


@register_mask("disagreement")
def _mask_disagreement(ctx: HybridMaskContext) -> torch.Tensor:
    """PG where the student-sampled token differs from teacher's top-1 id.

    Note: compares student's *sampled* action to teacher's *argmax*, not
    student_argmax vs teacher_argmax (since student_argmax is not always
    materialized — see use_fused_kernels).
    """
    teacher_top1_id = ctx.teacher_topk_ids[..., 0]  # (B, T)
    return ctx.student_sampled_ids != teacher_top1_id


@register_mask("none")
def _mask_none(ctx: HybridMaskContext) -> torch.Tensor:
    """Route every token to the supervised arm. Equivalent to pure forward_kl_topk."""
    return torch.zeros_like(ctx.response_mask, dtype=torch.bool)


@register_mask("all")
def _mask_all(ctx: HybridMaskContext) -> torch.Tensor:
    """Route every token to the PG arm. Equivalent to k1 + PG on topk teacher data."""
    return torch.ones_like(ctx.response_mask, dtype=torch.bool)


@register_mask("coverage_threshold")
def _mask_coverage_threshold(ctx: HybridMaskContext) -> torch.Tensor:
    """PG/sup routing by student probability mass on teacher's top-p region.

    coverage = Σ_{i ∈ teacher_top_p} p_student(teacher_topk_ids[i])    ∈ [0, 1]

    where teacher_top_p is the smallest prefix of teacher's (descending-sorted)
    top-k whose cumulative probability ≥ top_p. top_p is applied inside the
    FSDP logit processor (compute_forward_kl_topk); this mask just thresholds
    the precomputed per-token scalar.

    Strategy kwargs:
      coverage_threshold (float, default 0.5): score threshold for routing.
      inverse (bool, default True):
        - True : coverage ≥ threshold → PG (student already aligned)
        - False: coverage <  threshold → PG (student misaligned, explore)
      top_p is read by the logit processor, not here.

    Requires coverage_scores in HybridMaskContext (FSDP backend only).
    """
    if ctx.coverage_scores is None:
        raise NotImplementedError(
            "coverage_threshold requires coverage_scores in HybridMaskContext. "
            "Currently surfaced only by the FSDP compute_forward_kl_topk logit "
            "processor; megatron's fused KL kernel does not expose it."
        )

    coverage_threshold = float(ctx.mask_kwargs.get("coverage_threshold", 0.5))
    inverse = bool(ctx.mask_kwargs.get("inverse", True))

    coverage = ctx.coverage_scores  # (B, T), in [0, 1]
    if inverse:
        pg_mask = coverage >= coverage_threshold
    else:
        pg_mask = coverage < coverage_threshold

    if not hasattr(ctx, "_debug_outputs"):
        ctx._debug_outputs = {}
    ctx._debug_outputs["coverage_scores"] = coverage

    return pg_mask


@register_mask("opd_theory_guided")
def _mask_opd_theory_guided(ctx: HybridMaskContext) -> torch.Tensor:
    """
    Theory-guided OPD routing.

    Return:
        pg_mask: (B, T)
            True  -> use PG/RL / reverse-style update
            False -> use FKL / supervised update

    Core principle:
        Use PG only if the sampled-token update moves teacher-supported tokens
        in the teacher-desired direction and the teacher-supported token is not
        missing from student support.

    Definitions:
        sampled token: a
        teacher candidate token: c

        A(a) = log pi_T(a) - log pi_S(a)

        For c == a:
            sign(Delta pi(c)) = sign(A(a))

        For c != a:
            Delta pi(c) ∝ A(a) * pi_S(c) * [S2 - pi_S(a) - pi_S(c)]
            so:
            sign(Delta pi(c)) = sign(A(a) * [S2 - pi_S(a) - pi_S(c)])

    FKL is used if:
        1. teacher wants c higher but student probability on c is too low;
        2. PG/RKL would move c in the opposite direction from teacher desire.

    Teacher high entropy is naturally handled because top-p/top-k will include
    more teacher-supported candidates c.
    """

    if ctx.student_topk_probs is None or ctx.student_s2 is None:
        raise ValueError(
            "opd_theory_guided requires student_topk_probs and student_s2 in "
            "HybridMaskContext. These are populated by compute_forward_kl_topk "
            "in fsdp/losses.py (FSDP backend only)."
        )

    student_low_threshold = float(ctx.mask_kwargs.get("student_low_threshold", 1e-2))
    teacher_top_p = float(ctx.mask_kwargs.get("teacher_top_p", 0.9))
    adv_eps = float(ctx.mask_kwargs.get("adv_eps", 0.0))

    use_low_coverage_rule = bool(ctx.mask_kwargs.get("use_low_coverage_rule", True))
    use_direction_rule = bool(ctx.mask_kwargs.get("use_direction_rule", True))

    s2 = ctx.student_s2  # (B, T)
    pi_c = ctx.student_topk_probs  # (B, T, K)

    # student prob at sampled token a
    pi_a = ctx.student_log_prob_at_sampled.exp()  # (B, T)

    teacher_ids = ctx.teacher_topk_ids  # (B, T, K)
    teacher_probs = ctx.teacher_topk_logprobs.exp()  # (B, T, K)

    # Teacher candidate set: top-p nucleus inside teacher top-k.
    # Include token k if cumulative mass before it is still < teacher_top_p.
    shifted_cumsum = torch.cat(
        [
            torch.zeros_like(teacher_probs[..., :1]),
            teacher_probs.cumsum(dim=-1)[..., :-1],
        ],
        dim=-1,
    )  # (B, T, K)

    teacher_candidate = shifted_cumsum < teacher_top_p  # top-1 always True

    sampled_ids = ctx.student_sampled_ids.unsqueeze(-1)  # (B, T, 1)
    sampled_at_c = teacher_ids == sampled_ids  # (B, T, K)

    # k1 = log pi_T(a) - log pi_S(a)
    # Current ctx.k1_per_token is assumed to be log pi_S(a) - log pi_T(a),
    # so we negate it.
    k1 = -ctx.k1_per_token  # (B, T)
    k1_b = k1.unsqueeze(-1)  # (B, T, 1)

    # Teacher-desired direction for candidate c:
    # teacher wants c higher iff pi_T(c) > pi_S(c).
    want_increase_c = teacher_probs > pi_c  # (B, T, K)
    want_decrease_c = teacher_probs < pi_c  # (B, T, K)

    # Correct PG-induced direction.
    #
    # For c == a:
    #   sign(Delta pi(c)) = sign(k1)
    #
    # For c != a:
    #   sign(Delta pi(c)) = sign(k1 * [S2 - pi_a - pi_c])
    bracket = s2.unsqueeze(-1) - pi_a.unsqueeze(-1) - pi_c  # (B, T, K)

    pg_dir_value = torch.where(
        sampled_at_c,
        k1_b,
        k1_b * bracket,
    )  # (B, T, K)

    pg_raises_c = pg_dir_value > adv_eps
    pg_lowers_c = pg_dir_value < -adv_eps

    # Direction conflict:
    # teacher wants c up but PG lowers it, or teacher wants c down but PG raises it.
    direction_conflict = (
        (want_increase_c & pg_lowers_c)
        | (want_decrease_c & pg_raises_c)
    )  # (B, T, K)

    # Support failure:
    # teacher wants c higher, but student probability is near zero.
    # Even if PG direction is correct, Delta pi(c) is proportional to pi_c,
    # so PG/RKL cannot reliably recover this missing mode.
    low_student_coverage = (
        (pi_c < student_low_threshold)
        & want_increase_c
    )  # (B, T, K)

    need_fkl_per_c = torch.zeros_like(teacher_candidate, dtype=torch.bool)

    if use_low_coverage_rule:
        need_fkl_per_c |= low_student_coverage

    if use_direction_rule:
        need_fkl_per_c |= direction_conflict

    # Only apply routing rules to teacher-supported candidates.
    need_fkl_per_c &= teacher_candidate

    # Position-level routing:
    # if any teacher-supported token needs FKL, route this position to FKL.
    pg_mask = ~need_fkl_per_c.any(dim=-1)  # (B, T)

    return pg_mask