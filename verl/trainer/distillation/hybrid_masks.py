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
    """Theory-guided OPD routing based on the sign of the per-token k1 advantage.

    Uses k1 = log pi_T(a) - log pi_S(a) as the advantage signal (positive means
    teacher favors a more than student), ctx.student_topk_probs for pi_c, and
    ctx.student_s2 for the collision probability.

    Returns pg_mask (B, T): True → PG/RL arm, False → FKL / teacher top-p arm.

    Routing logic (a = sampled token, c = teacher top-p candidate, S2 = Σ_b pi_S(b)²):

      Rule 1 (low-coverage):  pi_c < student_low_threshold  → FKL
      Rule 2 (negative k1):   k1(a) < 0 and pi_a + pi_c ≤ S2  → FKL
                               (student overrates a; RL cannot reliably raise pi_c)
      Rule 3 (positive k1):   k1(a) > 0, a ∉ teacher top-p, pi_a + pi_c ≥ S2  → FKL
                               (optimizing non-teacher token a would squeeze out c)

    If any teacher-candidate c at a position triggers a FKL rule, the whole
    position routes to FKL.

    Strategy kwargs:
      student_low_threshold (float, default 1e-3)
      teacher_top_p (float, default 0.9): nucleus probability mass for teacher candidates
      adv_eps (float, default 0.0): dead-zone around zero for k1 sign
      use_low_coverage_rule (bool, default True)
      use_negative_rule (bool, default True)
      use_positive_rule (bool, default True)

    Requires ctx.student_topk_probs (B, T, K) and ctx.student_s2 (B, T) to be populated.
    Both are computed by compute_forward_kl_topk in fsdp/losses.py (FSDP backend only).
    """
    if ctx.student_topk_probs is None or ctx.student_s2 is None:
        raise ValueError(
            "opd_theory_guided requires student_topk_probs and student_s2 in "
            "HybridMaskContext. These are populated by compute_forward_kl_topk "
            "in fsdp/losses.py (FSDP backend only)."
        )

    student_low_threshold = float(ctx.mask_kwargs.get("student_low_threshold", 1e-3))
    teacher_top_p = float(ctx.mask_kwargs.get("teacher_top_p", 0.9))
    adv_eps = float(ctx.mask_kwargs.get("adv_eps", 0.0))
    use_low_coverage_rule = bool(ctx.mask_kwargs.get("use_low_coverage_rule", True))
    use_negative_rule = bool(ctx.mask_kwargs.get("use_negative_rule", True))
    use_positive_rule = bool(ctx.mask_kwargs.get("use_positive_rule", True))

    s2 = ctx.student_s2   # (B, T)
    pi_c = ctx.student_topk_probs  # (B, T, K)

    # pi_a: student prob at the sampled token, already available pre-computed
    pi_a = ctx.student_log_prob_at_sampled.exp()  # (B, T)

    teacher_ids = ctx.teacher_topk_ids              # (B, T, K)
    teacher_probs = ctx.teacher_topk_logprobs.exp()  # (B, T, K)

    # Top-p nucleus: include token k if cumulative mass before it is still < teacher_top_p.
    # teacher_topk_logprobs are sorted descending, so cumsum is monotonically increasing.
    shifted_cumsum = torch.cat(
        [torch.zeros_like(teacher_probs[..., :1]), teacher_probs.cumsum(dim=-1)[..., :-1]],
        dim=-1,
    )  # (B, T, K)
    teacher_candidate = shifted_cumsum < teacher_top_p  # (B, T, K); top-1 always True

    sampled_ids = ctx.student_sampled_ids.unsqueeze(-1)  # (B, T, 1)
    sampled_in_teacher_topp = (teacher_ids == sampled_ids) & teacher_candidate  # (B, T, K)
    sampled_in_teacher_topp = sampled_in_teacher_topp.any(dim=-1)  # (B, T)

    # k1 = log pi_T(a) - log pi_S(a): positive when teacher favors a more than student
    k1 = -ctx.k1_per_token  # (B, T)
    positive_adv = k1 > adv_eps
    negative_adv = k1 < -adv_eps

    # Rule 1: student probability on teacher token c is too low
    low_student_coverage = pi_c < student_low_threshold  # (B, T, K)

    # Rule 2: k1 < 0 (student overrates a) — RL cannot reliably raise pi_c
    negative_rl_cannot_help_c = (
        negative_adv.unsqueeze(-1)
        & ((pi_a.unsqueeze(-1) + pi_c) <= s2.unsqueeze(-1))
    )  # (B, T, K)

    # Rule 3: k1 > 0, a not in teacher top-p — PG for a squeezes out c
    positive_pg_hurts_c = (
        positive_adv.unsqueeze(-1)
        & (~sampled_in_teacher_topp).unsqueeze(-1)
        & ((pi_a.unsqueeze(-1) + pi_c) >= s2.unsqueeze(-1))
    )  # (B, T, K)

    need_fkl_per_c = torch.zeros_like(teacher_candidate, dtype=torch.bool)
    if use_low_coverage_rule:
        need_fkl_per_c |= low_student_coverage
    if use_negative_rule:
        need_fkl_per_c |= negative_rl_cannot_help_c
    if use_positive_rule:
        need_fkl_per_c |= positive_pg_hurts_c

    need_fkl_per_c &= teacher_candidate  # only apply to active teacher candidates

    # Any qualifying teacher candidate forces the position to FKL
    pg_mask = ~need_fkl_per_c.any(dim=-1)  # (B, T)

    return pg_mask
