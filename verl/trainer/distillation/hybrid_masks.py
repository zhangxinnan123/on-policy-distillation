# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-token mask strategies for hybrid PG + supervised distillation losses.

A strategy returns either:
  - a single bool tensor (B, T): True → PG arm, False → supervised arm. The
    caller derives sup_mask = response_mask & ~pg_mask, so every valid
    response token is routed to one of the two arms.
  - a tuple (pg_mask, sup_mask), both bool (B, T): the caller uses them as-is
    after ANDing with response_mask. Tokens in neither mask are *dropped* —
    they contribute to no loss. Used by strategies where some positions
    should receive no update at all (e.g. opd_theory_guided with
    conflict_action="drop").

In either form, pg_mask and sup_mask must be disjoint subsets of response_mask.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch


@dataclass
class HybridMaskContext:
    """Inputs available to a mask strategy.

    All tensors are (bsz, seqlen[, ...]) padded — strategies don't need to
    handle nested/packed layouts. Tensors unrelated to a particular strategy
    are still populated; strategies just ignore what they don't use.
    """

    response_mask: torch.Tensor  # (B, T) bool
    student_sampled_ids: torch.Tensor  # (B, T) int
    student_log_prob_at_sampled: torch.Tensor  # (B, T) float
    teacher_log_prob_at_sampled: torch.Tensor  # (B, T) float — from teacher_next_token_logprobs
    teacher_topk_ids: torch.Tensor  # (B, T, K) int, sorted descending by logprob
    teacher_topk_logprobs: torch.Tensor  # (B, T, K) float
    k1_per_token: torch.Tensor  # (B, T) float = student_lp - teacher_lp at sampled
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


MaskReturn = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
MaskFn = Callable[[HybridMaskContext], MaskReturn]
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
        raise ValueError(f"Unknown hybrid mask strategy '{name}'. Available: {sorted(MASK_REGISTRY.keys())}")
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
    direction_conflict = (want_increase_c & pg_lowers_c) | (want_decrease_c & pg_raises_c)  # (B, T, K)

    # Support failure:
    # teacher wants c higher, but student probability is near zero.
    # Even if PG direction is correct, Delta pi(c) is proportional to pi_c,
    # so PG/RKL cannot reliably recover this missing mode.
    low_student_coverage = (pi_c < student_low_threshold) & want_increase_c  # (B, T, K)

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

    # Compute separate metrics for conflict and low_coverage
    # Only count within teacher candidates
    if use_direction_rule:
        conflict_per_position = (direction_conflict & teacher_candidate).any(dim=-1)  # (B, T)
        ctx.extras["opd_conflict_mask"] = conflict_per_position

    if use_low_coverage_rule:
        low_coverage_per_position = (low_student_coverage & teacher_candidate).any(dim=-1)  # (B, T)
        ctx.extras["opd_low_coverage_mask"] = low_coverage_per_position

    return pg_mask


@register_mask("opd_aligned")
def _mask_opd_aligned(ctx: HybridMaskContext) -> torch.Tensor:
    """Aggregate PG-vs-teacher alignment score per position; route by sign.

    First-order proxy for the PG-induced change in pi_S at candidate c (same
    derivation as `opd_drop_conflict`):

        c == a:  Delta pi(a) ∝ k1 * pi_a * (1 − 2*pi_a + S2)        (bracket ≥ 0)
        c != a:  Delta pi(c) ∝ k1 * pi_c * (S2 − pi_a − pi_c)

    where k1 = log pi_T(a) − log pi_S(a). Per-position alignment score:

        score = Σ_{c ∈ teacher top-p} (log pi_T(c) − log pi_S(c)) * Delta pi_proxy(c)

    Sign interpretation (Σ_c Delta pi(c) = 0 over full vocab):
      score > 0  →  PG step *decreases* RKL on the nucleus on average  →  use PG (k1)
      score ≤ 0  →  PG step would *increase* RKL on the nucleus       →  use supervised (FKL/JSD)

    Differences:
      - vs `opd_drop_conflict`: that strategy *drops* conflicting positions
        (no gradient at all); this one routes them to the supervised arm so
        they still contribute a gradient via FKL or JSD.
      - vs `opd_theory_guided`: that one is conservative (any single
        misaligned candidate triggers supervised); this one is aggregate (a
        single misaligned candidate can be outweighed by many aligned ones).

    Strategy kwargs:
      teacher_top_p (float, default 0.95): nucleus mass for teacher candidates.
      score_eps (float, default 0.0): dead-zone; route to supervised if
        score ≤ score_eps. Tune via `distillation/pg_token_ratio`.
    """
    if ctx.student_topk_probs is None or ctx.student_s2 is None:
        raise ValueError(
            "opd_aligned requires student_topk_probs and student_s2 in "
            "HybridMaskContext. These are populated by compute_forward_kl_topk "
            "/ compute_jsd_topk in fsdp/losses.py (FSDP backend only)."
        )

    teacher_top_p = float(ctx.mask_kwargs.get("teacher_top_p", 0.95))
    score_eps = float(ctx.mask_kwargs.get("score_eps", 0.0))

    s2 = ctx.student_s2  # (B, T)
    pi_c = ctx.student_topk_probs  # (B, T, K)
    pi_a = ctx.student_log_prob_at_sampled.exp()  # (B, T)

    teacher_ids = ctx.teacher_topk_ids  # (B, T, K)
    teacher_logp = ctx.teacher_topk_logprobs  # (B, T, K)
    teacher_probs = teacher_logp.exp()  # (B, T, K)

    shifted_cumsum = torch.cat(
        [
            torch.zeros_like(teacher_probs[..., :1]),
            teacher_probs.cumsum(dim=-1)[..., :-1],
        ],
        dim=-1,
    )
    teacher_candidate = shifted_cumsum < teacher_top_p  # (B, T, K)

    sampled_at_c = teacher_ids == ctx.student_sampled_ids.unsqueeze(-1)  # (B, T, K)

    k1_b = (-ctx.k1_per_token).unsqueeze(-1)  # (B, T, 1)
    pi_a_b = pi_a.unsqueeze(-1)  # (B, T, 1)
    s2_b = s2.unsqueeze(-1)  # (B, T, 1)

    bracket_at = 1.0 - 2.0 * pi_a_b + s2_b
    bracket_off = s2_b - pi_a_b - pi_c
    delta_pi_proxy = torch.where(
        sampled_at_c,
        k1_b * pi_a_b * bracket_at,
        k1_b * pi_c * bracket_off,
    )  # (B, T, K)

    log_pi_s_c = pi_c.clamp_min(1e-20).log()  # (B, T, K)
    log_ratio = teacher_logp - log_pi_s_c  # (B, T, K)

    contrib = log_ratio * delta_pi_proxy * teacher_candidate.float()  # (B, T, K)
    score = contrib.sum(dim=-1)  # (B, T)

    pg_mask = score > score_eps
    return pg_mask


@register_mask("opd_drop_conflict")
def _mask_opd_drop_conflict(ctx: HybridMaskContext) -> MaskReturn:
    """Drop positions where the PG step is predicted to increase RKL on the
    teacher nucleus.

    Per teacher-nucleus candidate c, build a signed proxy for the PG-induced
    change in pi_S:

        c == a:  Δpi(a) ∝ k1 · pi_a · (1 − 2·pi_a + S2)        (bracket ≥ 0)
        c != a:  Δpi(c) ∝ k1 · pi_c · (S2 − pi_a − pi_c)

    where k1 = log pi_T(a) − log pi_S(a). The signed score is

        score = Σ_{c ∈ nucleus} (log pi_T(c) − log pi_S(c)) · Δpi_proxy(c)

    Using Σ_c Δpi(c) = 0 (over full vocab), this equals −Δ D_KL(pi_S || pi_T)
    to first order in the PG step (restricted to the nucleus). Sign:
      score > 0  →  PG decreases RKL on the nucleus (helpful)
      score < 0  →  PG increases RKL on the nucleus (harmful)

    Drop iff score < drop_threshold. With drop_threshold=0, drop net-harmful
    positions. Higher thresholds additionally drop weakly-helpful positions.
    Remaining positions get the PG (k1) update. No FKL arm.

    Strategy kwargs:
      teacher_top_p (float, default 0.95): nucleus mass for teacher candidates.
      drop_threshold (float, default 0.0): drop iff score < threshold. Tune
        via `distillation/dropped_token_ratio`.
    """

    if ctx.student_topk_probs is None or ctx.student_s2 is None:
        raise ValueError(
            "opd_drop_conflict requires student_topk_probs and student_s2 in "
            "HybridMaskContext. These are populated by compute_forward_kl_topk "
            "in fsdp/losses.py (FSDP backend only)."
        )

    teacher_top_p = float(ctx.mask_kwargs.get("teacher_top_p", 0.95))
    drop_threshold = float(ctx.mask_kwargs.get("drop_threshold", 0.0))

    s2 = ctx.student_s2  # (B, T)
    pi_c = ctx.student_topk_probs  # (B, T, K)
    pi_a = ctx.student_log_prob_at_sampled.exp()  # (B, T)

    teacher_ids = ctx.teacher_topk_ids  # (B, T, K)
    teacher_logp = ctx.teacher_topk_logprobs  # (B, T, K)
    teacher_probs = teacher_logp.exp()  # (B, T, K)

    # Teacher candidate set: top-p nucleus inside teacher top-k.
    shifted_cumsum = torch.cat(
        [
            torch.zeros_like(teacher_probs[..., :1]),
            teacher_probs.cumsum(dim=-1)[..., :-1],
        ],
        dim=-1,
    )
    teacher_candidate = shifted_cumsum < teacher_top_p  # (B, T, K)

    sampled_at_c = teacher_ids == ctx.student_sampled_ids.unsqueeze(-1)  # (B, T, K)

    # k1 = log pi_T(a) - log pi_S(a). ctx.k1_per_token is the negation.
    k1_b = (-ctx.k1_per_token).unsqueeze(-1)  # (B, T, 1)
    pi_a_b = pi_a.unsqueeze(-1)  # (B, T, 1)
    s2_b = s2.unsqueeze(-1)  # (B, T, 1)

    # Signed proxy for Δpi(c), amplitude-aware (modulo positive learning-rate scalar).
    # c == a:  Δpi(a) ∝ k1 · pi_a · (1 − 2·pi_a + S2)         — bracket always ≥ 0
    # c != a:  Δpi(c) ∝ k1 · pi_c · (S2 − pi_a − pi_c)
    bracket_at = 1.0 - 2.0 * pi_a_b + s2_b
    bracket_off = s2_b - pi_a_b - pi_c
    delta_pi_proxy = torch.where(
        sampled_at_c,
        k1_b * pi_a_b * bracket_at,
        k1_b * pi_c * bracket_off,
    )  # (B, T, K)

    # Per-candidate log-ratio. log pi_S at teacher candidates derived from pi_c.
    log_pi_s_c = pi_c.clamp_min(1e-20).log()  # (B, T, K)
    log_ratio = teacher_logp - log_pi_s_c  # (B, T, K)

    contrib = log_ratio * delta_pi_proxy  # (B, T, K)
    contrib = contrib * teacher_candidate.float()  # restrict to nucleus

    score = contrib.sum(dim=-1)  # (B, T)
    drop_pos = score < drop_threshold  # (B, T)

    pg_mask = ~drop_pos
    sup_mask = torch.zeros_like(pg_mask)
    return pg_mask, sup_mask

