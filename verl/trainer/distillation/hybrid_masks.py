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


@register_mask("coverage_adaptive")
def _mask_coverage_adaptive(ctx: HybridMaskContext) -> torch.Tensor:
    """Adaptive coverage-based masking with position and confidence weighting.
    
    Strategy kwargs:
      top_p (float, default 0.9): Teacher's top-p threshold.
      base_threshold (float, default 0.5): Base coverage threshold.
      position_decay (float, default 0.1): Reduce threshold for later positions.
      confidence_weight (float, default 0.2): Weight teacher confidence in threshold.
      min_threshold (float, default 0.1): Minimum threshold value.
      max_threshold (float, default 0.8): Maximum threshold value.
    
    Rationale: Early tokens need more context preservation (lower threshold),
    later tokens can be more aggressively masked. Teacher confidence also
    affects the threshold - when teacher is confident, require higher coverage.
    """
    top_p = float(ctx.mask_kwargs.get("top_p", 0.9))
    base_threshold = float(ctx.mask_kwargs.get("base_threshold", 0.5))
    position_decay = float(ctx.mask_kwargs.get("position_decay", 0.1))
    confidence_weight = float(ctx.mask_kwargs.get("confidence_weight", 0.2))
    min_threshold = float(ctx.mask_kwargs.get("min_threshold", 0.1))
    max_threshold = float(ctx.mask_kwargs.get("max_threshold", 0.8))
    
    batch_size, seq_len = ctx.response_mask.shape
    device = ctx.response_mask.device
    
    # Calculate position-dependent threshold
    positions = torch.arange(seq_len, device=device).float()
    position_factor = torch.exp(-position_decay * positions / seq_len)
    
    # Calculate teacher confidence (top-1 probability)
    teacher_confidence = ctx.teacher_topk_logprobs[..., 0].exp()  # (B, T)
    
    # Adaptive threshold per position
    adaptive_threshold = base_threshold * position_factor.unsqueeze(0)  # (1, T)
    adaptive_threshold = adaptive_threshold + confidence_weight * teacher_confidence
    adaptive_threshold = torch.clamp(adaptive_threshold, min_threshold, max_threshold)
    
    # Calculate coverage (simplified version using teacher prob at sampled token)
    teacher_prob_at_sampled = ctx.teacher_log_prob_at_sampled.exp()  # (B, T)
    
    # Check if sampled token is in teacher's top-k (proxy for top-p)
    sampled_ids = ctx.student_sampled_ids.unsqueeze(-1)  # (B, T, 1)
    in_teacher_topk = (ctx.teacher_topk_ids == sampled_ids).any(dim=-1)  # (B, T)
    
    # Coverage score: teacher probability if in top-k, else 0
    coverage_scores = torch.where(in_teacher_topk, teacher_prob_at_sampled, torch.zeros_like(teacher_prob_at_sampled))
    
    # Apply adaptive threshold
    pg_mask = coverage_scores < adaptive_threshold
    
    return pg_mask
