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
    """PG where student's coverage of teacher's top-p region is below a threshold.
    
    Coverage is defined as the sum of student probabilities assigned to tokens 
    in teacher's top-p region (cumulative probability >= top_p).
    
    Strategy kwargs:
      top_p (float, default 0.9): Teacher's top-p threshold for defining high-confidence region.
      coverage_threshold (float, default 0.5): Coverage threshold for routing decision.
      inverse (bool, default True): Recommended routing logic.
                                   - True: low coverage → supervised (force alignment)
                                           high coverage → PG (preserve diversity)
                                   - False: low coverage → PG (exploration) 
                                            high coverage → supervised (refinement)
    
    Rationale: When student has low coverage of teacher's high-confidence region,
    use supervised FKL to force alignment. When coverage is high (good alignment),
    use PG to preserve diversity and avoid over-optimization.
    """
    top_p = float(ctx.mask_kwargs.get("top_p", 0.9))
    coverage_threshold = float(ctx.mask_kwargs.get("coverage_threshold", 0.5))
    inverse = bool(ctx.mask_kwargs.get("inverse", True))
    
    # Get teacher top-k probabilities and sort by probability (descending)
    teacher_probs = ctx.teacher_topk_logprobs.exp()  # (B, T, K)
    teacher_ids = ctx.teacher_topk_ids  # (B, T, K)
    
    batch_size, seq_len, top_k = teacher_probs.shape
    device = teacher_probs.device
    
    # Vectorized approach for efficiency
    # Sort teacher probabilities along top-k dimension
    sorted_probs, sorted_indices = torch.sort(teacher_probs, dim=-1, descending=True)  # (B, T, K)
    sorted_ids = torch.gather(teacher_ids, dim=-1, index=sorted_indices)  # (B, T, K)
    
    # Calculate cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, T, K)
    
    # Find top-p boundary: first position where cumsum >= top_p
    # Create mask for tokens in top-p
    in_top_p = cumsum_probs <= top_p  # (B, T, K)
    # Include at least the first token
    in_top_p[..., 0] = True
    # Also include the first token that pushes us over top_p
    shifted_cumsum = torch.cat([torch.zeros_like(cumsum_probs[..., :1]), cumsum_probs[..., :-1]], dim=-1)
    boundary = (shifted_cumsum < top_p) & (cumsum_probs >= top_p)
    in_top_p = in_top_p | boundary
    
    # Check if student's sampled token is in teacher's top-p
    student_sampled_expanded = ctx.student_sampled_ids.unsqueeze(-1)  # (B, T, 1)
    matches = (sorted_ids == student_sampled_expanded) & in_top_p  # (B, T, K)
    any_match = matches.any(dim=-1)  # (B, T)
    
    # Get the probability for matched tokens
    # Where there's a match, extract the teacher's probability
    coverage_scores = torch.zeros(batch_size, seq_len, device=device)
    match_indices = matches.float().argmax(dim=-1)  # Get index of match (or 0 if no match)
    matched_probs = torch.gather(sorted_probs, dim=-1, index=match_indices.unsqueeze(-1)).squeeze(-1)
    coverage_scores = torch.where(any_match, matched_probs, torch.zeros_like(matched_probs))
    
    # Apply response mask
    coverage_scores = coverage_scores * ctx.response_mask.float()
    
    # Apply threshold to determine mask
    if inverse:
        # Use PG when coverage is HIGH (for aggressive alignment)
        pg_mask = coverage_scores >= coverage_threshold
    else:
        # Use PG when coverage is LOW (for exploration)
        pg_mask = coverage_scores < coverage_threshold
    
    # Store coverage scores for metrics logging (will be passed to model_output)
    # Use a special key that won't conflict with user-provided kwargs
    if not hasattr(ctx, '_debug_outputs'):
        ctx._debug_outputs = {}
    ctx._debug_outputs['coverage_scores'] = coverage_scores
    # Don't store scalars, they can't be processed in model_output
    
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
