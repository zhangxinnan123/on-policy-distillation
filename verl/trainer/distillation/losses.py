# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

# module-level flag so the argmax-mask warning is only emitted once per process
_ARGMAX_MASK_NO_OP_WARNED = False

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
        use_hybrid (bool): Whether the loss function combines a PG arm and a supervised arm
            via disjoint per-token sub-masks. Hybrid modes also set use_topk=True because the
            teacher-side data format is identical to the top-k path (prompt_logprobs=topk).
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False
    use_hybrid: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if self.use_hybrid:
            if not self.use_topk:
                raise ValueError("Hybrid distillation losses must also set use_topk=True.")
            if self.use_estimator:
                raise ValueError("Hybrid distillation losses cannot set use_estimator=True.")
        elif sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def _perj_keys() -> tuple[str, ...]:
    return (
        "student_mass_at_",
        "teacher_mass_at_",
        "abs_diff_at_",
        "overlap_ratio_at_",
        "overlap_student_mass_at_",
        "overlap_teacher_mass_at_",
    )


def _build_hybrid_masks(
    *,
    model_output: dict,
    data: TensorDict,
    loss_config: "DistillationLossConfig",
    response_mask_bool: torch.Tensor,
    student_lp_at_sampled: torch.Tensor,
    teacher_lp_at_sampled: torch.Tensor,
    k1_per_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the configured hybrid mask strategy and return (pg_mask, sup_mask).

    Both masks are bool (B, T) and disjoint subsets of ``response_mask_bool``.
    Tokens routed to neither are dropped from both arms.
    """
    from verl.trainer.distillation.hybrid_masks import HybridMaskContext, get_mask_fn

    teacher_topk_ids = no_padding_2_padding(data["teacher_ids"], data)
    teacher_topk_logprobs = no_padding_2_padding(data["teacher_logprobs"], data)
    responses = data["responses"]
    if responses.is_nested:
        responses = responses.to_padded_tensor(padding=0)
    assert responses.shape == response_mask_bool.shape, (
        f"Expected responses {response_mask_bool.shape}, got {responses.shape}"
    )

    def _maybe(key: str) -> Optional[torch.Tensor]:
        if key not in model_output:
            return None
        return no_padding_2_padding(model_output[key], data)

    is_argmax = _maybe("is_argmax")
    if is_argmax is not None:
        is_argmax = is_argmax.bool()

    ctx = HybridMaskContext(
        response_mask=response_mask_bool,
        student_sampled_ids=responses,
        student_log_prob_at_sampled=student_lp_at_sampled,
        teacher_log_prob_at_sampled=teacher_lp_at_sampled,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        k1_per_token=k1_per_token,
        is_argmax=is_argmax,
        coverage_scores=_maybe("coverage_scores"),
        student_topk_probs=_maybe("student_topk_probs"),
        student_s2=_maybe("student_s2"),
        mask_kwargs=dict(loss_config.hybrid_mask_kwargs) if loss_config.hybrid_mask_kwargs else {},
    )
    mask_result = get_mask_fn(loss_config.hybrid_mask_strategy)(ctx)
    if isinstance(mask_result, tuple):
        pg_mask_raw, sup_mask_raw = mask_result
        assert pg_mask_raw.shape == sup_mask_raw.shape == response_mask_bool.shape, (
            f"Mask strategy returned ({pg_mask_raw.shape}, {sup_mask_raw.shape}), "
            f"expected both {response_mask_bool.shape}"
        )
        pg_mask = pg_mask_raw.bool() & response_mask_bool
        sup_mask = sup_mask_raw.bool() & response_mask_bool
        assert not (pg_mask & sup_mask).any(), "Mask strategy returned overlapping pg_mask and sup_mask"
    else:
        assert mask_result.shape == response_mask_bool.shape, (
            f"Mask strategy returned {mask_result.shape}, expected {response_mask_bool.shape}"
        )
        pg_mask = mask_result.bool() & response_mask_bool
        sup_mask = response_mask_bool & ~pg_mask
    return pg_mask, sup_mask, ctx.extras


def _emit_topk_diagnostics(
    *,
    model_output: dict,
    data: TensorDict,
    distillation_config: DistillationConfig,
    response_mask_bool: torch.Tensor,
    pg_mask: Optional[torch.Tensor] = None,
    sup_mask: Optional[torch.Tensor] = None,
    k1_per_token: Optional[torch.Tensor] = None,
    mask_extras: Optional[dict] = None,
) -> dict[str, Any]:
    """Unified diagnostic metric dict for any top-k distillation loss.

    Static diagnostics — mass (mean/min/max), per-j overlap, top-p coverage,
    JSD KL halves — are emitted whenever the underlying tensor is present in
    ``model_output``.

    PG routing diagnostics (``pg_token_ratio``, ``dropped_token_ratio``,
    ``mean_k1``, ``std_k1``) come from one of two sources:
      - Hybrid losses pass the actual ``pg_mask``/``sup_mask``/``k1_per_token``
        used for routing — the metric describes real routing decisions.
      - Non-hybrid losses pass ``None``; the helper builds the same partition
        from the configured ``hybrid_mask_strategy`` as a diagnostic. The loss
        does not actually route on it. Skipped silently if the strategy needs
        an input the kernel didn't surface (e.g. ``is_argmax`` under fused).
    """
    metrics: dict[str, Any] = {}
    loss_config: DistillationLossConfig = distillation_config.distillation_loss

    # --- Mass on teacher top-k support ---
    if "student_mass" in model_output:
        sm = no_padding_2_padding(model_output["student_mass"], data)[response_mask_bool]
        metrics["distillation/student_mass"] = sm.mean().item()
        metrics["distillation/student_mass_min"] = Metric(AggregationType.MIN, sm.min())
        metrics["distillation/student_mass_max"] = Metric(AggregationType.MAX, sm.max())
    if "teacher_mass" in model_output:
        tm = no_padding_2_padding(model_output["teacher_mass"], data)[response_mask_bool]
        metrics["distillation/teacher_mass"] = tm.mean().item()
        metrics["distillation/teacher_mass_min"] = Metric(AggregationType.MIN, tm.min())
        metrics["distillation/teacher_mass_max"] = Metric(AggregationType.MAX, tm.max())

    # --- Per-j overlap diagnostics (j ∈ {1, 2, 4, ..., topk}) ---
    topk = loss_config.topk or 1
    j = 1
    while True:
        for prefix in _perj_keys():
            key = f"{prefix}{j}"
            if key in model_output:
                vals = no_padding_2_padding(model_output[key], data)
                metrics[f"distillation/{key}"] = Metric(AggregationType.MEAN, vals[response_mask_bool].mean())
        if j >= topk:
            break
        j = min(j * 2, topk)

    # --- JSD KL halves ---
    if "jsd_kl_T_M" in model_output:
        kl_T = no_padding_2_padding(model_output["jsd_kl_T_M"], data)[response_mask_bool]
        kl_S = no_padding_2_padding(model_output["jsd_kl_S_M"], data)[response_mask_bool]
        metrics["distillation/jsd_kl_T_M"] = kl_T.mean().item()
        metrics["distillation/jsd_kl_S_M"] = kl_S.mean().item()
        metrics["distillation/jsd_beta"] = float(loss_config.jsd_beta)

    # --- Top-p coverage (student mass on teacher's top-p prefix) ---
    if "coverage_scores" in model_output:
        cov = no_padding_2_padding(model_output["coverage_scores"], data)[response_mask_bool]
        coverage_threshold = float((loss_config.hybrid_mask_kwargs or {}).get("coverage_threshold", 0.5))
        metrics["distillation/top_p_coverage_mean"] = Metric(AggregationType.MEAN, cov.mean())
        metrics["distillation/top_p_coverage_std"] = Metric(AggregationType.MEAN, cov.std())
        metrics["distillation/top_p_high_coverage_ratio"] = Metric(
            AggregationType.MEAN, (cov >= coverage_threshold).float().mean()
        )
        metrics["distillation/top_p_zero_coverage_ratio"] = Metric(AggregationType.MEAN, (cov == 0).float().mean())
        metrics["distillation/top_p_coverage_min"] = Metric(AggregationType.MIN, cov.min())
        metrics["distillation/top_p_coverage_max"] = Metric(AggregationType.MAX, cov.max())

    # --- PG routing diagnostics ---
    is_hybrid = loss_config.loss_settings.use_hybrid
    if not is_hybrid and pg_mask is None:
        diag = _compute_pg_diag_for_non_hybrid(
            model_output=model_output,
            data=data,
            distillation_config=distillation_config,
            response_mask_bool=response_mask_bool,
        )
        if diag is not None:
            pg_mask, sup_mask, k1_per_token, mask_extras = diag

    if pg_mask is not None and k1_per_token is not None:
        total = response_mask_bool.sum().clamp(min=1).float()
        metrics["distillation/pg_token_ratio"] = Metric(AggregationType.MEAN, pg_mask.sum().float() / total)
        if sup_mask is not None:
            dropped_mask = response_mask_bool & ~pg_mask & ~sup_mask
            metrics["distillation/dropped_token_ratio"] = Metric(
                AggregationType.MEAN, dropped_mask.sum().float() / total
            )
        k1_resp = k1_per_token[response_mask_bool]
        metrics["distillation/mean_k1"] = Metric(AggregationType.MEAN, k1_resp.mean())
        metrics["distillation/std_k1"] = Metric(AggregationType.MEAN, k1_resp.std())

        # Add separate metrics for opd_theory_guided mask components if present
        if mask_extras is not None:
            if "opd_conflict_mask" in mask_extras:
                conflict_mask = mask_extras["opd_conflict_mask"] & response_mask_bool
                metrics["distillation/opd_conflict_ratio"] = Metric(
                    AggregationType.MEAN, conflict_mask.sum().float() / total
                )
            if "opd_low_coverage_mask" in mask_extras:
                low_coverage_mask = mask_extras["opd_low_coverage_mask"] & response_mask_bool
                metrics["distillation/opd_low_coverage_ratio"] = Metric(
                    AggregationType.MEAN, low_coverage_mask.sum().float() / total
                )
            if "opd_coverage_low_mask" in mask_extras:
                # R1 of opd_theory_guided4/5. Never exported before: those masks write
                # "opd_coverage_low_mask" while only "opd_low_coverage_mask" (R2) was read,
                # so R1's firing rate was invisible in every run recorded so far.
                coverage_low_mask = mask_extras["opd_coverage_low_mask"] & response_mask_bool
                metrics["distillation/opd_coverage_low_ratio"] = Metric(
                    AggregationType.MEAN, coverage_low_mask.sum().float() / total
                )
            if "opd_high_coverage_mask" in mask_extras:
                high_coverage_mask = mask_extras["opd_high_coverage_mask"] & response_mask_bool
                metrics["distillation/opd_high_coverage_ratio"] = Metric(
                    AggregationType.MEAN, high_coverage_mask.sum().float() / total
                )

    # --- Per-token PG/teacher direction conflict ratio (top-k modes only) ---
    # A position is "in conflict" if any teacher-supported top-p candidate would
    # be moved by the PG step in the opposite direction the teacher wants. Same
    # per-candidate check as opd_theory_guided, aggregated to position-level via
    # any() — no summation across candidates (matches user spec).
    conflict_inputs_present = (
        "student_topk_probs" in model_output
        and "student_s2" in model_output
        and "log_probs" in model_output
        and "teacher_ids" in data
        and "teacher_logprobs" in data
        and "responses" in data
    )
    if conflict_inputs_present:
        s2 = no_padding_2_padding(model_output["student_s2"], data)
        pi_c = no_padding_2_padding(model_output["student_topk_probs"], data)
        student_lp = no_padding_2_padding(model_output["log_probs"], data)
        teacher_topk_ids = no_padding_2_padding(data["teacher_ids"], data)
        teacher_topk_logp = no_padding_2_padding(data["teacher_logprobs"], data)
        responses = data["responses"]
        if responses.is_nested:
            responses = responses.to_padded_tensor(padding=0)

        if k1_per_token is None and "teacher_next_token_logprobs" in data:
            teacher_lp = no_padding_2_padding(data["teacher_next_token_logprobs"], data)
            k1_local = student_lp - teacher_lp
        else:
            k1_local = k1_per_token

        if (
            k1_local is not None
            and s2.shape == response_mask_bool.shape
            and student_lp.shape == response_mask_bool.shape
            and teacher_topk_ids.shape[:2] == response_mask_bool.shape
            and pi_c.shape[:2] == response_mask_bool.shape
            and responses.shape == response_mask_bool.shape
        ):
            top_p = float((loss_config.hybrid_mask_kwargs or {}).get("top_p", 0.95))
            teacher_probs = teacher_topk_logp.exp()
            shifted_cumsum = torch.cat(
                [
                    torch.zeros_like(teacher_probs[..., :1]),
                    teacher_probs.cumsum(dim=-1)[..., :-1],
                ],
                dim=-1,
            )
            teacher_candidate = shifted_cumsum < top_p  # (B, T, K)

            sampled_at_c = teacher_topk_ids == responses.unsqueeze(-1)  # (B, T, K)

            # k1_local = student_lp - teacher_lp; opd_theory_guided uses the
            # opposite sign (teacher_lp - student_lp), so negate.
            k1_b = (-k1_local).unsqueeze(-1)  # (B, T, 1)
            pi_a_b = student_lp.exp().unsqueeze(-1)  # (B, T, 1)
            bracket = s2.unsqueeze(-1) - pi_a_b - pi_c  # (B, T, K)
            pg_dir_value = torch.where(sampled_at_c, k1_b.expand_as(pi_c), k1_b * bracket)
            pg_raises_c = pg_dir_value > 0
            pg_lowers_c = pg_dir_value < 0

            want_increase_c = teacher_probs > pi_c
            want_decrease_c = teacher_probs < pi_c

            direction_conflict = (
                (want_increase_c & pg_lowers_c) | (want_decrease_c & pg_raises_c)
            ) & teacher_candidate  # (B, T, K)
            conflict_at_pos = direction_conflict.any(dim=-1)  # (B, T)

            metrics["distillation/pg_conflict_ratio"] = Metric(
                AggregationType.MEAN,
                conflict_at_pos[response_mask_bool].float().mean(),
            )

    return metrics


def _compute_pg_diag_for_non_hybrid(
    *,
    model_output: dict,
    data: TensorDict,
    distillation_config: DistillationConfig,
    response_mask_bool: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]:
    """Simulate the PG/sup partition under the configured ``hybrid_mask_strategy``
    so non-hybrid top-k losses can surface the same routing diagnostic that
    hybrid losses emit. Returns ``(pg_mask, sup_mask, k1_per_token, mask_extras)``
    or ``None`` if the strategy can't run on what the kernel surfaced. The
    ``mask_extras`` dict carries strategy-populated diagnostic tensors (e.g.
    opd_theory_guided's per-position conflict / low-coverage / high-coverage masks).
    """
    from verl.trainer.distillation.hybrid_masks import MASK_REGISTRY, HybridMaskContext, get_mask_fn

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    strategy = loss_config.hybrid_mask_strategy
    if strategy not in MASK_REGISTRY or "teacher_next_token_logprobs" not in data:
        return None
    if "log_probs" not in model_output or "responses" not in data:
        return None

    student_lp = no_padding_2_padding(model_output["log_probs"], data)
    teacher_lp = no_padding_2_padding(data["teacher_next_token_logprobs"], data)
    if student_lp.shape != response_mask_bool.shape or teacher_lp.shape != response_mask_bool.shape:
        return None
    k1_per_token = kl_penalty(logprob=student_lp, ref_logprob=teacher_lp, kl_penalty="k1")
    if loss_config.loss_max_clamp is not None:
        k1_per_token = k1_per_token.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    teacher_topk_ids = no_padding_2_padding(data["teacher_ids"], data)
    teacher_topk_logprobs = no_padding_2_padding(data["teacher_logprobs"], data)
    responses = data["responses"]
    if responses.is_nested:
        responses = responses.to_padded_tensor(padding=0)

    def _maybe(key: str) -> Optional[torch.Tensor]:
        if key not in model_output:
            return None
        return no_padding_2_padding(model_output[key], data)

    is_argmax = _maybe("is_argmax")
    if is_argmax is not None:
        is_argmax = is_argmax.bool()

    ctx = HybridMaskContext(
        response_mask=response_mask_bool,
        student_sampled_ids=responses,
        student_log_prob_at_sampled=student_lp,
        teacher_log_prob_at_sampled=teacher_lp,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        k1_per_token=k1_per_token,
        is_argmax=is_argmax,
        coverage_scores=_maybe("coverage_scores"),
        student_topk_probs=_maybe("student_topk_probs"),
        student_s2=_maybe("student_s2"),
        mask_kwargs=dict(loss_config.hybrid_mask_kwargs) if loss_config.hybrid_mask_kwargs else {},
    )
    try:
        mask_result = get_mask_fn(strategy)(ctx)
    except (ValueError, NotImplementedError):
        return None

    if isinstance(mask_result, tuple):
        pg_mask_raw, sup_mask_raw = mask_result
        pg_mask = pg_mask_raw.bool() & response_mask_bool
        sup_mask = sup_mask_raw.bool() & response_mask_bool
    else:
        pg_mask = mask_result.bool() & response_mask_bool
        sup_mask = response_mask_bool & ~pg_mask
    return pg_mask, sup_mask, k1_per_token, ctx.extras


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    loss_mode = distillation_config.distillation_loss.loss_mode
    match config.strategy:
        case "fsdp":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            if loss_mode == "k1_topk_overlap":
                distillation_loss_fn = fsdp_losses.compute_student_topk_overlap_k1
            elif loss_mode == "forward_kl_topk_approx":
                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk_approx
            elif loss_mode == "jsd_topk":
                distillation_loss_fn = fsdp_losses.compute_jsd_topk
            elif loss_mode == "k1_pg_jsd_topk":
                # Hybrid mode reuses the jsd_topk kernel; the PG arm is computed
                # later in the outer registered fn from teacher_next_token_logprobs.
                distillation_loss_fn = fsdp_losses.compute_jsd_topk
            elif loss_mode == "k1_pg_fkl_topk":
                # Hybrid mode reuses the forward_kl_topk kernel; the PG arm is
                # computed later in the outer registered fn from the same teacher
                # top-k data + teacher_next_token_logprobs.
                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
            else:
                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            if loss_mode == "forward_kl_topk_approx":
                raise NotImplementedError("forward_kl_topk_approx is only implemented for the fsdp backend.")
            if loss_mode == "jsd_topk":
                raise NotImplementedError("jsd_topk is only implemented for the fsdp backend.")
            if loss_mode == "k1_pg_jsd_topk":
                raise NotImplementedError("k1_pg_jsd_topk is only implemented for the fsdp backend.")
            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape[:2] == expected_shape, f"Expected leading shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    #### OLD: dp_group was not passed to distillation_loss
    # distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    ####
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data, dp_group)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
    dp_group=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    # Populate global batch info directly from data (rather than relying on
    # ppo_loss to set config.global_batch_info, which runs AFTER distillation_loss
    # in distillation_ppo_loss — for the first call, that dict would be empty
    # and agg_loss would fall back to local mask sum on each rank).
    loss_config.global_batch_info["dp_size"] = data["dp_size"]
    loss_config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    loss_config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    loss_config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if loss_config.loss_settings.use_hybrid:
        # Hybrid: partition response tokens via a per-token mask strategy. The registered
        # fn stashes k1_per_token + disjoint sub-masks on model_output, plus a combine_mode
        # key that selects how the two arms are combined.
        pg_mask = model_output["_hybrid_pg_mask"]
        sup_mask = model_output["_hybrid_sup_mask"]
        k1_per_token = model_output["_hybrid_k1_per_token"]
        combine_mode = model_output.get("_hybrid_combine_mode", "k1_pg_fkl_topk")

        if combine_mode == "rkl_all_fkl_masked":
            # RKL arm: PG update using k1 as per-token advantage, over ALL response tokens.
            # Mirrors the pure use_policy_gradient path but without restricting to a sub-mask.
            rkl_advantages = -k1_per_token.detach()
            rkl_loss, rkl_pg_metrics, rkl_extra_metrics = _compute_pg_distillation_loss(
                config=config,
                loss_config=loss_config,
                loss_agg_mode=loss_agg_mode,
                model_output=model_output,
                data=data,
                dp_group=dp_group,
                response_mask=response_mask,
                pg_start_mask=response_mask,
                distill_advantages=rkl_advantages,
            )
            distillation_metrics.update(rkl_pg_metrics)
            distillation_metrics.update(rkl_extra_metrics)
            # FKL arm: forward KL topk supervised loss restricted to sup_mask tokens.
            fkl_loss = agg_loss(
                loss_mat=distillation_losses,
                loss_mask=sup_mask,
                loss_agg_mode=loss_agg_mode,
                **loss_config.global_batch_info,
            )
            distillation_loss = loss_config.pg_loss_coef * rkl_loss + loss_config.supervised_loss_coef * fkl_loss
            distillation_metrics["distillation/rkl_all_loss"] = Metric(
                value=rkl_loss.detach(), aggregation=AggregationType.MEAN
            )
            distillation_metrics["distillation/fkl_masked_loss"] = Metric(
                value=fkl_loss.detach(), aggregation=AggregationType.MEAN
            )
        else:
            # k1_pg_fkl_topk / k1_pg_jsd_topk: PG arm on pg_mask, supervised arm on sup_mask.
            # The supervised tensor is whatever the registered fn put in distillation_losses
            # (per-token FKL or per-token JSD) — the combine path is identical.
            # Supervised arm: aggregate per-token losses over sup_mask only. Using the
            # global response-token denominator (batch_num_tokens) so each arm's scalar
            # contribution scales naturally with its mask share.
            sup_loss = agg_loss(
                loss_mat=distillation_losses,
                loss_mask=sup_mask,
                loss_agg_mode=loss_agg_mode,
                **loss_config.global_batch_info,
            )

            # PG arm: use -k1_per_token as advantage. Share the same advantage-gating logic
            # as the pure use_policy_gradient path. advantage_mask_* further prunes pg_mask.
            pg_advantages = -k1_per_token.detach()
            pg_loss, pg_metrics, pg_extra_metrics = _compute_pg_distillation_loss(
                config=config,
                loss_config=loss_config,
                loss_agg_mode=loss_agg_mode,
                model_output=model_output,
                data=data,
                dp_group=dp_group,
                response_mask=response_mask,
                pg_start_mask=pg_mask,
                distill_advantages=pg_advantages,
            )
            distillation_metrics.update(pg_metrics)
            distillation_metrics.update(pg_extra_metrics)

            distillation_loss = loss_config.pg_loss_coef * pg_loss + loss_config.supervised_loss_coef * sup_loss
            distillation_metrics["distillation/pg_loss"] = Metric(
                value=pg_loss.detach(), aggregation=AggregationType.MEAN
            )
            distillation_metrics["distillation/supervised_loss"] = Metric(
                value=sup_loss.detach(), aggregation=AggregationType.MEAN
            )
    elif loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        distill_advantages = -distillation_losses.detach()
        distillation_loss, pg_metrics, pg_extra_metrics = _compute_pg_distillation_loss(
            config=config,
            loss_config=loss_config,
            loss_agg_mode=loss_agg_mode,
            model_output=model_output,
            data=data,
            dp_group=dp_group,
            response_mask=response_mask,
            pg_start_mask=response_mask,
            distill_advantages=distill_advantages,
        )
        distillation_metrics.update(pg_metrics)
        distillation_metrics.update(pg_extra_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **loss_config.global_batch_info,
        )

    # Strip internal hybrid-routing keys before returning. They were stashed on
    # model_output to communicate between the registered loss and the combine path
    # above; the engine's postprocess_batch_func then iterates every key and calls
    # `.unbind()`, which fails on the non-tensor `_hybrid_combine_mode` (a string).
    for _k in list(model_output.keys()):
        if _k.startswith("_hybrid_"):
            del model_output[_k]

    return distillation_loss, distillation_metrics


def _compute_pg_distillation_loss(
    *,
    config: ActorConfig,
    loss_config: "DistillationLossConfig",
    loss_agg_mode,
    model_output: dict,
    data: TensorDict,
    dp_group,
    response_mask: torch.Tensor,
    pg_start_mask: torch.Tensor,
    distill_advantages: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """PG-style distillation update shared by use_policy_gradient and use_hybrid paths.

    Args:
        pg_start_mask: initial response mask for the PG arm. Pure PG passes the full
            response_mask; hybrid passes the OOT/adv/... sub-mask from the strategy.
            advantage_mask_* filters further narrow this.
        distill_advantages: per-token advantage tensor (already detached). For pure PG
            this is -distillation_losses; for hybrid it's -k1_per_token.

    Returns:
        (pg_loss, policy_metrics, extra_mask_metrics). `policy_metrics` is the dict
        returned by `policy_loss_fn` (already prefixed with distillation/). `extra_mask_metrics`
        contains advantage-mask + argmax-mask diagnostics.
    """
    policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    old_log_prob = data["old_log_probs"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    pg_response_mask = pg_start_mask
    masks_applied = False

    adv_mask_low = config.get("advantage_mask_low", None)
    adv_mask_high = config.get("advantage_mask_high", None)
    adv_mask_percent = config.get("advantage_mask_percent", None)

    if adv_mask_percent is not None:
        # Derive low/high from exact global percentiles across all DP ranks.
        # e.g. adv_mask_percent=0.9 → keep bottom 5% and top 5%, mask middle 90%.
        tail = (1.0 - adv_mask_percent) / 2.0  # 0.05 for 90%
        valid_advs = distill_advantages[response_mask.bool()].float()  # only response tokens

        if dp_group is not None:
            world_size = torch.distributed.get_world_size(dp_group)

            local_size = torch.tensor([valid_advs.numel()], device=valid_advs.device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            torch.distributed.all_gather(all_sizes, local_size, group=dp_group)
            all_sizes = [s.item() for s in all_sizes]

            max_size = max(all_sizes)
            padded = torch.nn.functional.pad(valid_advs, (0, max_size - valid_advs.numel()))

            gathered = [torch.zeros(max_size, dtype=padded.dtype, device=padded.device) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, padded, group=dp_group)

            all_advs = torch.cat([t[:s] for t, s in zip(gathered, all_sizes, strict=False)])
        else:
            all_advs = valid_advs

        if all_advs.numel() > 0:
            adv_mask_low = torch.quantile(all_advs, tail).item()
            adv_mask_high = torch.quantile(all_advs, 1.0 - tail).item()
        else:
            adv_mask_low = adv_mask_high = None

    if adv_mask_low is not None and adv_mask_high is not None:
        adv_keep = ~((distill_advantages >= adv_mask_low) & (distill_advantages <= adv_mask_high))
        pg_response_mask = pg_response_mask * adv_keep
        masks_applied = True

    # Apply argmax mask: skip update when advantage > 0 and token is already student's argmax.
    # When advantage > 0, the teacher wants to push the token probability higher, but if the
    # student already assigns the highest probability to this token, skip it to avoid being
    # overly aggressive. When advantage < 0 (teacher wants to lower the probability), the
    # argmax token should still be updated.
    pre_argmax_mask = pg_response_mask  # save state before argmax mask for metrics
    argmax_skip_requested = config.get("advantage_mask_skip_argmax", False)
    argmax_available = "is_argmax" in model_output
    argmax_to_mask = None
    if argmax_skip_requested and argmax_available:
        is_argmax = no_padding_2_padding(model_output["is_argmax"], data)
        argmax_to_mask = (distill_advantages > 0) & is_argmax
        pg_response_mask = pg_response_mask * ~argmax_to_mask
        masks_applied = True
    elif argmax_skip_requested and not argmax_available:
        global _ARGMAX_MASK_NO_OP_WARNED
        if not _ARGMAX_MASK_NO_OP_WARNED:
            _ARGMAX_MASK_NO_OP_WARNED = True
            warnings.warn(
                "advantage_mask_skip_argmax=True but model_output has no 'is_argmax' tensor. "
                "This typically means use_fused_kernels=True or the Megatron backend is in use, "
                "neither of which currently materializes full logits. The argmax mask will be a no-op. "
                "Set use_fused_kernels=False (FSDP) or do not enable advantage_mask_skip_argmax.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Adjust normalizer based on advantage_mask_norm setting:
    #   "kept": normalize by kept tokens only (larger per-token loss, larger grad norm)
    #   "all":  normalize by all response tokens (same scale as no masking)
    if masks_applied and config.get("advantage_mask_norm", "all") == "kept":
        global_kept_tokens = pg_response_mask.sum().to(distill_advantages.device)
        if dp_group is not None:
            torch.distributed.all_reduce(global_kept_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        global_kept_tokens = torch.clamp(global_kept_tokens, min=1)
        loss_config.global_batch_info["batch_num_tokens"] = global_kept_tokens

    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=distill_advantages,
        response_mask=pg_response_mask,
        loss_agg_mode=loss_agg_mode,
        config=loss_config,
        rollout_is_weights=rollout_is_weights,
    )
    pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}

    extra_metrics: dict[str, Any] = {}
    if masks_applied:
        total_tokens = response_mask.sum().detach().item()
        kept_tokens = pg_response_mask.sum().detach().item()
        ratio = kept_tokens / total_tokens if total_tokens > 0 else 1.0
        extra_metrics["distillation/adv_mask_keep_ratio"] = Metric(value=ratio, aggregation=AggregationType.MEAN)
        extra_metrics["distillation/adv_mask_total_tokens"] = Metric(
            value=total_tokens, aggregation=AggregationType.SUM
        )
        extra_metrics["distillation/adv_mask_kept_tokens"] = Metric(value=kept_tokens, aggregation=AggregationType.SUM)
        if adv_mask_low is not None and adv_mask_high is not None:
            extra_metrics["distillation/adv_mask_low"] = Metric(value=adv_mask_low, aggregation=AggregationType.MEAN)
            extra_metrics["distillation/adv_mask_high"] = Metric(value=adv_mask_high, aggregation=AggregationType.MEAN)
    if argmax_skip_requested and argmax_available:
        argmax_actually_masked = (argmax_to_mask & pre_argmax_mask.bool()).sum().detach().item()
        survived_adv_mask = pre_argmax_mask.sum().detach().item()
        high_adv_survived = ((distill_advantages > 0) & pre_argmax_mask.bool()).sum().detach().item()
        ratio_in_survived = argmax_actually_masked / survived_adv_mask if survived_adv_mask > 0 else 0.0
        ratio_in_high_adv = argmax_actually_masked / high_adv_survived if high_adv_survived > 0 else 0.0
        extra_metrics["distillation/argmax_mask_count"] = Metric(
            value=argmax_actually_masked, aggregation=AggregationType.SUM
        )
        extra_metrics["distillation/argmax_mask_ratio_in_survived"] = Metric(
            value=ratio_in_survived, aggregation=AggregationType.MEAN
        )
        extra_metrics["distillation/argmax_mask_ratio_in_high_adv"] = Metric(
            value=ratio_in_high_adv, aggregation=AggregationType.MEAN
        )

    return pg_loss, pg_metrics, extra_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["forward_kl_topk", "forward_kl_topk_approx"], use_topk=True)
)  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == response_mask_bool.shape

    distillation_metrics = _emit_topk_diagnostics(
        model_output=model_output,
        data=data,
        distillation_config=distillation_config,
        response_mask_bool=response_mask_bool,
    )

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["jsd_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_jsd_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generalized JSD on teacher's top-K support (renormalized over K).

    Both teacher and student top-K log-probs are renormalized so each sums to 1
    on teacher's K-support. Loss = beta * KL(p_T || M) + (1-beta) * KL(p_S || M)
    with M = beta * p_T + (1-beta) * p_S. Both KL terms are surfaced as
    diagnostics so the per-side contribution is observable.

    Loss is non-negative on proper distributions; no `clamp_min(0)` needed.
    """
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == response_mask_bool.shape

    distillation_metrics = _emit_topk_diagnostics(
        model_output=model_output,
        data=data,
        distillation_config=distillation_config,
        response_mask_bool=response_mask_bool,
    )
    return distillation_losses, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["k1_pg_fkl_topk"], use_topk=True, use_hybrid=True))  # type: ignore[arg-type]
def compute_k1_pg_fkl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Hybrid per-token partition of k1 PG and supervised forward_kl_topk.

    Teacher-side data layout matches `forward_kl_topk` (prompt_logprobs=topk) plus the
    per-position sampled-token log-prob (`teacher_next_token_logprobs`), both already
    produced by `AsyncTeacherLLMServerManager` for top-k modes.

    Flow:
      1. Per-token supervised FKL losses — already computed by the logit processor via
         compute_forward_kl_topk (stored under model_output["distillation_losses"]).
      2. Per-token k1 KL at the student-sampled token — computed here from student
         log_probs vs teacher_next_token_logprobs.
      3. Per-token mask — delegated to the configured strategy in hybrid_masks.py,
         returning a bool (B, T) selecting PG positions. sup_mask = response_mask & ~pg_mask.
      4. The outer `distillation_loss` combine path reads the stashed tensors from
         model_output (under keys prefixed with "_hybrid_") and computes the final
         scalar as α·PG(pg_mask) + β·Supervised(sup_mask).

    Returns:
      distillation_losses: per-token supervised FKL losses (B, T), clamped to ≥ 0.
                            The outer combine path aggregates these over sup_mask only.
      metrics: diagnostics dict (FKL mass + overlap + mask coverage).
    """
    sup_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    response_mask = data["response_mask"]
    response_mask_bool = response_mask.bool()
    assert sup_losses.shape == response_mask_bool.shape

    # FKL tops out at 0 from below due to truncated teacher support (same rationale
    # as compute_forward_kl_topk).
    sup_losses = sup_losses.clamp_min(0.0)

    loss_config = distillation_config.distillation_loss

    # Per-token k1 KL at the sampled token. Clamp symmetrically to loss_max_clamp
    # here — the outer combine path applies the same clamp to the FKL tensor, so
    # keeping both capped in their computation sites avoids duplicating the rule.
    student_lp_at_sampled = no_padding_2_padding(model_output["log_probs"], data)
    teacher_lp_at_sampled = no_padding_2_padding(data["teacher_next_token_logprobs"], data)
    assert student_lp_at_sampled.shape == teacher_lp_at_sampled.shape == response_mask_bool.shape
    k1_per_token = kl_penalty(logprob=student_lp_at_sampled, ref_logprob=teacher_lp_at_sampled, kl_penalty="k1")
    if loss_config.loss_max_clamp is not None:
        k1_per_token = k1_per_token.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    pg_mask, sup_mask, mask_extras = _build_hybrid_masks(
        model_output=model_output,
        data=data,
        loss_config=loss_config,
        response_mask_bool=response_mask_bool,
        student_lp_at_sampled=student_lp_at_sampled,
        teacher_lp_at_sampled=teacher_lp_at_sampled,
        k1_per_token=k1_per_token,
    )

    # Stash tensors for the outer combine path. Using "_hybrid_*" keys to signal
    # intra-module coupling (see distillation_loss() for the reader side).
    model_output["_hybrid_k1_per_token"] = k1_per_token
    model_output["_hybrid_pg_mask"] = pg_mask
    model_output["_hybrid_sup_mask"] = sup_mask
    model_output["_hybrid_combine_mode"] = "k1_pg_fkl_topk"

    distillation_metrics = _emit_topk_diagnostics(
        model_output=model_output,
        data=data,
        distillation_config=distillation_config,
        response_mask_bool=response_mask_bool,
        pg_mask=pg_mask,
        sup_mask=sup_mask,
        k1_per_token=k1_per_token,
        mask_extras=mask_extras,
    )

    return sup_losses, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["k1_pg_jsd_topk"], use_topk=True, use_hybrid=True))  # type: ignore[arg-type]
def compute_k1_pg_jsd_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Hybrid per-token partition of k1 PG and supervised jsd_topk.

    Mirrors `compute_k1_pg_fkl_topk` but the supervised arm uses generalized JSD
    on teacher's K-renormalized support (see `compute_jsd_topk`). Both KL halves
    of the JSD are surfaced as diagnostics. The supervised tensor is non-negative
    by construction so no `clamp_min(0)` is applied.
    """
    sup_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    response_mask = data["response_mask"]
    response_mask_bool = response_mask.bool()
    assert sup_losses.shape == response_mask_bool.shape

    loss_config = distillation_config.distillation_loss

    student_lp_at_sampled = no_padding_2_padding(model_output["log_probs"], data)
    teacher_lp_at_sampled = no_padding_2_padding(data["teacher_next_token_logprobs"], data)
    assert student_lp_at_sampled.shape == teacher_lp_at_sampled.shape == response_mask_bool.shape
    k1_per_token = kl_penalty(logprob=student_lp_at_sampled, ref_logprob=teacher_lp_at_sampled, kl_penalty="k1")
    if loss_config.loss_max_clamp is not None:
        k1_per_token = k1_per_token.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    pg_mask, sup_mask, mask_extras = _build_hybrid_masks(
        model_output=model_output,
        data=data,
        loss_config=loss_config,
        response_mask_bool=response_mask_bool,
        student_lp_at_sampled=student_lp_at_sampled,
        teacher_lp_at_sampled=teacher_lp_at_sampled,
        k1_per_token=k1_per_token,
    )

    # Stash for the outer combine path. The combine path treats any combine_mode
    # other than "rkl_all_fkl_masked" as "PG arm + supervised arm on sup_mask",
    # which is exactly what we want here.
    model_output["_hybrid_k1_per_token"] = k1_per_token
    model_output["_hybrid_pg_mask"] = pg_mask
    model_output["_hybrid_sup_mask"] = sup_mask
    model_output["_hybrid_combine_mode"] = "k1_pg_jsd_topk"

    distillation_metrics = _emit_topk_diagnostics(
        model_output=model_output,
        data=data,
        distillation_config=distillation_config,
        response_mask_bool=response_mask_bool,
        pg_mask=pg_mask,
        sup_mask=sup_mask,
        k1_per_token=k1_per_token,
        mask_extras=mask_extras,
    )
    return sup_losses, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["rkl_all_fkl_masked"], use_topk=True, use_hybrid=True))  # type: ignore[arg-type]
def compute_rkl_all_fkl_masked(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Hybrid: reverse KL (k1) on all response tokens + forward KL topk on masked subset.

    Uses the same mask strategy as k1_pg_fkl_topk to split response tokens into two groups,
    but instead of a PG arm on pg_mask, applies reverse KL (k1 estimator) to every token.
    The forward KL topk supervised loss is restricted to the sup_mask (FKL-routed) tokens.

    Combine path (in distillation_loss):
        total = pg_loss_coef * RKL(all) + supervised_loss_coef * FKL(sup_mask)
    """
    sup_losses, metrics = compute_k1_pg_fkl_topk(config, distillation_config, model_output, data)
    model_output["_hybrid_combine_mode"] = "rkl_all_fkl_masked"
    return sup_losses, metrics


@register_distillation_loss(DistillationLossSettings(names=["k1_topk_overlap"], use_topk=True))  # type: ignore[arg-type]
def compute_k1_topk_overlap(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """k1 loss with top-k overlap diagnostics.

    Loss: k1 single-sample KL estimator using teacher_next_token_logprobs.
    Metrics: per-j student/teacher mass and symmetric overlap ratio, computed
             in the logits processor via compute_student_topk_overlap_k1.
    """

    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_next_token_logprobs"], data)
    response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    distillation_losses = kl_penalty(logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty="k1")

    metrics = _emit_topk_diagnostics(
        model_output=model_output,
        data=data,
        distillation_config=distillation_config,
        response_mask_bool=response_mask_bool,
    )
    # Loss-value distribution stats. mean_loss/std_loss/abs_loss are kept under
    # their historical names for dashboards; mean_k1/std_k1 (added by the helper
    # via the non-hybrid PG diag path) carry the same values when both are present.
    metrics["distillation/mean_loss"] = Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].mean())
    metrics["distillation/std_loss"] = Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].std())
    metrics["distillation/abs_loss"] = Metric(
        AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()
    )

    return distillation_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Log raw k1 metrics to preserve sign information
    metrics = {
        "distillation/mean_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].mean()),
        "distillation/std_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].std()),
        # Keep abs_loss for backward compatibility
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
