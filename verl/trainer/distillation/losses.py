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
                raise NotImplementedError(
                    "forward_kl_topk_approx is only implemented for the fsdp backend."
                )
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
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

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
        # Hybrid: partition response tokens into PG (from k1 advantage) and supervised
        # (from forward_kl_topk) via a per-token mask strategy. The registered fn has
        # stashed k1_per_token + disjoint sub-masks on model_output.
        pg_mask = model_output["_hybrid_pg_mask"]
        sup_mask = model_output["_hybrid_sup_mask"]
        k1_per_token = model_output["_hybrid_k1_per_token"]

        # Supervised arm: aggregate FKL per-token losses over sup_mask only. Using the
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

        distillation_loss = (
            loss_config.pg_loss_coef * pg_loss + loss_config.supervised_loss_coef * sup_loss
        )
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

            gathered = [torch.zeros(max_size, dtype=padded.dtype, device=padded.device)
                        for _ in range(world_size)]
            torch.distributed.all_gather(gathered, padded, group=dp_group)

            all_advs = torch.cat([t[:s] for t, s in zip(gathered, all_sizes)])
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
        extra_metrics["distillation/adv_mask_keep_ratio"] = Metric(
            value=ratio, aggregation=AggregationType.MEAN
        )
        extra_metrics["distillation/adv_mask_total_tokens"] = Metric(
            value=total_tokens, aggregation=AggregationType.SUM
        )
        extra_metrics["distillation/adv_mask_kept_tokens"] = Metric(
            value=kept_tokens, aggregation=AggregationType.SUM
        )
        if adv_mask_low is not None and adv_mask_high is not None:
            extra_metrics["distillation/adv_mask_low"] = Metric(
                value=adv_mask_low, aggregation=AggregationType.MEAN
            )
            extra_metrics["distillation/adv_mask_high"] = Metric(
                value=adv_mask_high, aggregation=AggregationType.MEAN
            )
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
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Per-j overlap diagnostics stored by the logits processor.
    topk = distillation_config.distillation_loss.topk
    j = 1
    while True:
        for prefix in (
            "student_mass_at_",
            "teacher_mass_at_",
            "overlap_ratio_at_",
            "overlap_student_mass_at_",
            "overlap_teacher_mass_at_",
        ):
            key = f"{prefix}{j}"
            if key in model_output:
                vals = no_padding_2_padding(model_output[key], data)
                distillation_metrics[f"distillation/{key}"] = Metric(
                    AggregationType.MEAN, vals[response_mask_bool].mean()
                )
        if j >= topk:
            break
        j = min(j * 2, topk)

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["k1_pg_fkl_topk"], use_topk=True, use_hybrid=True)
)  # type: ignore[arg-type]
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
    from verl.trainer.distillation.hybrid_masks import HybridMaskContext, get_mask_fn

    # 1. Supervised FKL losses + mass diagnostics (same as compute_forward_kl_topk).
    sup_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    response_mask = data["response_mask"]
    response_mask_bool = response_mask.bool()
    assert sup_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # FKL tops out at 0 from below due to truncated teacher support (same rationale
    # as compute_forward_kl_topk).
    sup_losses = sup_losses.clamp_min(0.0)

    distillation_metrics: dict[str, Any] = {
        "distillation/student_mass": student_mass[response_mask_bool].mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass[response_mask_bool].min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass[response_mask_bool].max()),
        "distillation/teacher_mass": teacher_mass[response_mask_bool].mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass[response_mask_bool].min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass[response_mask_bool].max()),
    }
    # Per-j overlap diagnostics stashed by the logit processor.
    topk = distillation_config.distillation_loss.topk
    j = 1
    while True:
        for prefix in (
            "student_mass_at_",
            "teacher_mass_at_",
            "overlap_ratio_at_",
            "overlap_student_mass_at_",
            "overlap_teacher_mass_at_",
        ):
            key = f"{prefix}{j}"
            if key in model_output:
                vals = no_padding_2_padding(model_output[key], data)
                distillation_metrics[f"distillation/{key}"] = Metric(
                    AggregationType.MEAN, vals[response_mask_bool].mean()
                )
        if j >= topk:
            break
        j = min(j * 2, topk)

    loss_config = distillation_config.distillation_loss

    # 2. Per-token k1 KL at the sampled token. Clamp symmetrically to loss_max_clamp
    #    here — the outer combine path applies the same clamp to the FKL tensor, so
    #    keeping both capped in their computation sites avoids duplicating the rule.
    student_lp_at_sampled = no_padding_2_padding(model_output["log_probs"], data)
    teacher_lp_at_sampled = no_padding_2_padding(data["teacher_next_token_logprobs"], data)
    assert student_lp_at_sampled.shape == teacher_lp_at_sampled.shape == response_mask_bool.shape
    k1_per_token = kl_penalty(
        logprob=student_lp_at_sampled, ref_logprob=teacher_lp_at_sampled, kl_penalty="k1"
    )
    if loss_config.loss_max_clamp is not None:
        k1_per_token = k1_per_token.clamp(
            min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp
        )

    # 3. Build mask-strategy context. Need top-k teacher ids/logprobs + student sampled ids
    #    in padded (B, T[, K]) form.
    teacher_topk_ids = no_padding_2_padding(data["teacher_ids"], data)
    teacher_topk_logprobs = no_padding_2_padding(data["teacher_logprobs"], data)
    responses = data["responses"]
    if responses.is_nested:
        responses = responses.to_padded_tensor(padding=0)
    # responses is the student's sampled token id at each response position (B, T).
    assert responses.shape == response_mask_bool.shape, (
        f"Expected responses {response_mask_bool.shape}, got {responses.shape}"
    )

    is_argmax_padded = None
    if "is_argmax" in model_output:
        is_argmax_padded = no_padding_2_padding(model_output["is_argmax"], data).bool()
    coverage_scores = None
    if "coverage_scores" in model_output:
        coverage_scores = no_padding_2_padding(model_output["coverage_scores"], data)
    ctx = HybridMaskContext(
        response_mask=response_mask_bool,
        student_sampled_ids=responses,
        student_log_prob_at_sampled=student_lp_at_sampled,
        teacher_log_prob_at_sampled=teacher_lp_at_sampled,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        k1_per_token=k1_per_token,
        is_argmax=is_argmax_padded,
        coverage_scores=coverage_scores,
        mask_kwargs=dict(loss_config.hybrid_mask_kwargs) if loss_config.hybrid_mask_kwargs else {},
    )
    mask_fn = get_mask_fn(loss_config.hybrid_mask_strategy)
    pg_mask_raw = mask_fn(ctx)
    assert pg_mask_raw.shape == response_mask_bool.shape, (
        f"Mask strategy returned {pg_mask_raw.shape}, expected {response_mask_bool.shape}"
    )
    pg_mask = pg_mask_raw.bool() & response_mask_bool
    sup_mask = response_mask_bool & ~pg_mask

    # Stash PG-side tensors for the outer combine path. Using "_hybrid_*" keys to signal
    # intra-module coupling (see distillation_loss() for the reader side).
    model_output["_hybrid_k1_per_token"] = k1_per_token
    model_output["_hybrid_pg_mask"] = pg_mask
    model_output["_hybrid_sup_mask"] = sup_mask
    
    # Store coverage scores if available (from coverage-based mask strategies)
    if hasattr(ctx, '_debug_outputs') and 'coverage_scores' in ctx._debug_outputs:
        model_output["_coverage_scores"] = ctx._debug_outputs['coverage_scores']
        # Don't store scalar values in model_output as they can't be processed as tensors

    # Mask coverage metrics.
    total = response_mask_bool.sum().clamp(min=1)
    distillation_metrics["distillation/pg_token_ratio"] = Metric(
        AggregationType.MEAN, pg_mask.sum().float() / total.float()
    )
    # Record raw k1 without abs to preserve sign information
    distillation_metrics["distillation/mean_k1"] = Metric(
        AggregationType.MEAN, k1_per_token[response_mask_bool].mean()
    )
    # Also record std to understand distribution
    distillation_metrics["distillation/std_k1"] = Metric(
        AggregationType.MEAN, k1_per_token[response_mask_bool].std()
    )
    
    # Add top-p coverage metrics if using coverage-based mask
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    if loss_config.hybrid_mask_strategy in ["coverage_threshold", "coverage_adaptive"]:
        # Check if coverage scores were stored in model_output by the mask function
        coverage_scores = model_output.get('_coverage_scores')
        if coverage_scores is not None and isinstance(coverage_scores, torch.Tensor):
            valid_coverage = coverage_scores[response_mask_bool]
            
            # Mean coverage score (how well student aligns with teacher's top-p)
            distillation_metrics["distillation/top_p_coverage_mean"] = Metric(
                AggregationType.MEAN, valid_coverage.mean()
            )
            
            # Coverage distribution metrics
            distillation_metrics["distillation/top_p_coverage_std"] = Metric(
                AggregationType.MEAN, valid_coverage.std()
            )
            
            # Percentage of tokens with high coverage (> threshold)
            coverage_threshold = float(loss_config.hybrid_mask_kwargs.get("coverage_threshold", 0.5) if loss_config.hybrid_mask_kwargs else 0.5)
            high_coverage_ratio = (valid_coverage >= coverage_threshold).float().mean()
            distillation_metrics["distillation/top_p_high_coverage_ratio"] = Metric(
                AggregationType.MEAN, high_coverage_ratio
            )
            
            # Percentage of tokens with zero coverage (outside teacher's top-p)
            zero_coverage_ratio = (valid_coverage == 0).float().mean()
            distillation_metrics["distillation/top_p_zero_coverage_ratio"] = Metric(
                AggregationType.MEAN, zero_coverage_ratio
            )
            
            # Min and max coverage for debugging
            distillation_metrics["distillation/top_p_coverage_min"] = Metric(
                AggregationType.MIN, valid_coverage.min()
            )
            distillation_metrics["distillation/top_p_coverage_max"] = Metric(
                AggregationType.MAX, valid_coverage.max()
            )

    return sup_losses, distillation_metrics


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

    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty="k1"
    )
    metrics: dict[str, Any] = {
        "distillation/mean_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].mean()),
        "distillation/std_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].std()),
        # Keep abs_loss for backward compatibility  
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }

    # Read overlap diagnostics stored by compute_student_topk_overlap_k1 in the logits processor.
    topk = distillation_config.distillation_loss.topk
    j = 1
    while True:
        for prefix in (
            "student_mass_at_",
            "teacher_mass_at_",
            "overlap_ratio_at_",
            "overlap_student_mass_at_",
            "overlap_teacher_mass_at_",
        ):
            key = f"{prefix}{j}"
            if key in model_output:
                vals = no_padding_2_padding(model_output[key], data)
                metrics[f"distillation/{key}"] = Metric(AggregationType.MEAN, vals[response_mask_bool].mean())
        if j >= topk:
            break
        j = min(j * 2, topk)

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
