# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

# Privileged "expert trajectory" template used by SDPO's modified ref-prompt path.
# Ends with the \boxed{} format instruction so the model still knows the answer
# format even after `strip_suffix` removes the dataset's original suffix.
# `{{}}` escapes the literal `{}` inside .format()'s substitution machinery.
# Direct-hint template: for short LLM-extracted hints (e.g. level_3 ~80 words).
# No "don't quote / paraphrase" clause — a short hint has no verbatim content
# worth copying, and forbidding paraphrase would suppress use of key terms the
# hint deliberately names (e.g. "partial fraction", "telescoping"). Frames the
# hint as a suggested direction rather than a full trajectory to distill from.
EXPERT_GUIDANCE_TEMPLATE = (
    "\n\nHint (a suggested approach — use it to guide your reasoning, but derive every step yourself):"
    "{expert}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

# Trailing text removed from the last user message before the expert template is
# appended. Matches what `examples/data_preprocess/deepmath_diff6to8.py` tacks
# onto every prompt; stripping prevents it from getting stranded between the
# question and the expert hint (the EXPERT_GUIDANCE_TEMPLATE itself ends with the
# same \boxed{} instruction). Override per-row via cfg.strip_suffix if needed; an
# empty string in the cfg disables stripping for that call.
DEFAULT_STRIP_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def compute_sdpo_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,        # log pi(y|x)        —— without hint
    sdpo_ref_log_prob: torch.Tensor,    # log pi(y|x, e)     —— with hint (SDPO ref)
    hint_solvable: Optional[np.ndarray] = None,  # legacy, ignored when is_hint_row is present
    is_hint_row: Optional[np.ndarray] = None,
    norm_adv_by_std_in_grpo: bool = True,
    sign_flip_lambda_pos: float = 0.5,
    sign_flip_lambda_neg: float = 0.0,
    sign_flip_epsilon: float = 0.2,
    regime_c_scale: float = 1.0,
    always_merge_hint_into_group: bool = False,
    regime_c_zero_hint_adv: bool = False,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SDPO + GRPO advantage with per-uid regime selection.

    When `is_hint_row` is provided, the batch contains both unhinted (main) and
    hinted rollouts grouped by `index` (uid). For each uid, pick a regime:

      A. No hint succeeds:
         vanilla GRPO on main rows only (group size = n).
         Hint rows → advantage = 0 (no gradient).

      B. Any hint succeeds AND any main also succeeds (mixed / all-succeed main):
         SDPO sign-flip on main rows (group size = n; sign-flip uses the existing
         delta = sdpo_ref_log_prob - old_log_probs on main rows).
         Hint rows → advantage = 0 (excluded from group + gradient).

      C. Any hint succeeds AND all main fail ("stuck student"):
         GRPO with group size = n + n_hint. Both halves get gradient.
         No sign-flip applied (signal is already strong; sign-flip would re-weight
         tokens that are already at the GRPO baseline).

    When `is_hint_row` is None, falls back to the previous single-regime behavior
    (vanilla GRPO + optional sign-flip on solvable prompts).
    """
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]
    device = scores.device
    advantages = torch.zeros_like(token_level_rewards)
    any_flip = (sign_flip_lambda_pos > 0.0) or (sign_flip_lambda_neg > 0.0)

    with torch.no_grad():
        if is_hint_row is None:
            # ---- Legacy single-regime path (no combined hint rollouts) ----
            return _compute_sdpo_grpo_advantage_legacy(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                old_log_probs=old_log_probs,
                sdpo_ref_log_prob=sdpo_ref_log_prob,
                hint_solvable=hint_solvable,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                sign_flip_lambda_pos=sign_flip_lambda_pos,
                sign_flip_lambda_neg=sign_flip_lambda_neg,
                sign_flip_epsilon=sign_flip_epsilon,
                epsilon=epsilon,
            )

        # ---- Per-uid regime dispatch (combined hint-rollout batch) ----
        hint_arr = np.asarray(is_hint_row, dtype=bool)
        idx_arr = np.asarray(index)

        # Build per-uid main/hint row index lists.
        uid_main: dict = defaultdict(list)
        uid_hint: dict = defaultdict(list)
        for i in range(bsz):
            (uid_hint if hint_arr[i] else uid_main)[idx_arr[i]].append(i)

        scores_cpu = scores.detach().cpu().numpy()
        n_regA = n_regB = n_regC = 0  # for diagnostics
        regC_hint_rows_all: list = []  # accumulate all Regime C hint row indices

        all_uids = set(list(uid_main.keys()) + list(uid_hint.keys()))
        for uid in all_uids:
            main_rows = uid_main.get(uid, [])
            hint_rows = uid_hint.get(uid, [])
            main_scores = scores_cpu[main_rows] if main_rows else np.array([])
            hint_scores = scores_cpu[hint_rows] if hint_rows else np.array([])

            any_hint_succ = bool((hint_scores > 0).any()) if hint_scores.size else False
            any_main_succ = bool((main_scores > 0).any()) if main_scores.size else False
            all_main_fail = (not any_main_succ) and (main_scores.size > 0)

            # Choose the GRPO group: main only, or main+hint (always).
            #   `always_merge_hint_into_group=True` → hint rows always in group
            #   EXCEPT in Regime A (hint failed): failed hint would pollute the
            #   group statistics without carrying useful gradient signal, so
            #   we skip it. Hint row keeps advantage=0 in Regime A.

            if not any_hint_succ:
                # Regime A: hint failed. Skip hint rows entirely (regardless of
                # always_merge_hint_into_group); hint stays at 0 advantage.
                n_regA += 1
                _fill_grpo_advantages(
                    advantages=advantages,
                    scores=scores,
                    response_mask=response_mask,
                    row_idx=main_rows,
                    norm_by_std=norm_adv_by_std_in_grpo,
                    epsilon=epsilon,
                )
            elif all_main_fail:
                # Regime C: student stuck, hint succeeds.
                # Skip GRPO normalization here. Set advantage per raw score:
                #   hint (succ, score=1) → A = +1  (positive learning signal)
                #   main (fail, score=0) → A =  0  (no gradient, no punishment)
                # Motivation: main-all-fail means "student doesn't know how", not
                # "student did something wrong". Punishing failures under GRPO
                # zero-sum in Regime C creates spurious negative gradient (~-0.09
                # after α=0.25) that has no clear signal. Instead we use hint as
                # the sole positive example. Downstream amplify (if enabled)
                # will further shape the hint row's per-token gradient.
                # NOTE: regime_c_scale still multiplies the +1 (defaults 1.0
                # under this branch — no further dampening needed).
                n_regC += 1
                # Broadcast per-row scalar to per-token via response_mask.
                # `regime_c_zero_hint_adv=True` sets hint A=0 too, delegating
                # hint gradient signal to hint_reg auxiliary loss (avoids double-
                # counting when hint_reg is enabled).
                if not regime_c_zero_hint_adv:
                    for i in hint_rows:
                        advantages[i] = regime_c_scale * response_mask[i]  # A=regime_c_scale on valid tokens
                # main_rows remain zero (initial `advantages = torch.zeros_like`).
                # Track Regime C hint rows so trainer can build hint_reg_mask
                # regardless of whether hint advantage was zeroed out.
                regC_hint_rows_all.extend(hint_rows)
            else:
                # Regime B: main has both successes and failures + hint succeeded.
                # If always_merge is ON, hint (which succeeded) joins the GRPO
                # group; otherwise only main rows normalized. Sign-flip still
                # reweights only main rows (its formula is defined for main).
                n_regB += 1
                rb_rows = (main_rows + hint_rows) if always_merge_hint_into_group else main_rows
                _fill_grpo_advantages(
                    advantages=advantages,
                    scores=scores,
                    response_mask=response_mask,
                    row_idx=rb_rows,
                    norm_by_std=norm_adv_by_std_in_grpo,
                    epsilon=epsilon,
                )
                if any_flip:
                    _apply_sign_flip_inplace(
                        advantages=advantages,
                        scores=scores,
                        response_mask=response_mask,
                        old_log_probs=old_log_probs,
                        sdpo_ref_log_prob=sdpo_ref_log_prob,
                        row_idx=main_rows,
                        lambda_pos=sign_flip_lambda_pos,
                        lambda_neg=sign_flip_lambda_neg,
                        eps=sign_flip_epsilon,
                    )

        # Stash per-step regime counts in a module-global so the trainer can pull
        # them into the metrics dict. (Function signature is locked.)
        global _LAST_SDPO_REGIME_COUNTS, _LAST_SDPO_REGC_HINT_ROWS
        _LAST_SDPO_REGIME_COUNTS = {
            "regA_hint_fail": n_regA,
            "regB_sign_flip": n_regB,
            "regC_stuck_student": n_regC,
        }
        _LAST_SDPO_REGC_HINT_ROWS = regC_hint_rows_all

    return advantages, advantages


# Filled in by `compute_sdpo_grpo_advantage`; the trainer reads this immediately
# after the call to surface per-step regime counts as wandb metrics.
_LAST_SDPO_REGIME_COUNTS: dict = {}
# List of row indices that belong to Regime C hint rows in the last
# compute_sdpo_grpo_advantage call. Read by the trainer to build hint_reg_mask.
_LAST_SDPO_REGC_HINT_ROWS: list = []


def sign_flip_decay_multiplier(
    step: int,
    total_steps: int,
    decay_type: str = "cosine",
    final_scale: float = 0.0,
    warmup_steps: int = 0,
) -> float:
    """Multiplier in [final_scale, 1.0] applied to the SDPO sign-flip lambdas.

    The "self-distilled KL" delta = log π_θ(y|x,e) - log π_θ(y|x) is weighted by
    sign_flip_lambda_{pos,neg}. This schedule decays those lambdas from their
    configured value (multiplier 1.0) toward `final_scale` (default 0.0 == fully
    off) over `total_steps`, so the hint signal fades as the policy matures.

    progress = clip((step - warmup) / (total - warmup), 0, 1)
      cosine: final + 0.5*(1-final)*(1+cos(pi*progress))   (smooth, default)
      linear: final + (1-final)*(1-progress)               (constant rate)

    During warmup (step <= warmup_steps) the multiplier stays 1.0.
    """
    if total_steps <= 0:
        return 1.0
    if step <= warmup_steps:
        return 1.0
    denom = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / denom
    progress = min(1.0, max(0.0, progress))
    if decay_type == "linear":
        m = final_scale + (1.0 - final_scale) * (1.0 - progress)
    elif decay_type == "cosine":
        m = final_scale + 0.5 * (1.0 - final_scale) * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"unknown sign_flip decay_type: {decay_type!r} (expected 'cosine' or 'linear')")
    return m


def _fill_grpo_advantages(
    advantages: torch.Tensor,
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    row_idx: list[int],
    norm_by_std: bool,
    epsilon: float,
) -> None:
    """In-place fill of `advantages[row_idx]` with GRPO centered-score advantages
    over the rows in `row_idx`. Degenerate groups (size <=1 or std==0) get 0.
    """
    if not row_idx:
        return
    group = scores[row_idx]
    mean = group.mean() if group.numel() > 0 else torch.tensor(0.0, device=scores.device)
    if group.numel() <= 1:
        std = torch.tensor(1.0, device=scores.device)
    else:
        std = group.std()
    centered = scores[row_idx] - mean
    if norm_by_std:
        centered = centered / (std + epsilon)
    advantages[row_idx] = centered.unsqueeze(-1) * response_mask[row_idx]


def _apply_sign_flip_inplace(
    advantages: torch.Tensor,
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    sdpo_ref_log_prob: torch.Tensor,
    row_idx: list[int],
    lambda_pos: float,
    lambda_neg: float,
    eps: float,
) -> None:
    """In-place rescale `advantages[row_idx]` by the SDPO sign-flip factor
    derived from delta = sdpo_ref_log_prob - old_log_probs. See class docstring."""
    if not row_idx:
        return
    delta = sdpo_ref_log_prob[row_idx] - old_log_probs[row_idx]             # [k, L]
    # Use sign of the per-row mean advantage (already scaled by response_mask).
    # advantages[row_idx] is [k, L] with the same value broadcast across L; take
    # any non-masked entry, or recover from `scores - mean` if all-zero mask.
    # Simpler: use sign of the GRPO-normalized score residual we just wrote.
    per_row_centered = (advantages[row_idx].sum(-1) / response_mask[row_idx].sum(-1).clamp(min=1.0))
    sign_A = torch.sign(per_row_centered).unsqueeze(-1)                     # [k, 1]
    w = torch.exp(sign_A * delta)                                           # [k, L]
    w = torch.clamp(w, 1.0 - eps, 1.0 + eps)
    lambda_t = torch.where(
        per_row_centered > 0,
        torch.full_like(per_row_centered, lambda_pos),
        torch.where(
            per_row_centered < 0,
            torch.full_like(per_row_centered, lambda_neg),
            torch.zeros_like(per_row_centered),
        ),
    ).unsqueeze(-1)
    scale = (1.0 - lambda_t) + lambda_t * w
    advantages[row_idx] = advantages[row_idx] * scale * response_mask[row_idx]


def _compute_sdpo_grpo_advantage_legacy(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sdpo_ref_log_prob: torch.Tensor,
    hint_solvable: Optional[np.ndarray],
    norm_adv_by_std_in_grpo: bool,
    sign_flip_lambda_pos: float,
    sign_flip_lambda_neg: float,
    sign_flip_epsilon: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Old single-regime path: GRPO baseline + optional sign-flip on hint-solvable
    prompts. Used when `is_hint_row` is None (i.e. hint rollouts not merged in)."""
    scores = token_level_rewards.sum(dim=-1)
    id2score: dict = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}
    bsz = scores.shape[0]
    with torch.no_grad():
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                t = torch.stack(id2score[idx])
                id2mean[idx] = t.mean()
                id2std[idx] = t.std()
        centered = scores.clone()
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                centered[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                centered[i] = scores[i] - id2mean[index[i]]
        grpo_adv = centered.unsqueeze(-1) * response_mask
        any_flip = (sign_flip_lambda_pos > 0.0) or (sign_flip_lambda_neg > 0.0)
        if any_flip:
            delta = sdpo_ref_log_prob - old_log_probs
            sign_A = torch.sign(centered).unsqueeze(-1)
            w = torch.exp(sign_A * delta)
            w = torch.clamp(w, 1.0 - sign_flip_epsilon, 1.0 + sign_flip_epsilon)
            lambda_t = torch.where(
                centered > 0,
                torch.full_like(centered, sign_flip_lambda_pos),
                torch.where(
                    centered < 0,
                    torch.full_like(centered, sign_flip_lambda_neg),
                    torch.zeros_like(centered),
                ),
            ).unsqueeze(-1)
            scale = (1.0 - lambda_t) + lambda_t * w
            if hint_solvable is not None:
                solvable_t = torch.as_tensor(
                    hint_solvable, device=scale.device, dtype=scale.dtype
                ).unsqueeze(-1)
                scale = torch.where(solvable_t > 0, scale, torch.ones_like(scale))
            advantages = grpo_adv * scale * response_mask
        else:
            advantages = grpo_adv
    return advantages, advantages

def _augment_messages_with_expert(
    messages: list,
    expert_text: Optional[str],
    strip_suffix: Optional[str] = None,
) -> list:
    """Append the expert-trajectory hint to the last user message.

    If `strip_suffix` is set and the last user message ends with it, that suffix
    is removed before the expert template is appended. Use this to peel off a
    dataset-injected instruction (e.g. deepmath's
    "\\nPlease reason step by step, and put your final answer within \\boxed{}.")
    so it doesn't end up stranded mid-prompt — the EXPERT_GUIDANCE_TEMPLATE itself
    ends with the boxed-format instruction, so format info is preserved.

    If no user message exists, appends a new user turn carrying the hint.
    Returns a shallow-copied list so the caller's `raw_prompt` is not mutated.
    """
    if not expert_text:
        return [dict(m) for m in messages]
    augmented = [dict(m) for m in messages]
    for i in range(len(augmented) - 1, -1, -1):
        if augmented[i].get("role") == "user":
            content = augmented[i].get("content", "") or ""
            if strip_suffix and content.endswith(strip_suffix):
                content = content[: -len(strip_suffix)]
            augmented[i]["content"] = content + EXPERT_GUIDANCE_TEMPLATE.format(expert=expert_text)
            return augmented
    augmented.append(
        {"role": "user", "content": EXPERT_GUIDANCE_TEMPLATE.format(expert=expert_text).lstrip("\n")}
    )
    return augmented


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "sdpo_grpo":
        # SDPO+GRPO: outcome-only GRPO baseline + sign-flip token reweighting on
        # solvable prompts (delta = log pi(y|x,e) - log pi(y|x); see
        # compute_sdpo_grpo_advantage for the exact formula). Knobs come from
        # data.meta_info["sdpo_grpo"]; missing knobs fall back to defaults that
        # match the function signature.
        sdpo_cfg = data.meta_info.get("sdpo_grpo", {}) or {}
        assert "old_log_probs" in data.batch, (
            "sdpo_grpo requires `old_log_probs` in batch. Run _compute_old_log_prob first."
        )
        assert "sdpo_ref_log_prob" in data.batch, (
            "sdpo_grpo requires `sdpo_ref_log_prob` in batch. Enable use_reference_policy and "
            "set sdpo.modify_ref_prompt.enabled=True so KL is against pi_theta(y|x,e)."
        )
        # The caller MUST NOT have applied apply_kl_penalty (use_kl_in_reward=False),
        # otherwise the GRPO baseline would already include a KL term and we'd double-count.
        assert "token_level_rewards" in data.batch, "token_level_rewards missing in batch"
        hint_solvable_arr = data.non_tensor_batch.get("hint_solvable")
        if hint_solvable_arr is None:
            print(
                "[sdpo_grpo] WARNING: `hint_solvable` not in batch at advantage time; "
                "sign-flip reweighting will apply to ALL prompts this step."
            )
        is_hint_row_arr = data.non_tensor_batch.get("is_hint_row")
        advantages, returns = compute_sdpo_grpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            old_log_probs=data.batch["old_log_probs"],
            sdpo_ref_log_prob=data.batch["sdpo_ref_log_prob"],
            hint_solvable=hint_solvable_arr,
            is_hint_row=is_hint_row_arr,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            sign_flip_lambda_pos=float(sdpo_cfg.get("sign_flip_lambda_pos", 0.5)),
            sign_flip_lambda_neg=float(sdpo_cfg.get("sign_flip_lambda_neg", 0.0)),
            sign_flip_epsilon=float(sdpo_cfg.get("sign_flip_epsilon", 0.2)),
            regime_c_scale=float(sdpo_cfg.get("regime_c_scale", 1.0)),
            always_merge_hint_into_group=bool(sdpo_cfg.get("always_merge_hint_into_group", False)),
            regime_c_zero_hint_adv=bool(sdpo_cfg.get("regime_c_zero_hint_adv", False)),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # sdpo_grpo needs sdpo_ref_log_prob in the batch (KL = pi(y|x) || pi(y|x,e))
        # even when neither use_kl_in_reward nor use_kl_loss is on.
        # need_reference_policy doesn't know about sdpo_grpo, so force the flag here.
        if str(self.config.algorithm.adv_estimator) == "sdpo_grpo":
            self.use_reference_policy = True
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

        # Lazy-init: built on first step when distillation.loop_metrics.enabled.
        self._loop_metrics_pool = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        keep_keys = {"data_source", "reward_model", "extra_info", "uid"}

        # SDPO: when modify_ref_prompt is enabled with source=dataset, retain the
        # expert column in `batch.non_tensor_batch` so that _compute_ref_log_prob
        # (which runs after rollout) can still find it. Otherwise the rollout
        # streaming reward loop drops it and the expert text resolves to None for
        # every row, silently making the ref-prompt modification a no-op.
        modify_cfg = self._get_modify_ref_prompt_cfg()
        if modify_cfg is not None and modify_cfg.get("source", "dataset") == "dataset":
            expert_key = modify_cfg.get("dataset_key", "answer")
            if expert_key:
                keep_keys.add(expert_key)

        reward_keys = keep_keys & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _should_compute_teacher_colocate(self, batch: DataProto) -> bool:
        return self.use_teacher_policy and not self.distillation_config.teacher_model.enable_resource_pool

    def _get_loop_metrics_cfg(self):
        """Return the loop_metrics OmegaConf node if enabled, else None.

        Allowed independently of distillation.enabled — a user may want loop
        diagnostics on a plain RL run too.
        """
        dist_cfg = self.config.get("distillation", None) if hasattr(self.config, "get") else None
        if dist_cfg is None:
            return None
        loop_cfg = dist_cfg.get("loop_metrics", None) if hasattr(dist_cfg, "get") else None
        if loop_cfg is None or not loop_cfg.get("enabled", False):
            return None
        return loop_cfg

    def _get_or_init_loop_metrics_pool(self):
        """Lazy-init the LoopMetricsPool on first use. Returns None if init fails;
        loop metrics are best-effort and must never block training.
        """
        if self._loop_metrics_pool is not None:
            return self._loop_metrics_pool
        loop_cfg = self._get_loop_metrics_cfg()
        if loop_cfg is None:
            return None
        try:
            from verl.trainer.distillation.loop_metrics import LoopMetricsPool

            tok_name = getattr(self.tokenizer, "name_or_path", None) or self.config.actor_rollout_ref.model.path
            self._loop_metrics_pool = LoopMetricsPool(
                tokenizer_name_or_path=tok_name,
                num_workers=int(loop_cfg.get("num_workers", 4)),
                inline_tokenizer=self.tokenizer if int(loop_cfg.get("num_workers", 4)) <= 0 else None,
            )
            return self._loop_metrics_pool
        except Exception as e:
            print(f"[loop_metrics] failed to init pool: {e!r}; disabling for this run.")
            self._loop_metrics_pool = None
            return None

    def _compute_teacher_colocate(self, batch: DataProto) -> DataProto:
        """Compute teacher logprobs after rollout when teacher and student are colocated."""
        assert self.teacher_model_manager is not None, "TeacherModelManager is None"
        teacher_batch = self.teacher_model_manager.compute_logprobs(batch)
        if "teacher_multi_modal_data" in batch.non_tensor_batch:
            batch.pop(non_tensor_batch_keys=["teacher_multi_modal_data"])
        return teacher_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_complete = []  # 1.0 if generation ended with EOS, else 0.0
        sample_lengths = []  # Response lengths for each sample

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Check if each response ends with the EOS token (complete generation)
            response_mask = test_output_gen_batch.batch["response_mask"]
            resp_lengths = response_mask.sum(dim=-1)  # (B,)
            last_idx = (resp_lengths - 1).clamp(min=0).long()
            last_tokens = output_ids[torch.arange(output_ids.size(0), device=output_ids.device), last_idx]
            is_complete = (last_tokens == self.tokenizer.eos_token_id) & (resp_lengths > 0)
            sample_complete.extend(is_complete.float().cpu().tolist())
            sample_lengths.extend(resp_lengths.cpu().tolist())

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        val_metrics = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        if sample_complete:
            val_metrics["val/response/complete_ratio"] = sum(sample_complete) / len(sample_complete)
        if sample_lengths:
            sample_lengths_array = np.array(sample_lengths)
            val_metrics["val/response_length/mean"] = float(sample_lengths_array.mean())
            val_metrics["val/response_length/std"] = float(sample_lengths_array.std())
            val_metrics["val/response_length/min"] = float(sample_lengths_array.min())
            val_metrics["val/response_length/max"] = float(sample_lengths_array.max())
        return val_metrics

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                engine_config: EngineConfig = orig_critic_cfg.engine
                engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            elif str(Role.ActorRolloutRef) in all_wg:
                # Model engine: ActorRolloutRefWorker
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]
            # else: deferred. With SDPO sdpo_grpo + modify_ref_prompt.use_current_actor=True
            # and use_kl_in_reward=False, use_kl_loss=False, neither Role.RefPolicy nor
            # Role.ActorRolloutRef is registered. sdpo_ref_log_prob is computed from the
            # actor in _compute_ref_log_prob, so we alias ref_policy_wg = actor_rollout_wg
            # below (just so other paths like profiling don't see an unset attribute).

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # Fallback alias for the SDPO sdpo_grpo + use_current_actor case described above.
        if self.use_reference_policy and not hasattr(self, "ref_policy_wg"):
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import TeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = TeacherModelManager(
                config=self.config.distillation,
                resource_pool=teacher_resource_pool,
                student_eos_token_id=self.tokenizer.eos_token_id,
                teacher_eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_model_manager=self.teacher_model_manager,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _get_modify_ref_prompt_cfg(self):
        """Return the sdpo.modify_ref_prompt OmegaConf node if enabled, else None.

        Reads from `self.config.sdpo.modify_ref_prompt`. Top-level `sdpo` is not part
        of the bundled trainer config schema — pass overrides on the CLI, e.g.
            +sdpo.modify_ref_prompt.enabled=True \
            +sdpo.modify_ref_prompt.source=dataset \
            +sdpo.modify_ref_prompt.dataset_key=answer
        """
        sdpo_cfg = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
        if sdpo_cfg is None:
            return None
        cfg = sdpo_cfg.get("modify_ref_prompt", None) if hasattr(sdpo_cfg, "get") else None
        if cfg is None or not cfg.get("enabled", False):
            return None
        return cfg

    def _get_expert_texts(self, batch: DataProto, cfg) -> list:
        """Return per-sample expert text (or None) used to augment the prompt.

        Sources:
        - "dataset": for each row, look up `dataset_key` first in
          `non_tensor_batch["extra_info"][key]`, then fall back to a top-level
          non_tensor column `non_tensor_batch[key]`. If the resolved value is a
          list, pick `expert_index` (default 0). Empty / missing values become None.
        - "rollout": decode `batch.batch["responses"]`; the model's own generation
          is fed back as a privileged hint. Only valid post-rollout (i.e. requires
          the batch to contain `responses`).
        """
        source = cfg.get("source", "dataset")
        bsz = len(batch)

        if source == "rollout":
            response_ids = batch.batch["responses"]
            return self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        if source == "dataset":
            key = cfg.get("dataset_key", "answer")
            expert_index = int(cfg.get("expert_index", 0))
            extra_info = batch.non_tensor_batch.get("extra_info", None)
            top_level = batch.non_tensor_batch.get(key, None)

            def _pick(val):
                # list / tuple / numpy ndarray (multiple expert trajectories per row):
                # pick `expert_index`. Strings (also iterable) are NOT split.
                if isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) == 0:
                        return None
                    idx = expert_index if 0 <= expert_index < len(val) else 0
                    val = val[idx]
                # Empty string / None / pandas NaN-ish → treat as missing.
                if val is None:
                    return None
                if isinstance(val, str) and not val:
                    return None
                return val

            out = []
            for i in range(bsz):
                v = None
                if extra_info is not None:
                    ei = extra_info[i]
                    if isinstance(ei, dict):
                        v = ei.get(key)
                if v is None and top_level is not None:
                    v = top_level[i]
                out.append(_pick(v))
            return out

        raise ValueError(f"Unknown expert source: {source!r} (expected 'dataset' or 'rollout')")

    def _get_hint_rollout_cfg(self):
        """Return the sdpo.hint_rollout OmegaConf node if enabled, else None.

        Config (Hydra overrides since `sdpo` is not in the bundled schema):
            +sdpo.hint_rollout.enabled=True \
            +sdpo.hint_rollout.n_hint=4
        Expert source is shared with `sdpo.modify_ref_prompt.{source,dataset_key}`
        (see `_get_hint_expert_cfg`).
        """
        sdpo_cfg = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
        if sdpo_cfg is None:
            return None
        cfg = sdpo_cfg.get("hint_rollout", None) if hasattr(sdpo_cfg, "get") else None
        if cfg is None or not cfg.get("enabled", False):
            return None
        return cfg

    def _get_hint_expert_cfg(self):
        """Return the expert-source config for the hint rollout.

        Reuses `sdpo.modify_ref_prompt.{source,dataset_key}` so the diagnostic and
        the actual SDPO mechanism agree on where the expert text comes from.
        Falls back to {source: dataset, dataset_key: answer} if modify_ref_prompt is
        absent. Always returns an object that supports `.get(key, default)`.
        """
        sdpo_cfg = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
        if sdpo_cfg is not None and hasattr(sdpo_cfg, "get"):
            mrp = sdpo_cfg.get("modify_ref_prompt", None)
            if mrp is not None:
                return mrp
        return {"source": "dataset", "dataset_key": "answer"}

    def _build_hint_gen_batch(self, gen_batch: DataProto, expert_cfg) -> DataProto:
        """Clone `gen_batch` and augment its `raw_prompt` messages with expert text.

        Only `non_tensor_batch["raw_prompt"]` is modified; the rollout will
        re-tokenize from the augmented messages. Samples whose expert text is
        empty are passed through unchanged so reward comparisons stay aligned.
        """
        hint_batch = DataProto(
            batch=gen_batch.batch.clone() if gen_batch.batch is not None else None,
            non_tensor_batch={k: v.copy() for k, v in gen_batch.non_tensor_batch.items()},
            meta_info=dict(gen_batch.meta_info),
        )

        expert_texts = self._get_expert_texts(hint_batch, expert_cfg)
        raw_prompts = hint_batch.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None:
            raise RuntimeError(
                "hint_rollout requires `raw_prompt` in gen_batch.non_tensor_batch; "
                "got None. Check that the dataset/agent_loop path populates raw_prompt."
            )

        # cfg can override the default; an explicit empty string disables stripping.
        strip_suffix = expert_cfg.get("strip_suffix", DEFAULT_STRIP_SUFFIX)
        if strip_suffix is None:
            strip_suffix = DEFAULT_STRIP_SUFFIX
        augmented = np.empty(len(hint_batch), dtype=object)
        for i in range(len(hint_batch)):
            base_messages = list(raw_prompts[i]) if raw_prompts[i] is not None else []
            augmented[i] = _augment_messages_with_expert(
                base_messages, expert_texts[i], strip_suffix=strip_suffix
            )
        hint_batch.non_tensor_batch["raw_prompt"] = augmented
        return hint_batch

    def _run_hint_rollout_diagnostics(
        self, gen_batch: DataProto, hint_cfg
    ) -> tuple[dict, dict, Optional[DataProto], dict]:
        """Generate `n_hint` privileged rollouts per prompt, score them, return
        (metrics_dict, uid_to_hint_solvable, hint_output, uid_to_hint_rewards).

        A prompt (uid) is labeled hint-solvable if ANY of its `n_hint` rollouts
        has summed reward > 0. The dict can then be propagated onto the main
        training batch via `_attach_hint_solvable_labels` to drive downstream
        advantage modulation.

        `hint_output` is the full DataProto of hint trajectories (with `rm_scores`
        and `uid` attached). When `sdpo.hint_rollout.merge_into_group=True`,
        `_select_and_merge_hint_rows` later concatenates a subset of these rows
        into the main training batch. Returns None when the hint path is skipped.

        `uid_to_hint_rewards` maps each uid to the per-rollout summed rewards.
        """
        n_hint = int(hint_cfg.get("n_hint", 1))
        if n_hint <= 0:
            print("[sdpo.hint_rollout] n_hint<=0, skipping. hint_solvable will be None.")
            return {}, {}, None, {}

        expert_cfg = self._get_hint_expert_cfg()
        # source="rollout" needs `responses`, which don't exist for the pre-rollout
        # hint path; skip with a print rather than fail mid-step.
        if expert_cfg.get("source", "dataset") == "rollout":
            print(
                "[sdpo.hint_rollout] expert source='rollout' is incompatible with the "
                "pre-rollout hint path; skipping. hint_solvable will be None."
            )
            return {}, {}, None, {}
        hint_input = self._build_hint_gen_batch(gen_batch, expert_cfg)
        hint_input = hint_input.repeat(repeat_times=n_hint, interleave=True)
        hint_input.meta_info["global_steps"] = self.global_steps

        hint_output = self.async_rollout_manager.generate_sequences(hint_input)
        hint_output.meta_info.pop("timing", None)

        # Agent rollout drops `input_non_tensor_batch` from its output whenever
        # agent-reward-loop is active (agent_loop.py only merges input_non_tensor_batch
        # when reward_loop_worker_handles is None). Re-attach uid from hint_input;
        # row order is preserved by asyncio.gather inside the rollout manager.
        for k in ("uid",):
            if k in hint_input.non_tensor_batch and k not in hint_output.non_tensor_batch:
                hint_output.non_tensor_batch[k] = hint_input.non_tensor_batch[k]

        # In colocate-RM mode rm_scores is not streamed by the rollout; compute it here.
        if "rm_scores" not in hint_output.batch.keys() and self.use_rm:
            hint_reward = self._compute_reward_colocate(hint_output)
            hint_output = hint_output.union(hint_reward)

        if "rm_scores" not in hint_output.batch.keys():
            print(
                "[sdpo.hint_rollout] rm_scores missing from hint_output.batch; "
                f"keys={list(hint_output.batch.keys())}. hint_solvable will be None."
            )
            return {}, {}, None, {}

        rm_scores = hint_output.batch["rm_scores"]
        per_row_reward = rm_scores.sum(-1).cpu().tolist()
        uids = hint_output.non_tensor_batch.get("uid", None)
        # breakpoint()
        if uids is None:
            print(
                "[sdpo.hint_rollout] uid missing from hint_output.non_tensor_batch; "
                f"keys={list(hint_output.non_tensor_batch.keys())}. hint_solvable will be None."
            )
            return {"sdpo/hint/mean_reward": float(np.mean(per_row_reward))}, {}, None, {}
        # Label as solvable if any of the n_hint rollouts for this uid has reward > 0.
        uid_to_solvable: dict[str, bool] = {}
        uid_to_rewards: dict[str, list] = defaultdict(list)
        for uid, r in zip(uids, per_row_reward):
            uid_to_solvable[uid] = uid_to_solvable.get(uid, False) or (r > 0)
            uid_to_rewards[uid].append(float(r))
        # breakpoint()
        n_prompts = max(1, len(uid_to_solvable))
        n_solvable = sum(1 for v in uid_to_solvable.values() if v)
        solvable_fraction = n_solvable / n_prompts
        metrics = {
            "sdpo/hint/mean_reward": float(np.mean(per_row_reward)),
            "sdpo/hint/solvable_fraction": float(solvable_fraction),
            "sdpo/hint/num_prompts": float(n_prompts),
        }
        print(
            f"[sdpo.hint_rollout] uid_to_solvable built: {n_solvable}/{n_prompts} prompts solvable "
            f"(mean_reward={metrics['sdpo/hint/mean_reward']:.3f}, n_hint={n_hint})."
        )
        return metrics, uid_to_solvable, hint_output, dict(uid_to_rewards)

    def _run_hint_rollout_no_score(
        self, gen_batch: DataProto, hint_cfg
    ) -> Optional[DataProto]:
        """Lighter-weight sibling of `_run_hint_rollout_diagnostics`: generates
        `n_hint` hint rollouts per prompt and returns the trajectory DataProto
        with `uid` attached. Does NOT compute rm_scores — the merged-rollout
        flow scores main + hint together in the standard reward pass.
        """
        n_hint = int(hint_cfg.get("n_hint", 1))
        if n_hint <= 0:
            return None
        expert_cfg = self._get_hint_expert_cfg()
        if expert_cfg.get("source", "dataset") == "rollout":
            print(
                "[sdpo.hint_rollout] expert source='rollout' incompatible with the "
                "pre-rollout hint path; skipping merge."
            )
            return None
        hint_input = self._build_hint_gen_batch(gen_batch, expert_cfg)
        hint_input = hint_input.repeat(repeat_times=n_hint, interleave=True)
        hint_input.meta_info["global_steps"] = self.global_steps
        hint_output = self.async_rollout_manager.generate_sequences(hint_input)
        hint_output.meta_info.pop("timing", None)
        for k in ("uid",):
            if k in hint_input.non_tensor_batch and k not in hint_output.non_tensor_batch:
                hint_output.non_tensor_batch[k] = hint_input.non_tensor_batch[k]
        return hint_output

    def _sdpo_is_correction_enabled(self) -> bool:
        """True when strict-unbiased IS correction for merged hint rows is on
        (sdpo.sdpo_grpo.is_correction.enabled). See _swap_hint_rows_to_unhinted."""
        sdpo_cfg = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
        if sdpo_cfg is None:
            return False
        grpo = sdpo_cfg.get("sdpo_grpo", None) if hasattr(sdpo_cfg, "get") else None
        if grpo is None:
            return False
        isc = grpo.get("is_correction", None) if hasattr(grpo, "get") else None
        return isc is not None and bool(isc.get("enabled", False))

    def _swap_hint_rows_to_unhinted(self, batch: DataProto) -> DataProto:
        """Strict-unbiased IS correction for merged hint rows (regime C).

        Hint rows enter the merged batch with HINTED input_ids = [x, e, y_hint].
        Their `old_log_probs` (computed earlier on those hinted ids) is therefore
        the behavior logprob log π_θ(y_hint | x, e). For an UNBIASED policy
        gradient on the deployment objective J(θ)=E_{y~π_θ(·|x)}[A], the score
        function must be ∇log π_θ(y_hint | x) — i.e. the actor must forward the
        UNHINTED prompt. This method swaps each hint row's prompt span back to the
        un-hinted [x] (stashed as unhinted_prompt_ids by _concat_hint_into_main_batch),
        leaving responses untouched.

        Combined with actor.use_rollout_log_probs=True (so the actor uses the
        stored old_log_probs instead of detaching), the PPO ratio for hint rows
        becomes exp(log π_θ(y|x) − log π_old(y|x,e)) = π_θ(y|x)/π_old(y|x,e),
        i.e. the IS weight — automatically bounded by the PPO clip. Main rows are
        untouched (their unhinted_prompt_ids == prompts, ratio ≈ 1).
        """
        is_hint = batch.non_tensor_batch.get("is_hint_row", None)
        if is_hint is None or "unhinted_prompt_ids" not in batch.batch:
            return batch
        is_hint_t = torch.as_tensor(np.asarray(is_hint, dtype=bool), device=batch.batch["prompts"].device)
        if not bool(is_hint_t.any()):
            return batch

        from verl.utils.model import compute_position_id_with_mask

        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attn = batch.batch["attention_mask"]
        pl = prompts.shape[1]
        unhinted_ids = batch.batch["unhinted_prompt_ids"]
        unhinted_mask = batch.batch["unhinted_prompt_mask"]

        sel = is_hint_t.unsqueeze(1)
        new_prompts = torch.where(sel, unhinted_ids, prompts)
        new_prompt_mask = torch.where(sel, unhinted_mask, attn[:, :pl])
        new_input_ids = torch.cat([new_prompts, responses], dim=1)
        new_attn = torch.cat([new_prompt_mask, attn[:, pl:]], dim=1)

        batch.batch["prompts"] = new_prompts
        batch.batch["input_ids"] = new_input_ids
        batch.batch["attention_mask"] = new_attn
        batch.batch["position_ids"] = compute_position_id_with_mask(new_attn)
        return batch

    def _sdpo_is_fn_const(self) -> float:
        """LUFFY reshape constant c for f(w)=w/(w+c) on merged hint rows.

        c<=0 → implicit-ratio mode: IS weight lives in the PPO ratio; strict
        unbiased when use_rollout_log_probs=True (see _swap_hint_rows_to_unhinted).
        c>0 → explicit mode: hint-row weight computed in the trainer, applied
        as rollout_is_weights. Requires use_rollout_log_probs=False. Read from
        sdpo.sdpo_grpo.is_correction.fn_const."""
        sdpo_cfg = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
        if sdpo_cfg is None:
            return 0.0
        grpo = sdpo_cfg.get("sdpo_grpo", None) if hasattr(sdpo_cfg, "get") else None
        if grpo is None:
            return 0.0
        isc = grpo.get("is_correction", None) if hasattr(grpo, "get") else None
        if isc is None:
            return 0.0
        return float(isc.get("fn_const", 0.0))

    def _sdpo_is_correction_params(self) -> tuple[str, float, float, float]:
        """Returns (mode, clip_low, clip_high, gamma) for the explicit IS-weight mode.

        mode: 'shaping' (default, LUFFY f(w)=w/(w+c)), 'clip' (PPO-style hard
              clip on IS ratio), or 'amplify' (LUFFY gradient modifier
              4γ²/(p+γ)² applied to positive-advantage hint tokens).
              Config: sdpo.sdpo_grpo.is_correction.mode.
        clip_low/high: bounds for mode='clip'. Config:
              sdpo.sdpo_grpo.is_correction.clip_low / .clip_high.
        gamma: inflection point for mode='amplify'. Config:
              sdpo.sdpo_grpo.is_correction.gamma.
        """
        sdpo_cfg = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
        if sdpo_cfg is None:
            return "shaping", 0.8, 1.28, 0.1
        grpo = sdpo_cfg.get("sdpo_grpo", None) if hasattr(sdpo_cfg, "get") else None
        if grpo is None:
            return "shaping", 0.8, 1.28, 0.1
        isc = grpo.get("is_correction", None) if hasattr(grpo, "get") else None
        if isc is None:
            return "shaping", 0.8, 1.28, 0.1
        mode = str(isc.get("mode", "shaping"))
        clip_low = float(isc.get("clip_low", 0.8))
        clip_high = float(isc.get("clip_high", 1.28))
        gamma = float(isc.get("gamma", 0.1))
        return mode, clip_low, clip_high, gamma

    def _build_is_rollout_weights(
        self,
        batch: DataProto,
        unhinted_logprob: torch.Tensor,
        hinted_logprob: torch.Tensor,
        c: float,
        mode: str = "shaping",
        clip_low: float = 0.8,
        clip_high: float = 1.28,
        gamma: float = 0.1,
    ) -> tuple[DataProto, dict]:
        """Off-policy correction weight for merged hint rows.

        We form the true per-token IS ratio (LUFFY paper only has the
        numerator; we have both because we sampled the hint rollouts ourselves):

            w_t = π_θ(y_t|x) / π_θ(y_t|x, e)
                = exp(unhinted_logprob - hinted_logprob)

        Then apply one of two per-token weightings, chosen by `mode`:

          mode="shaping" (LUFFY-style, default):
              f(w) = w / (w + c)
              Smooth bounded in [0, 1). Tiny w lifted to ~w/c, huge w
              saturates at 1. Config: `+sdpo.sdpo_grpo.is_correction.fn_const=c`.

          mode="clip" (PPO-style hard clip on the IS ratio, DAPO defaults):
              g(w) = clip(w, clip_low, clip_high)
              Defaults [0.8, 1.28] follow DAPO's asymmetric PPO clip
              (1-ε, 1+ε_high). Aggressive: any hint token whose IS ratio drops
              below 0.8 is snapped up to 0.8, so most hint gradient survives at
              near-full magnitude. Any explosion above 1.28 is capped.
              Config: `+sdpo.sdpo_grpo.is_correction.mode=clip
                       +sdpo.sdpo_grpo.is_correction.clip_low=0.8
                       +sdpo.sdpo_grpo.is_correction.clip_high=1.28`.

          mode="amplify" (LUFFY-style gradient modifier for rare-token boost):
              g(p) = 4·γ² / (p + γ)²    where p = π_θ(y|x)
              Multiplies the PPO-clipped surrogate loss to amplify rare
              (low-p) tokens and suppress common (high-p) ones. The multiplier
              is 1.0 at p=γ (inflection), > 1 for p<γ (boost), and → 0 for
              p >> γ. Applied ONLY on hint rows with positive advantage
              (i.e. the "guided token learning" case). Other tokens keep
              weight = 1.0 (standard PPO update).
              Config: `+sdpo.sdpo_grpo.is_correction.mode=amplify
                       +sdpo.sdpo_grpo.is_correction.gamma=0.1`.
              NOTE: this mode is compatible with `use_rollout_log_probs=True`;
              the amplifier multiplies onto the clipped PPO surrogate rather
              than replacing the ratio, so PPO's min-clip semantics remain.

        Written to batch["rollout_is_weights"] (1.0 on main rows), which
        compute_policy_loss_vanilla multiplies into the per-token pg_loss.

        For "shaping" and "clip" modes, requires a NEUTRAL PPO ratio
        (use_rollout_log_probs=False, on-policy) so the effective coefficient
        is exactly the returned weight (not weight·ratio).
        For "amplify" mode, PPO clip via `use_rollout_log_probs=True` is
        REQUIRED so the amplifier multiplies onto the true PPO clipped surrogate.
        """
        is_hint = batch.non_tensor_batch.get("is_hint_row", None)
        response_mask = batch.batch["response_mask"]
        # Clamp the log-ratio for numerical safety: exp(diff) can overflow if
        # the two forwards disagree massively (rare, but happens on padding /
        # early-dead tokens). No clamp on the individual logprobs themselves.
        w = torch.exp(torch.clamp(unhinted_logprob - hinted_logprob, min=-10.0, max=10.0))

        if mode == "shaping":
            shaped = w / (w + c)
        elif mode == "clip":
            shaped = torch.clamp(w, min=clip_low, max=clip_high)
        elif mode == "amplify":
            # p = π_θ(y|x) — raw deployment-distribution probability per hint token.
            # gradient_modifier = 4γ² / (p + γ)²    — 1.0 at p=γ, boosts p<γ, damps p>>γ.
            # Only apply to hint rows with positive advantage (guided learning);
            # other tokens stay at 1.0 (standard PPO). advantage>0 filter is done
            # below in the where(...) combining with is_hint_t.
            p = torch.exp(unhinted_logprob)
            shaped = (4.0 * (gamma ** 2)) / (p + gamma).square()
        elif mode == "amplify_linear":
            # Ablation on 'amplify': drop the square in the denominator.
            # gradient_modifier = 2γ / (p + γ)     — still 1.0 at p=γ, but softer
            # amplification on rare tokens and softer suppression on common ones.
            #   p=0.001 → 1.98   (vs 3.31 for amplify)
            #   p=0.1   → 1.00   (inflection)
            #   p=0.5   → 0.33   (vs 0.11 for amplify)
            # Same advantage-sign gating as 'amplify' (only hint & adv>0).
            p = torch.exp(unhinted_logprob)
            shaped = (2.0 * gamma) / (p + gamma)
        else:
            raise ValueError(
                f"unknown is_correction mode={mode!r}, expected "
                f"'shaping' or 'clip' or 'amplify' or 'amplify_linear'"
            )

        if is_hint is None:
            weights = torch.ones_like(shaped)
        else:
            is_hint_t = torch.as_tensor(
                np.asarray(is_hint, dtype=bool), device=shaped.device
            ).unsqueeze(1)
            if mode in ("amplify", "amplify_linear"):
                # Extra filter: only positive-advantage hint tokens get amplified.
                # Others (negative-adv hint, or all main rows) keep weight=1.
                adv = batch.batch.get("advantages", None)
                if adv is None:
                    raise RuntimeError(
                        f"mode={mode!r} requires advantages in batch; "
                        "compute_advantage must have run before _build_is_rollout_weights."
                    )
                # adv is per-token but broadcast from per-row (GRPO); take sign.
                pos_adv_t = adv > 0
                apply_mask = is_hint_t & pos_adv_t
                weights = torch.where(apply_mask, shaped, torch.ones_like(shaped))
            else:
                weights = torch.where(is_hint_t, shaped, torch.ones_like(shaped))
        batch.batch["rollout_is_weights"] = weights

        metrics: dict = {}
        if is_hint is not None:
            hint_resp = is_hint_t & response_mask.bool()
            if bool(hint_resp.any()):
                metrics["sdpo/is/mean_w"] = float(w[hint_resp].mean())
                metrics["sdpo/is/mean_shaped"] = float(shaped[hint_resp].mean())
                metrics["sdpo/is/min_w"] = float(w[hint_resp].min())
                metrics["sdpo/is/max_w"] = float(w[hint_resp].max())
                if mode in ("amplify", "amplify_linear"):
                    # Distribution of the amplifier on hint tokens with positive advantage.
                    apply_tok = apply_mask.expand_as(shaped) & response_mask.bool()
                    if bool(apply_tok.any()):
                        amp_hint = shaped[apply_tok]
                        metrics["sdpo/is/amp_mean"] = float(amp_hint.mean())
                        metrics["sdpo/is/amp_max"] = float(amp_hint.max())
                        metrics["sdpo/is/amp_min"] = float(amp_hint.min())
                        # Fraction of amplified tokens (amp>1 = boost, amp<1 = damp)
                        metrics["sdpo/is/amp_boost_frac"] = float((amp_hint > 1.0).float().mean())
                if mode == "clip":
                    # Fraction of hint tokens hitting each clip bound.
                    hint_tok = is_hint_t.expand_as(w) & response_mask.bool()
                    denom = float(max(1.0, hint_tok.sum().item()))
                    metrics["sdpo/is/clip_low_frac"] = float(
                        ((w < clip_low) & hint_tok).sum().item() / denom
                    )
                    metrics["sdpo/is/clip_high_frac"] = float(
                        ((w > clip_high) & hint_tok).sum().item() / denom
                    )
        return batch, metrics

    def _concat_hint_into_main_batch(
        self,
        batch: DataProto,
        hint_output: Optional[DataProto],
        source_batch: DataProto,
        n_hint: int,
    ) -> DataProto:
        """Concatenate hint trajectories onto the end of the main training batch.

        Tags every row with `is_hint_row` (False for main, True for hint). When
        `hint_output` is None or empty, only tags main rows and returns.

        Args:
            batch: main batch after `repeat(rollout.n)` and union with the main
                rollout output. Shape: [n_prompts * rollout.n, ...].
            hint_output: rollout output for hint prompts. Shape: [n_prompts * n_hint, ...].
            source_batch: the original 1-row-per-prompt batch from the dataloader,
                used to broadcast non-rollout columns (data_source, reward_model,
                extra_info, uid) onto the hint half.
            n_hint: rollouts per hint prompt.
        """
        batch.non_tensor_batch["is_hint_row"] = np.zeros(len(batch), dtype=bool)

        # Strict-unbiased IS correction: stash the UN-HINTED prompt span so the
        # actor update can score hint responses under [x] (see
        # _swap_hint_rows_to_unhinted). Main rows' unhinted prompt == their own
        # prompt (never hinted). Hint rows borrow [x] from a same-uid main row
        # (all rollouts of a uid share the same prompt). NOTE: source_batch is
        # PRE-rollout and (in the agent-loop dataset path) carries no tokenized
        # `prompts`, so we must read from the post-rollout main `batch` here.
        # Gated so non-IS merge runs keep the old schema.
        is_corr = self._sdpo_is_correction_enabled()
        uid_to_xprompt: dict = {}
        if is_corr:
            pl_main = batch.batch["prompts"].shape[1]
            batch.batch["unhinted_prompt_ids"] = batch.batch["prompts"].clone()
            batch.batch["unhinted_prompt_mask"] = batch.batch["attention_mask"][:, :pl_main].clone()
            main_uids = batch.non_tensor_batch.get("uid")
            if main_uids is not None:
                for i, u in enumerate(main_uids):
                    if u not in uid_to_xprompt:
                        uid_to_xprompt[u] = (
                            batch.batch["unhinted_prompt_ids"][i],
                            batch.batch["unhinted_prompt_mask"][i],
                        )

        if hint_output is None or len(hint_output) == 0:
            return batch

        # Repeat the source batch n_hint times to align with hint_output rows,
        # then union the rollout output so the hint half carries the same
        # non-rollout columns as the main half.
        hint_batch = source_batch.repeat(repeat_times=n_hint, interleave=True)
        hint_batch = hint_batch.union(hint_output)
        hint_batch.non_tensor_batch["is_hint_row"] = np.ones(len(hint_batch), dtype=bool)
        if is_corr and uid_to_xprompt:
            # Per hint row, look up the un-hinted [x] prompt of its uid (from a
            # main row). Same width (max_prompt_length) as the main half. If a uid
            # is somehow missing (shouldn't happen), fall back to the hint row's
            # own hinted prompt so the swap is a no-op there (IS weight = 1).
            hint_uids = hint_batch.non_tensor_batch.get("uid")
            pl_h = hint_batch.batch["prompts"].shape[1]
            ids_rows, mask_rows = [], []
            for j, u in enumerate(hint_uids):
                if u in uid_to_xprompt:
                    xi, xm = uid_to_xprompt[u]
                else:
                    xi = hint_batch.batch["prompts"][j]
                    xm = hint_batch.batch["attention_mask"][j, :pl_h]
                ids_rows.append(xi)
                mask_rows.append(xm)
            hint_batch.batch["unhinted_prompt_ids"] = torch.stack(ids_rows, dim=0)
            hint_batch.batch["unhinted_prompt_mask"] = torch.stack(mask_rows, dim=0)

        # response_mask is added to the main batch between rollout and reward;
        # mirror that on the hint half so the concat is shape-consistent.
        if "response_mask" in batch.batch.keys() and "response_mask" not in hint_batch.batch.keys():
            from verl.trainer.ppo.ray_trainer import compute_response_mask

            hint_batch.batch["response_mask"] = compute_response_mask(hint_batch)

        # Schema reconciliation. Rollout outputs are normally identical between
        # main and hint paths; this just guards against an asymmetric union.
        main_t_keys = set(batch.batch.keys())
        hint_t_keys = set(hint_batch.batch.keys())
        for k in main_t_keys - hint_t_keys:
            print(f"[sdpo.hint.merge] dropping main tensor key '{k}' (missing from hint).")
            batch.batch.pop(k)
        for k in hint_t_keys - main_t_keys:
            hint_batch.batch.pop(k)

        # Broadcast missing non-tensor defaults from main onto hint.
        main_n_keys = set(batch.non_tensor_batch.keys())
        hint_n_keys = set(hint_batch.non_tensor_batch.keys())
        for k in main_n_keys - hint_n_keys:
            arr = batch.non_tensor_batch[k]
            fill = arr[0] if len(arr) > 0 else None
            hint_batch.non_tensor_batch[k] = np.array(
                [fill] * len(hint_batch), dtype=arr.dtype
            )
        for k in hint_n_keys - main_n_keys:
            hint_batch.non_tensor_batch.pop(k)

        return DataProto.concat([batch, hint_batch])

    def _attach_hint_solvable_labels(self, batch: DataProto, uid_to_solvable: dict) -> None:
        """Attach a per-row `hint_solvable` bool array to `batch.non_tensor_batch`,
        derived by looking up each row's uid in `uid_to_solvable`.

        Rows whose uid is not in the dict (shouldn't happen in normal flow) get
        labeled False. Downstream code (advantage modulation, metrics) can read
        `batch.non_tensor_batch["hint_solvable"]`.
        """
        if not uid_to_solvable:
            print("[sdpo.hint_rollout] _attach: uid_to_solvable is empty; hint_solvable NOT attached.")
            return
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is None:
            print("[sdpo.hint_rollout] _attach: batch has no `uid`; hint_solvable NOT attached.")
            return
        labels = np.array([bool(uid_to_solvable.get(uid, False)) for uid in uids], dtype=bool)
        batch.non_tensor_batch["hint_solvable"] = labels
        n_solvable_rows = int(labels.sum())
        unmatched = sum(1 for u in uids if u not in uid_to_solvable)
        print(
            f"[sdpo.hint_rollout] _attach: wrote hint_solvable to {len(labels)} rows "
            f"({n_solvable_rows} solvable, {unmatched} rows with uid not in uid_to_solvable)."
        )

    def _modify_prompts_for_ref_policy(self, batch: DataProto, cfg) -> DataProto:
        """Build a new batch whose `prompts` are augmented with an expert hint
        while `responses` are unchanged. `input_ids`, `attention_mask`,
        `position_ids`, and `response_mask` are rebuilt to be consistent.
        """
        import copy as _copy

        import verl.utils.torch_functional as verl_F
        from verl.utils.model import compute_position_id_with_mask

        modified_batch = _copy.copy(batch)
        modified_batch.batch = batch.batch.clone()
        modified_batch.non_tensor_batch = dict(batch.non_tensor_batch)
        modified_batch.meta_info = dict(batch.meta_info)
        # breakpoint()
        prompt_ids = batch.batch["prompts"]
        response_ids = batch.batch["responses"]
        device = prompt_ids.device
        bsz = prompt_ids.shape[0]

        expert_texts = self._get_expert_texts(batch, cfg)
        # cfg can override the default; an explicit empty string disables stripping.
        strip_suffix = DEFAULT_STRIP_SUFFIX

        raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
        # Hint rows (merged into the batch by _select_and_merge_hint_rows) already
        # carry hinted prompts from the rollout — re-augmenting would double-hint
        # and reshape into a different chat-template string. Pass those through.
        is_hint_row = batch.non_tensor_batch.get("is_hint_row", None)
        modified_chat_prompts = []
        for i in range(bsz):
            if is_hint_row is not None and bool(is_hint_row[i]):
                modified_chat_prompts.append(None)  # sentinel: keep original ids
                continue
            if raw_prompts is not None and raw_prompts[i] is not None:
                base_messages = list(raw_prompts[i])
            else:
                # Fallback: decode original prompt as a single user message.
                decoded = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                base_messages = [{"role": "user", "content": decoded}]
            augmented = _augment_messages_with_expert(
                base_messages, expert_texts[i], strip_suffix=strip_suffix
            )
            chat_text = self.tokenizer.apply_chat_template(
                augmented, add_generation_prompt=True, tokenize=False
            )
            modified_chat_prompts.append(chat_text)

        max_prompt_length = self.config.data.max_prompt_length
        pad_token_id = self.tokenizer.pad_token_id
        truncation = self.config.data.get("truncation", "error")

        original_prompt_length = prompt_ids.shape[1]
        orig_prompt_attention_mask = batch.batch["attention_mask"][:, :original_prompt_length]

        ids_list, mask_list = [], []
        for i, chat_text in enumerate(modified_chat_prompts):
            if chat_text is None:
                # Hint row: reuse the rollout's original prompt + mask unchanged.
                ids_list.append(prompt_ids[i : i + 1].to("cpu"))
                mask_list.append(orig_prompt_attention_mask[i : i + 1].to("cpu"))
                continue
            ids, mask = verl_F.tokenize_and_postprocess_data(
                prompt=chat_text,
                tokenizer=self.tokenizer,
                max_length=max_prompt_length,
                pad_token_id=pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            ids_list.append(ids)
            mask_list.append(mask)

        modified_prompt_ids = torch.cat(ids_list, dim=0).to(device)
        modified_prompt_attention_mask = torch.cat(mask_list, dim=0).to(device)

        original_prompt_length = prompt_ids.shape[1]
        response_attention_mask = batch.batch["attention_mask"][:, original_prompt_length:]

        combined_input_ids = torch.cat([modified_prompt_ids, response_ids], dim=1)
        combined_attention_mask = torch.cat([modified_prompt_attention_mask, response_attention_mask], dim=1)
        # breakpoint()
        modified_batch.batch["prompts"] = modified_prompt_ids
        modified_batch.batch["responses"] = response_ids
        modified_batch.batch["input_ids"] = combined_input_ids
        modified_batch.batch["attention_mask"] = combined_attention_mask
        modified_batch.batch["response_mask"] = response_attention_mask
        if "position_ids" in batch.batch:
            # Vision models use 3D mrope position_ids (bsz, 4, seq_len); recomputing
            # from attention_mask alone would silently produce a 2D tensor and break
            # the worker. SDPO's modify_ref_prompt only supports text-only models.
            assert batch.batch["position_ids"].dim() == 2, (
                "modify_ref_prompt does not support multi-modal models "
                f"(got position_ids with shape {tuple(batch.batch['position_ids'].shape)}); "
                "text-only models only."
            )
            modified_batch.batch["position_ids"] = compute_position_id_with_mask(combined_attention_mask)

        modified_batch.non_tensor_batch["modified_prompt_texts"] = np.array(
            modified_chat_prompts, dtype=object
        )
        modified_batch.meta_info["ref_prompt_modified"] = True

        return modified_batch

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """Produce ref log-probs for the batch.

        Two distinct keys may be written, depending on configuration:

        - ``sdpo_ref_log_prob`` — SDPO sign-flip ref. Computed on the HINTED
          batch (prompt augmented with the expert trajectory). Source is either
          the current actor (use_current_actor=True) or the frozen
          Role.RefPolicy worker (use_current_actor=False). Consumed by
          ``compute_sdpo_grpo_advantage``.

        - ``ref_log_prob`` — standard verl origin-KL anchor. Computed on the
          ORIGINAL un-hinted batch via the frozen Role.RefPolicy worker.
          Written only when SDPO uses the current actor AND
          ``actor.use_kl_loss=True``; in that combo the SDPO key is the actor's
          own hinted logprob (useless as an origin anchor) so we add a second
          forward through the frozen ref. Consumed natively by
          ``dp_actor.py:compute_policy_loss`` via the standard ref_log_prob key.

        Outside SDPO (modify_cfg is None), only ``ref_log_prob`` is written
        (same as upstream verl behavior).
        """
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            modify_cfg = self._get_modify_ref_prompt_cfg()
            # breakpoint()
            use_current_actor = False
            if modify_cfg is not None:
                batch_for_ref = self._modify_prompts_for_ref_policy(batch, modify_cfg)
                # breakpoint()
                # SDPO sign-flip path: compute log pi(y|x, e) and write it to
                # `sdpo_ref_log_prob`. Source is the CURRENT actor (default,
                # use_current_actor=True) or a frozen Role.RefPolicy worker.
                use_current_actor = bool(modify_cfg.get("use_current_actor", True))
                if use_current_actor:
                    actor_out = self.actor_rollout_wg.compute_log_prob(batch_for_ref)
                    if "ref_log_prob" in actor_out.batch.keys():
                        sdpo_tensor = actor_out.batch["ref_log_prob"]
                    else:
                        # Standard path (non-lora actor) returns the "old_log_probs" key.
                        sdpo_tensor = actor_out.batch["old_log_probs"]
                else:
                    sdpo_out = self.ref_policy_wg.compute_ref_log_prob(batch_for_ref)
                    if "ref_log_prob" in sdpo_out.batch.keys():
                        sdpo_tensor = sdpo_out.batch["ref_log_prob"]
                    else:
                        sdpo_tensor = sdpo_out.batch["old_log_probs"]
                ref_log_prob = DataProto.from_dict(
                    tensors={"sdpo_ref_log_prob": sdpo_tensor}
                )
            else:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

            # Origin-KL second pass: fires when SDPO is using the CURRENT actor
            # as the SDPO ref AND the user enabled either actor.use_kl_loss=True
            # OR algorithm.use_kl_in_reward=True (both consume the standard
            # `ref_log_prob` key). Calls the frozen Role.RefPolicy worker (loaded
            # from actor_rollout_ref.ref.model.path, typically the initial
            # student checkpoint) on the ORIGINAL un-hinted batch and writes the
            # result to `ref_log_prob`:
            #   - dp_actor.py reads it for the KL loss term (use_kl_loss)
            #   - apply_kl_penalty reads it to subtract β·KL from rewards
            #     (use_kl_in_reward)
            use_kl_loss = bool(self.config.actor_rollout_ref.actor.get("use_kl_loss", False))
            use_kl_in_reward = bool(self.config.algorithm.get("use_kl_in_reward", False))
            need_origin_ref = use_kl_loss or use_kl_in_reward
            ref_is_real_worker = (
                getattr(self, "ref_policy_wg", None) is not None
                and self.ref_policy_wg is not self.actor_rollout_wg
                and not self.ref_in_actor
            )
            if use_current_actor and need_origin_ref and ref_is_real_worker:
                init_ref_out = self.ref_policy_wg.compute_ref_log_prob(batch)
                if "ref_log_prob" in init_ref_out.batch.keys():
                    init_tensor = init_ref_out.batch["ref_log_prob"]
                else:
                    init_tensor = init_ref_out.batch["old_log_probs"]
                ref_log_prob.batch["ref_log_prob"] = init_tensor

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            routed_experts = tu.get(output, "routed_experts")

            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            if routed_experts is not None:
                old_log_prob = tu.get_tensordict(
                    {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
                )
            else:
                old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature

        # SDPO hint_reg: pass config via meta_info (not FSDPActorConfig dataclass,
        # which rejects unknown fields). dp_actor reads batch.meta_info["hint_reg"].
        sdpo_cfg = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
        hint_reg_cfg = sdpo_cfg.get("hint_reg", None) if sdpo_cfg is not None and hasattr(sdpo_cfg, "get") else None
        if hint_reg_cfg is not None:
            # Serialize to plain dict for Ray transport.
            batch.meta_info["hint_reg"] = {
                "enabled": bool(hint_reg_cfg.get("enabled", False)),
                "gamma": float(hint_reg_cfg.get("gamma", 0.1)),
                "clip_eps": float(hint_reg_cfg.get("clip_eps", 0.2)),
                "coef": float(hint_reg_cfg.get("coef", 1.0)),
                "input_type": str(hint_reg_cfg.get("input_type", "ratio")),
            }
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            distillation_use_topk = (
                self.distillation_config.distillation_loss.loss_settings.use_topk
                if is_distillation_enabled(self.config.get("distillation"))
                else False
            )
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                distillation_use_topk=distillation_use_topk,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
                compute_loss=True,
            )
            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        # Privileged hint rollout. Two modes:
                        #   - merge_into_group=True: keep the hint trajectories, no
                        #     reward scoring here; they get concatenated onto the
                        #     main batch and scored together in the standard reward
                        #     pass. Per-uid regime selection happens in
                        #     compute_sdpo_grpo_advantage.
                        #   - merge_into_group=False (default, legacy): diagnostic
                        #     scoring only; produces uid_to_hint_solvable to gate
                        #     the SDPO sign-flip. Trajectories are discarded.
                        # Must run before sleep_replicas in either case.
                        hint_cfg = self._get_hint_rollout_cfg()
                        merge_into_group = (
                            hint_cfg is not None
                            and bool(hint_cfg.get("merge_into_group", False))
                        )
                        hint_uid_to_solvable: dict = {}
                        hint_output_for_merge: Optional[DataProto] = None
                        if hint_cfg is not None:
                            with marked_timer("hint_gen", timing_raw, color="magenta"):
                                if merge_into_group:
                                    hint_output_for_merge = self._run_hint_rollout_no_score(
                                        gen_batch, hint_cfg
                                    )
                                else:
                                    hint_metrics, hint_uid_to_solvable, _, _ = (
                                        self._run_hint_rollout_diagnostics(gen_batch, hint_cfg)
                                    )
                                    metrics.update(hint_metrics)

                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    source_batch = batch  # 1 row per prompt; needed for hint-half alignment
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # Propagate per-uid hint-solvable labels onto every row of the
                    # main batch (n copies per uid all share the same label). Available
                    # for downstream advantage modulation via batch.non_tensor_batch["hint_solvable"].
                    if hint_uid_to_solvable:
                        self._attach_hint_solvable_labels(batch, hint_uid_to_solvable)
                    # Concatenate hint rollouts onto the batch when merge_into_group=True.
                    # Also writes is_hint_row=False on the main half unconditionally so
                    # downstream code (modify_ref_prompt, advantage) can rely on it.
                    if merge_into_group and hint_cfg is not None:
                        n_hint = int(hint_cfg.get("n_hint", 1))
                        batch = self._concat_hint_into_main_batch(
                            batch=batch,
                            hint_output=hint_output_for_merge,
                            source_batch=source_batch,
                            n_hint=n_hint,
                        )
                    if self._should_compute_teacher_colocate(batch):
                        with marked_timer("teacher", timing_raw, color="cyan"):
                            batch_teacher = self._compute_teacher_colocate(batch)
                            batch = batch.union(batch_teacher)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Optional driver-side loop-detection diagnostics. Best-effort:
                    # a failure here never blocks training.
                    loop_cfg = self._get_loop_metrics_cfg()
                    if loop_cfg is not None:
                        with marked_timer("loop_metrics", timing_raw, color="magenta"):
                            try:
                                pool = self._get_or_init_loop_metrics_pool()
                                if pool is not None:
                                    from verl.trainer.distillation.loop_metrics import compute_loop_metrics

                                    loop_metrics_out = compute_loop_metrics(
                                        token_ids_2d=batch.batch["responses"],
                                        response_mask_2d=batch.batch["response_mask"],
                                        pool=pool,
                                        min_repeats=int(loop_cfg.get("min_repeats", 2)),
                                        min_pattern_tokens=int(loop_cfg.get("min_pattern_tokens", 5)),
                                        sample_fraction=float(loop_cfg.get("sample_fraction", 1.0)),
                                        rng_seed=self.global_steps,
                                        mode=str(loop_cfg.get("detector_mode", "suffix")),
                                        max_period_chunks=int(loop_cfg.get("max_period_chunks", 64)),
                                    )
                                    metrics.update(loop_metrics_out)
                            except Exception as e:
                                print(f"[loop_metrics] step {self.global_steps}: {e!r}; skipping.")

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # When merge_into_group is on, derive uid → hint_solvable from
                    # the hint rows we just scored (replaces the diagnostic-call
                    # version), then attach to every row of the batch. Also emit
                    # per-half score metrics: critic/score/mean stays main-only;
                    # critic/score_w_hint/mean covers the full n+n_hint group.
                    if merge_into_group:
                        is_hint_arr = batch.non_tensor_batch.get("is_hint_row")
                        if is_hint_arr is not None:
                            per_row = reward_tensor.sum(-1).detach().cpu().numpy()
                            uids_arr = batch.non_tensor_batch.get("uid")
                            if uids_arr is not None:
                                hint_uid_to_solvable = {}
                                for i, (u, h) in enumerate(zip(uids_arr, is_hint_arr)):
                                    if not bool(h):
                                        continue
                                    hint_uid_to_solvable[u] = (
                                        hint_uid_to_solvable.get(u, False) or (per_row[i] > 0)
                                    )
                                self._attach_hint_solvable_labels(batch, hint_uid_to_solvable)
                            main_mask = ~np.asarray(is_hint_arr, dtype=bool)
                            hint_mask = np.asarray(is_hint_arr, dtype=bool)
                            n_main = int(main_mask.sum())
                            n_hint_rows = int(hint_mask.sum())
                            score_w_hint_mean = float(per_row.mean()) if per_row.size else 0.0
                            score_main_mean = (
                                float(per_row[main_mask].mean()) if n_main > 0 else 0.0
                            )
                            score_hint_mean = (
                                float(per_row[hint_mask].mean()) if n_hint_rows > 0 else 0.0
                            )
                            metrics.update(
                                {
                                    "critic/score/mean": score_main_mean,
                                    "critic/score_w_hint/mean": score_w_hint_mean,
                                    "sdpo/hint/mean_reward": score_hint_mean,
                                    "sdpo/hint/num_hint_rows": float(n_hint_rows),
                                    "sdpo/hint/num_main_rows": float(n_main),
                                }
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # Stash sdpo.sdpo_grpo knobs into meta_info so the new
                        # `sdpo_grpo` advantage estimator (module-level) can read them
                        # without us extending compute_advantage's signature.
                        if str(self.config.algorithm.adv_estimator) == "sdpo_grpo":
                            # NOTE: use_kl_in_reward=True is now ALLOWED with sdpo_grpo.
                            # The standard apply_kl_penalty subtracts β·KL(π_cur || π_init)
                            # from token_level_rewards using the `ref_log_prob` key (frozen
                            # un-hinted origin anchor). The SDPO sign-flip uses a DIFFERENT
                            # key (`sdpo_ref_log_prob`, hinted) for its delta, so no
                            # double-counting. The previous defensive assert is removed.
                            sdpo_cfg = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
                            grpo_knobs = sdpo_cfg.get("sdpo_grpo", None) if (sdpo_cfg is not None and hasattr(sdpo_cfg, "get")) else None
                            # Materialize into a plain mutable dict so we can apply the
                            # sign-flip decay schedule without mutating the OmegaConf node.
                            if grpo_knobs is None:
                                grpo_knobs = {}
                            elif not isinstance(grpo_knobs, dict):
                                grpo_knobs = OmegaConf.to_container(grpo_knobs, resolve=True)
                            else:
                                grpo_knobs = dict(grpo_knobs)

                            # Self-distilled KL decay: scale sign_flip_lambda_{pos,neg}
                            # toward final_scale (default 0) over training so the hint
                            # delta = log π_θ(y|x,e) - log π_θ(y|x) fades as the policy
                            # matures. Configured under sdpo.sdpo_grpo.sign_flip_decay.
                            decay_cfg = grpo_knobs.get("sign_flip_decay", None)
                            if decay_cfg is not None and bool(decay_cfg.get("enabled", False)):
                                total_steps = int(
                                    decay_cfg.get("total_steps", 0)
                                    or self.config.trainer.get("total_training_steps", 0)
                                    or 0
                                )
                                mult = sign_flip_decay_multiplier(
                                    step=self.global_steps,
                                    total_steps=total_steps,
                                    decay_type=str(decay_cfg.get("decay_type", "cosine")),
                                    final_scale=float(decay_cfg.get("final_scale", 0.0)),
                                    warmup_steps=int(decay_cfg.get("warmup_steps", 0)),
                                )
                                base_pos = float(grpo_knobs.get("sign_flip_lambda_pos", 0.0))
                                base_neg = float(grpo_knobs.get("sign_flip_lambda_neg", 0.0))
                                grpo_knobs["sign_flip_lambda_pos"] = base_pos * mult
                                grpo_knobs["sign_flip_lambda_neg"] = base_neg * mult
                                metrics["sdpo/sign_flip/decay_mult"] = mult
                                metrics["sdpo/sign_flip/lambda_pos_eff"] = base_pos * mult
                                metrics["sdpo/sign_flip/lambda_neg_eff"] = base_neg * mult

                            batch.meta_info["sdpo_grpo"] = grpo_knobs

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        # Surface per-step regime counts produced inside
                        # compute_sdpo_grpo_advantage. Useful for tracking how
                        # often the "stuck student" (regime C) merge is firing.
                        if _LAST_SDPO_REGIME_COUNTS:
                            n_total = sum(_LAST_SDPO_REGIME_COUNTS.values()) or 1
                            for k, v in _LAST_SDPO_REGIME_COUNTS.items():
                                metrics[f"sdpo/regime/{k}_count"] = float(v)
                                metrics[f"sdpo/regime/{k}_frac"] = float(v) / n_total

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # IS correction for merged hint rows. The score function must
                        # be ∇log π_θ(y_hint|x), so always swap hint rows to their
                        # un-hinted prompt first. Two weighting modes:
                        if self._sdpo_is_correction_enabled():
                            fn_const = self._sdpo_is_fn_const()
                            is_mode, clip_low, clip_high, gamma_amp = self._sdpo_is_correction_params()
                            use_rlp = bool(
                                self.config.actor_rollout_ref.actor.get("use_rollout_log_probs", False)
                            )
                            # Explicit mode activates on fn_const>0 (shaping),
                            # mode='clip' (bounded IS), or mode='amplify' /
                            # 'amplify_linear' (LUFFY grad modifier variants).
                            explicit_mode = (
                                fn_const > 0.0
                                or is_mode == "clip"
                                or is_mode in ("amplify", "amplify_linear")
                            )
                            if explicit_mode:
                                # Explicit IS weight on hint rows using our own hinted
                                # rollout logprob (which LUFFY paper lacks):
                                #     w = π_θ(y|x) / π_θ(y|x, e)
                                # Then either shaping f(w)=w/(w+c), clip(w, low, high),
                                # or amplify (4γ²/(p+γ)²) — see _build_is_rollout_weights.
                                #
                                # NOTE on use_rollout_log_probs:
                                #  - shaping/clip: require it OFF (NEUTRAL PPO ratio)
                                #  - amplify: requires it ON (multiplies onto PPO clipped surrogate)
                                if is_mode in ("amplify", "amplify_linear"):
                                    if not use_rlp:
                                        print(
                                            f"[sdpo.is_correction] WARNING: mode={is_mode!r} but "
                                            "use_rollout_log_probs=False — PPO ratio collapses to 1, "
                                            "amplifier will multiply only base advantage. Set "
                                            "use_rollout_log_probs=True."
                                        )
                                elif use_rlp:
                                    print(
                                        "[sdpo.is_correction] WARNING: explicit IS mode "
                                        f"(mode={is_mode}, fn_const={fn_const}) but "
                                        "use_rollout_log_probs=True — ratio would double-count. "
                                        "Set use_rollout_log_probs=False for shaping/clip."
                                    )
                                # Grab hinted logprob BEFORE swap (batch["old_log_probs"]
                                # was populated by _compute_old_log_prob earlier this step
                                # on the hinted prompts x+e; for hint rows this is
                                # log π_θ(y_hint|x,e)).
                                hinted_old = batch.batch["old_log_probs"].clone()
                                batch = self._swap_hint_rows_to_unhinted(batch)
                                unhinted_dp, _ = self._compute_old_log_prob(batch)
                                unhinted = unhinted_dp.batch["old_log_probs"]
                                batch, is_metrics = self._build_is_rollout_weights(
                                    batch, unhinted, hinted_old, fn_const,
                                    mode=is_mode, clip_low=clip_low, clip_high=clip_high,
                                    gamma=gamma_amp,
                                )
                                metrics.update(is_metrics)
                            else:
                                # Implicit mode: strict-unbiased, IS weight = PPO ratio.
                                # Requires use_rollout_log_probs=True (else on-policy detach
                                # collapses the ratio to 1 and the IS weight vanishes).
                                if not use_rlp:
                                    print(
                                        "[sdpo.is_correction] WARNING: implicit IS mode but "
                                        "use_rollout_log_probs=False — ratio≡1, IS weight lost. "
                                        "Set use_rollout_log_probs=True or fn_const>0."
                                    )
                                batch = self._swap_hint_rows_to_unhinted(batch)

                        # SDPO hint_reg auxiliary loss mask: precompute the
                        # per-token mask over Regime C hint rows only, using
                        # the row indices exposed by compute_sdpo_grpo_advantage.
                        # Works even when regime_c_zero_hint_adv=True (advantage=0
                        # would otherwise erase the "hint & adv>0" heuristic).
                        sdpo_cfg_ref = self.config.get("sdpo", None) if hasattr(self.config, "get") else None
                        hint_reg_cfg = sdpo_cfg_ref.get("hint_reg", None) if sdpo_cfg_ref is not None and hasattr(sdpo_cfg_ref, "get") else None
                        if hint_reg_cfg is not None and bool(hint_reg_cfg.get("enabled", False)):
                            rc_hint_rows = list(_LAST_SDPO_REGC_HINT_ROWS)
                            resp_mask = batch.batch.get("response_mask", None)
                            if resp_mask is not None:
                                hint_reg_mask = torch.zeros_like(resp_mask, dtype=torch.bool)
                                if rc_hint_rows:
                                    hint_reg_mask[rc_hint_rows] = resp_mask[rc_hint_rows].bool()
                                batch.batch["hint_reg_mask"] = hint_reg_mask
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, eos_token_id=self.tokenizer.eos_token_id))
                # Re-apply main-only score override after compute_data_metrics
                # (which would otherwise overwrite critic/score/mean with the
                # full 1152-row mean, polluting it with hint rows).
                if merge_into_group:
                    is_hint_arr2 = batch.non_tensor_batch.get("is_hint_row")
                    if is_hint_arr2 is not None and "token_level_scores" in batch.batch:
                        per_row2 = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
                        main_mask2 = ~np.asarray(is_hint_arr2, dtype=bool)
                        hint_mask2 = np.asarray(is_hint_arr2, dtype=bool)
                        if main_mask2.any():
                            metrics["critic/score/mean"] = float(per_row2[main_mask2].mean())
                            metrics["critic/rewards/mean"] = float(
                                batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy()[main_mask2].mean()
                            )
                        metrics["critic/score_w_hint/mean"] = float(per_row2.mean())
                        if hint_mask2.any():
                            metrics["sdpo/hint/mean_reward"] = float(per_row2[hint_mask2].mean())
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
