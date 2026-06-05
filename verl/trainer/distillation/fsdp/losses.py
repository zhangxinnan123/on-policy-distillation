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


import torch

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def _chunked_student_topk_ids(
    student_logits: torch.Tensor, topk: int, T_chunk: int = 4096
) -> torch.Tensor:
    """Top-k ids on the vocab dim, chunked along T to bound scratch memory."""
    T_total = student_logits.shape[1]
    if T_total <= T_chunk:
        _, student_topk_ids = student_logits.topk(topk, dim=-1)
        return student_topk_ids
    parts = []
    for i in range(0, T_total, T_chunk):
        _, idx_i = student_logits[:, i : i + T_chunk].topk(topk, dim=-1)
        parts.append(idx_i)
    return torch.cat(parts, dim=1)


def _perj_overlap_diag(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    student_log_probs_at_teacher: torch.Tensor,
    log_Z: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-j student/teacher mass and symmetric top-j overlap diagnostics.

    For each j in {1, 2, 4, ..., topk} emits:
      - student_mass_at_{j}:          Σ p_student(v) for v in teacher top-j
      - teacher_mass_at_{j}:          Σ p_teacher(v) for v in teacher top-j (cumulative cdf)
      - abs_diff_at_{j}:              Σ |p_student(v) - p_teacher(v)| for v in teacher top-j
      - overlap_ratio_at_{j}:         |student top-j ∩ teacher top-j| / j
      - overlap_student_mass_at_{j}:  Σ p_student(v) for v in (student top-j ∩ teacher top-j)
      - overlap_teacher_mass_at_{j}:  Σ p_teacher(v) for v in (student top-j ∩ teacher top-j)

    Gradient-free; safe to call on a graph-tracking student_log_probs_at_teacher.
    """
    with torch.no_grad():
        topk = teacher_topk_ids.shape[-1]
        student_topk_ids = _chunked_student_topk_ids(student_logits, topk)

        student_probs_at_teacher = student_log_probs_at_teacher.detach().exp()
        teacher_probs_topk = teacher_topk_log_probs.exp()
        student_probs_at_student = (
            torch.gather(student_logits, dim=-1, index=student_topk_ids) - log_Z
        ).exp()

        outputs: dict[str, torch.Tensor] = {}
        for j in _overlap_thresholds(topk):
            outputs[f"student_mass_at_{j}"] = student_probs_at_teacher[..., :j].sum(dim=-1)
            outputs[f"teacher_mass_at_{j}"] = teacher_probs_topk[..., :j].sum(dim=-1)
            outputs[f"abs_diff_at_{j}"] = (
                student_probs_at_teacher[..., :j] - teacher_probs_topk[..., :j]
            ).abs().sum(dim=-1)
            s_j = student_topk_ids[..., :j]
            t_j_sorted, sort_idx = teacher_topk_ids[..., :j].sort(dim=-1)
            teacher_lp_sorted_j = teacher_topk_log_probs[..., :j].gather(-1, sort_idx)
            pos = torch.searchsorted(t_j_sorted, s_j).clamp(max=j - 1)
            in_set = t_j_sorted.gather(-1, pos) == s_j
            in_set_f = in_set.float()
            outputs[f"overlap_ratio_at_{j}"] = in_set_f.sum(dim=-1) / j
            outputs[f"overlap_student_mass_at_{j}"] = (
                student_probs_at_student[..., :j] * in_set_f
            ).sum(dim=-1)
            teacher_prob_at_s = teacher_lp_sorted_j.gather(-1, pos).exp()
            outputs[f"overlap_teacher_mass_at_{j}"] = (teacher_prob_at_s * in_set_f).sum(dim=-1)
        return outputs


def _coverage_scores(
    teacher_probs_topk: torch.Tensor,
    student_probs_at_teacher: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """Σ p_student(t) for t in teacher's top-p prefix on teacher top-k support."""
    cumsum = teacher_probs_topk.cumsum(dim=-1)
    shifted = torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum[..., :-1]], dim=-1)
    in_top_p = shifted < top_p
    return (student_probs_at_teacher * in_top_p.float()).sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups.
    # Avoid materializing the full (1, T, V) log_softmax — backward through
    # `log_softmax` would retain it, adding ~V/topk memory for no benefit.
    # log_softmax(x).gather(i) == x.gather(i) - logsumexp(x).
    log_Z = student_logits.logsumexp(dim=-1, keepdim=True)  # (1, T, 1)
    student_topk_log_probs = torch.gather(student_logits, dim=-1, index=teacher_topk_ids) - log_Z
    teacher_probs_topk = teacher_topk_log_probs.exp()
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_probs_topk.sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    # if loss_config.log_prob_min_clamp is not None:
    #     student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    #     teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Student probs at teacher top-k positions — detached, used by opd_theory_guided mask.
    student_topk_probs = student_topk_log_probs.detach().exp()  # (1, T, K)

    # S2 = Σ_b pi_S(b)² via logsumexp identity: exp(logsumexp(2·x) − 2·log_Z).
    # Avoids materializing the full (1, T, V) softmax. Detached — routing signal only.
    student_s2 = (student_logits.detach().mul(2).logsumexp(dim=-1) - 2 * log_Z.detach().squeeze(-1)).exp()  # (1, T)

    # Per-token coverage = Σ p_student(t) for t ∈ teacher's top-p prefix.
    # Precomputed here so the logit-processor output contract stays (1, T) —
    # the engine wraps these as per-token jagged nested tensors downstream.
    # Consumed by the coverage_threshold hybrid mask strategy.
    hybrid_kwargs = loss_config.hybrid_mask_kwargs or {}
    top_p = float(hybrid_kwargs.get("top_p", 0.9))
    coverage_scores = _coverage_scores(teacher_probs_topk, student_topk_probs, top_p)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "coverage_scores": coverage_scores,
        "student_topk_probs": student_topk_probs,
        "student_s2": student_s2,
    }

    outputs.update(
        _perj_overlap_diag(
            student_logits=student_logits,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_log_probs=teacher_topk_log_probs,
            student_log_probs_at_teacher=student_topk_log_probs,
            log_Z=log_Z,
        )
    )

    return outputs


def compute_jsd_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: "DistillationConfig",
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Generalized JSD on teacher's top-K support, with both distributions
    renormalized over K so each sums to 1.

        log p_T = teacher_topk_log_probs - logsumexp(teacher_topk_log_probs)
        log p_S = student_logits.gather(teacher_topk_ids) - logsumexp(...)
        M       = beta * p_T + (1-beta) * p_S
        JSD_b   = beta * KL(p_T || M) + (1-beta) * KL(p_S || M)

    Full-vocab logsumexp is not needed for the loss (cancels under
    K-renormalization); it is computed only for the `student_mass` /
    `teacher_mass` health diagnostics.
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, T, K)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)              # (1, T, K)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    loss_config: DistillationLossConfig = config.distillation_loss
    beta = float(loss_config.jsd_beta)
    assert 0.0 < beta < 1.0, f"jsd_beta must be in (0, 1) (got {beta})"

    # --- Full-vocab quantities (for student_mass + parity tensors used by hybrid masks) ---
    log_Z = student_logits.logsumexp(dim=-1, keepdim=True)                   # (1, T, 1)
    student_topk_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_ids)
    student_topk_log_probs_full = student_topk_logits - log_Z                 # full-vocab norm
    teacher_probs_topk = teacher_topk_log_probs.exp()
    student_mass = student_topk_log_probs_full.exp().sum(dim=-1)              # (1, T)
    teacher_mass = teacher_probs_topk.sum(dim=-1)

    # --- Renormalize over K-support for JSD math ---
    log_p_T = teacher_topk_log_probs - teacher_topk_log_probs.logsumexp(dim=-1, keepdim=True)
    log_p_S = student_topk_logits - student_topk_logits.logsumexp(dim=-1, keepdim=True)

    # log M = log(beta * p_T + (1-beta) * p_S) via logaddexp.
    log_beta = torch.log(torch.tensor(beta, device=log_p_T.device, dtype=log_p_T.dtype))
    log_1mbeta = torch.log(torch.tensor(1.0 - beta, device=log_p_T.device, dtype=log_p_T.dtype))
    log_M = torch.logaddexp(log_p_T + log_beta, log_p_S + log_1mbeta)        # (1, T, K)

    # KL(p_T || M) and KL(p_S || M); use float32 for numerical stability.
    p_T = log_p_T.exp().float()
    p_S = log_p_S.exp().float()
    log_M_f = log_M.float()
    kl_T_M = (p_T * (log_p_T.float() - log_M_f)).sum(dim=-1)                 # (1, T)
    kl_S_M = (p_S * (log_p_S.float() - log_M_f)).sum(dim=-1)
    distillation_losses = beta * kl_T_M + (1.0 - beta) * kl_S_M

    # --- Parity tensors for hybrid mask strategies (mirror compute_forward_kl_topk) ---
    student_topk_probs = student_topk_log_probs_full.detach().exp()           # (1, T, K)
    student_s2 = (
        student_logits.detach().mul(2).logsumexp(dim=-1) - 2 * log_Z.detach().squeeze(-1)
    ).exp()                                                                   # (1, T)
    hybrid_kwargs = loss_config.hybrid_mask_kwargs or {}
    top_p = float(hybrid_kwargs.get("top_p", 0.9))
    coverage_scores = _coverage_scores(teacher_probs_topk, student_topk_probs, top_p)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "jsd_kl_T_M": kl_T_M,
        "jsd_kl_S_M": kl_S_M,
        "coverage_scores": coverage_scores,
        "student_topk_probs": student_topk_probs,
        "student_s2": student_s2,
    }
    outputs.update(
        _perj_overlap_diag(
            student_logits=student_logits,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_log_probs=teacher_topk_log_probs,
            student_log_probs_at_teacher=student_topk_log_probs_full,
            log_Z=log_Z,
        )
    )
    return outputs


def compute_forward_kl_topk_approx(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """
    Union-support cross-entropy distillation loss.

    Student is normalized over the union of teacher's top-k and student's top-k
    (size ≤ 2K). Only teacher's top-k positions contribute to the CE term
    (-Σ p_teacher log π_student), but student's normalization denominator includes
    its own high-prob tokens, so probability mass leaked onto student-preferred
    tokens outside teacher's top-k is penalized.

    Compared to the earlier renormalized top-k KL, this variant:
      * replaces KL by CE (teacher entropy dropped — doesn't depend on student),
      * widens student's normalization to teacher ∪ student top-k instead of
        collapsing it to teacher top-k, so self-reinforcement on student-only
        tokens is directly visible in the denominator.

    Returns:
      - distillation_losses: (bsz, seqlen/sp_size)
      - student_mass: probability mass that full student assigns to teacher top-k
      - teacher_mass: probability mass that teacher top-k covers
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, K)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)              # (1, total_nnz, K)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    loss_config: DistillationLossConfig = config.distillation_loss
    topk = teacher_topk_ids.shape[-1]

    # --- Student top-k ids (chunked over T to bound scratch) ---
    T_CHUNK = 4096
    T_total = student_logits.shape[1]
    if T_total <= T_CHUNK:
        _, student_topk_ids = student_logits.topk(topk, dim=-1)
    else:
        parts = []
        for i in range(0, T_total, T_CHUNK):
            _, idx_i = student_logits[:, i : i + T_CHUNK].topk(topk, dim=-1)
            parts.append(idx_i)
        student_topk_ids = torch.cat(parts, dim=1)

    # --- Build union support and dedupe ---
    # Teacher occupies slots [0, K); student occupies [K, 2K). Student slots that
    # duplicate a teacher id are masked to -inf so log_softmax ignores them
    # (equivalent to a variable-size unique union without ragged tensors).
    t_sorted, _ = teacher_topk_ids.sort(dim=-1)
    pos = torch.searchsorted(t_sorted, student_topk_ids).clamp_(max=topk - 1)  # in-place clamp
    student_in_teacher = t_sorted.gather(-1, pos) == student_topk_ids  # (B, T, K)

    U_ids = torch.cat([teacher_topk_ids, student_topk_ids], dim=-1)  # (B, T, 2K)
    keep_mask = torch.cat(
        [torch.ones_like(teacher_topk_ids, dtype=torch.bool), ~student_in_teacher],
        dim=-1,
    )  # (B, T, 2K)
    del student_in_teacher  # free memory

    student_U_logits = torch.gather(student_logits, dim=-1, index=U_ids)  # (B, T, 2K)
    student_U_logits.masked_fill_(~keep_mask, float("-inf"))  # in-place fill
    del keep_mask  # free memory

    # --- Student log-softmax over the union (not over full V) ---
    student_U_log_probs = torch.log_softmax(student_U_logits.float(), dim=-1).to(student_U_logits.dtype)

    # Teacher positions inside U are the first K slots by construction.
    student_log_probs_on_teacher = student_U_log_probs[..., :topk]  # (B, T, K)

    if loss_config.log_prob_min_clamp is not None:
        student_log_probs_on_teacher = student_log_probs_on_teacher.clamp_min(loss_config.log_prob_min_clamp)

    # --- Compute teacher probs once and reuse ---
    teacher_probs_topk = teacher_topk_log_probs.exp()  # (B, T, K)
    teacher_mass = teacher_probs_topk.sum(dim=-1)      # (B, T)
    
    # --- Renormalize teacher over its top-k to a valid distribution ---
    # Equivalent to using (teacher_topk_log_probs - log(teacher_mass)) as log-probs;
    # makes the CE independent of teacher top-k coverage (each position equally weighted).
    teacher_mass_safe = teacher_mass.unsqueeze(-1).clamp_min(1e-12)            # (B, T, 1)
    teacher_probs_renorm = teacher_probs_topk / teacher_mass_safe    # (B, T, K), Σ = 1

    # --- Cross-entropy: -Σ_t p̃_teacher(t) · log π_student(t | U) ---
    distillation_losses = -(teacher_probs_renorm * student_log_probs_on_teacher).sum(dim=-1)

    # --- Diagnostics under full-vocab softmax (no grad) ---
    # Kept for monitoring; does not participate in the CE loss.
    with torch.no_grad():
        # Chunked logsumexp to bound peak scratch on large vocabularies.
        T = student_logits.shape[1]
        CHUNK_SIZE = 2048
        if T <= CHUNK_SIZE:
            student_log_Z_full = student_logits.logsumexp(dim=-1, keepdim=True)  # (B, T, 1)
        else:
            log_Z_chunks = []
            for i in range(0, T, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE, T)
                log_Z_chunks.append(student_logits[:, i:end_idx, :].logsumexp(dim=-1, keepdim=True))
            student_log_Z_full = torch.cat(log_Z_chunks, dim=1)

        student_topk_log_probs_full = (
            torch.gather(student_logits, dim=-1, index=teacher_topk_ids) - student_log_Z_full
        )
        student_topk_probs = student_topk_log_probs_full.exp()  # (B, T, K)
        student_mass = student_topk_probs.sum(dim=-1)            # (B, T)

        # S2 = Σ_b pi_S(b)² for opd_* mask strategies (non-hybrid PG ratio diagnostic).
        student_s2 = (
            student_logits.mul(2).logsumexp(dim=-1) - 2 * student_log_Z_full.squeeze(-1)
        ).exp()  # (B, T)

        hybrid_kwargs = loss_config.hybrid_mask_kwargs or {}
        top_p = float(hybrid_kwargs.get("top_p", 0.9))
        coverage_scores = _coverage_scores(teacher_probs_topk, student_topk_probs, top_p)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "coverage_scores": coverage_scores,
        "student_topk_probs": student_topk_probs,
        "student_s2": student_s2,
    }
    outputs.update(
        _perj_overlap_diag(
            student_logits=student_logits,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_log_probs=teacher_topk_log_probs,
            student_log_probs_at_teacher=student_topk_log_probs_full,
            log_Z=student_log_Z_full,
        )
    )

    return outputs


def _overlap_thresholds(topk: int) -> list[int]:
    """[1, 2, 4, 8, ...] up to topk, with topk always appended once."""
    out: list[int] = [1]
    j = 2
    while j < topk:
        out.append(j)
        j *= 2
    if topk != out[-1]:
        out.append(topk)
    return out


def compute_student_topk_overlap_k1(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Per-j top-k diagnostics. Memory stays within compute_forward_kl_topk's envelope.

    For each j in {1, 2, 4, ..., topk} emits:
      - student_mass_at_{j}:          Σ p_student(v) for v in teacher top-j
      - teacher_mass_at_{j}:          Σ p_teacher(v) for v in teacher top-j (cumulative cdf)
      - overlap_ratio_at_{j}:         symmetric |student top-j ∩ teacher top-j| / j
      - overlap_student_mass_at_{j}:  Σ p_student(v) for v in (student top-j ∩ teacher top-j)
      - overlap_teacher_mass_at_{j}:  Σ p_teacher(v) for v in (student top-j ∩ teacher top-j)

    The k1 loss itself is computed in the outer registered loss fn.

    Memory notes:
      - `.topk` on the vocab dim is chunked along T so its scratch never exceeds
        what forward_kl_topk's log_softmax allocates.
      - Overlap uses `searchsorted` (O(T·j) memory) rather than the full pairwise
        bool (O(T·j²)).
    """
    del data_format  # unused; signature kept for backend-parity
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested

    # All overlap/mass diagnostics are gradient-free — no activations need to be retained.
    with torch.no_grad():
        teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
        teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0).long()  # (1, total_nnz, topk)
        if get_ulysses_sequence_parallel_world_size() > 1:
            teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
            teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
        assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

        # Mass diagnostics — avoid allocating the full (1, T, V) log_softmax.
        # log_softmax(x).gather(i) == x.gather(i) - logsumexp(x), so we only ever
        # hold (1, T, 1) logsumexp and the (1, T, topk) gathered slices.
        log_Z = student_logits.logsumexp(dim=-1, keepdim=True)  # (1, T, 1)
        student_log_probs_at_teacher = torch.gather(student_logits, dim=-1, index=teacher_topk_ids) - log_Z
        student_topk_probs = student_log_probs_at_teacher.exp()
        teacher_probs_topk = teacher_topk_log_probs.exp()

        # Global student/teacher mass on teacher top-k support (parity with
        # compute_forward_kl_topk so any top-k loss can surface these).
        student_mass = student_topk_probs.sum(dim=-1)
        teacher_mass = teacher_probs_topk.sum(dim=-1)

        # S2 = Σ_b pi_S(b)² via logsumexp identity — needed by opd_* mask strategies
        # for the non-hybrid PG ratio diagnostic.
        student_s2 = (
            student_logits.mul(2).logsumexp(dim=-1) - 2 * log_Z.squeeze(-1)
        ).exp()  # (1, T)

        # Diagnostic-only coverage = Σ p_student(t) for t ∈ teacher's top-p prefix.
        # Mirrors compute_forward_kl_topk so the registered loss can emit
        # top_p_coverage_* metrics for cross-mode comparison. No mask routing here.
        hybrid_kwargs = config.distillation_loss.hybrid_mask_kwargs or {}
        top_p = float(hybrid_kwargs.get("top_p", 0.9))
        coverage_scores = _coverage_scores(teacher_probs_topk, student_topk_probs, top_p)

        outputs: dict[str, torch.Tensor] = {
            "student_mass": student_mass,
            "teacher_mass": teacher_mass,
            "coverage_scores": coverage_scores,
            "student_topk_probs": student_topk_probs,
            "student_s2": student_s2,
            # log_Z exposed for the update_consistency callback: lets it
            # recover Δlogit_u = Δlog π_u + Δlog_Z (unnormalized logit
            # diagnostic). Not used by any loss term itself. Cast to fp32 so
            # the callback's Δ measurement isn't bf16-quantized.
            "student_log_Z": log_Z.squeeze(-1).to(torch.float32),  # (1, T)
        }
        outputs.update(
            _perj_overlap_diag(
                student_logits=student_logits,
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_log_probs=teacher_topk_log_probs,
                student_log_probs_at_teacher=student_log_probs_at_teacher,
                log_Z=log_Z,
            )
        )

    return outputs
