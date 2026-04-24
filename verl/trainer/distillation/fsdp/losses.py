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

    # Per-token coverage = Σ p_student(t) for t ∈ teacher's top-p prefix.
    # Precomputed here so the logit-processor output contract stays (1, T) —
    # the engine wraps these as per-token jagged nested tensors downstream.
    # Consumed by the coverage_threshold hybrid mask strategy.
    hybrid_kwargs = loss_config.hybrid_mask_kwargs or {}
    top_p = float(hybrid_kwargs.get("top_p", 0.9))
    cumsum = teacher_probs_topk.cumsum(dim=-1)
    shifted = torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum[..., :-1]], dim=-1)
    in_top_p = shifted < top_p
    coverage_scores = (student_topk_log_probs.detach().exp() * in_top_p.float()).sum(dim=-1)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "coverage_scores": coverage_scores,
    }

    # 3. per-j overlap diagnostics (no_grad — diagnostic only, no activations retained)
    with torch.no_grad():
        topk = teacher_topk_ids.shape[-1]
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

        student_probs_at_teacher = student_topk_log_probs.detach().exp()
        teacher_probs_topk = teacher_topk_log_probs.exp()
        # Student probs at student's own top-k ids, for mass-weighted overlap.
        student_probs_at_student = (
            torch.gather(student_logits, dim=-1, index=student_topk_ids) - log_Z
        ).exp()
        for j in _overlap_thresholds(topk):
            outputs[f"student_mass_at_{j}"] = student_probs_at_teacher[..., :j].sum(dim=-1)
            outputs[f"teacher_mass_at_{j}"] = teacher_probs_topk[..., :j].sum(dim=-1)
            s_j = student_topk_ids[..., :j]
            t_j_sorted, sort_idx = teacher_topk_ids[..., :j].sort(dim=-1)
            teacher_lp_sorted_j = teacher_topk_log_probs[..., :j].gather(-1, sort_idx)
            pos = torch.searchsorted(t_j_sorted, s_j).clamp(max=j - 1)
            in_set = t_j_sorted.gather(-1, pos) == s_j
            in_set_f = in_set.float()
            outputs[f"overlap_ratio_at_{j}"] = in_set_f.sum(dim=-1) / j
            outputs[f"overlap_student_mass_at_{j}"] = (student_probs_at_student[..., :j] * in_set_f).sum(dim=-1)
            teacher_prob_at_s = teacher_lp_sorted_j.gather(-1, pos).exp()
            outputs[f"overlap_teacher_mass_at_{j}"] = (teacher_prob_at_s * in_set_f).sum(dim=-1)

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
    
    # --- Diagnostics: captured mass under full student ---
    # Kept for monitoring; does not participate in the CE loss.
    with torch.no_grad():
        # Chunked computation to prevent OOM with large vocabularies
        B, T, V = student_logits.shape
        CHUNK_SIZE = 2048  # Process 2048 sequence positions at once
        
        if T <= CHUNK_SIZE:
            # Small enough to compute directly
            student_log_Z_full = student_logits.logsumexp(dim=-1, keepdim=True)  # (B, T, 1)
        else:
            # Process in chunks to save memory
            log_Z_chunks = []
            for i in range(0, T, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE, T)
                chunk_log_Z = student_logits[:, i:end_idx, :].logsumexp(dim=-1, keepdim=True)
                log_Z_chunks.append(chunk_log_Z)
            student_log_Z_full = torch.cat(log_Z_chunks, dim=1)
        
        student_topk_log_probs_full = (
            torch.gather(student_logits, dim=-1, index=teacher_topk_ids) - student_log_Z_full
        )  # full-softmax student log-probs at teacher top-k
        student_mass = student_topk_log_probs_full.exp().sum(dim=-1)                  # (B, T)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }

    # Optional diagnostics: overlap metrics (unchanged, reuses student_topk_ids
    # and the full-vocab diagnostic quantities computed above).
    with torch.no_grad():
        student_probs_at_teacher = student_topk_log_probs_full.exp()
        # teacher_probs_topk already computed above, reuse it
        
        # Compute student probs at student positions only when needed
        student_probs_at_student = (
            torch.gather(student_logits, dim=-1, index=student_topk_ids) - student_log_Z_full
        ).exp()

        for j in _overlap_thresholds(topk):
            outputs[f"student_mass_at_{j}"] = student_probs_at_teacher[..., :j].sum(dim=-1)
            outputs[f"teacher_mass_at_{j}"] = teacher_probs_topk[..., :j].sum(dim=-1)

            s_j = student_topk_ids[..., :j]
            t_j_sorted, sort_idx = teacher_topk_ids[..., :j].sort(dim=-1)
            teacher_lp_sorted_j = teacher_topk_log_probs[..., :j].gather(-1, sort_idx)

            pos_j = torch.searchsorted(t_j_sorted, s_j).clamp(max=j - 1)
            in_set = t_j_sorted.gather(-1, pos_j) == s_j
            in_set_f = in_set.float()

            outputs[f"overlap_ratio_at_{j}"] = in_set_f.sum(dim=-1) / j
            outputs[f"overlap_student_mass_at_{j}"] = (student_probs_at_student[..., :j] * in_set_f).sum(dim=-1)

            teacher_prob_at_s = teacher_lp_sorted_j.gather(-1, pos_j).exp()
            outputs[f"overlap_teacher_mass_at_{j}"] = (teacher_prob_at_s * in_set_f).sum(dim=-1)

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
    del config, data_format  # unused; signature kept for backend-parity
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested

    # All overlap/mass diagnostics are gradient-free — no activations need to be retained.
    with torch.no_grad():
        teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
        teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0).long()  # (1, total_nnz, topk)
        if get_ulysses_sequence_parallel_world_size() > 1:
            teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
            teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
        assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

        topk = teacher_topk_ids.shape[-1]

        # Student top-k ids via chunked topk on raw logits (monotonic with softmax).
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

        # Mass diagnostics — avoid allocating the full (1, T, V) log_softmax.
        # log_softmax(x).gather(i) == x.gather(i) - logsumexp(x), so we only ever
        # hold (1, T, 1) logsumexp and the (1, T, topk) gathered slices.
        log_Z = student_logits.logsumexp(dim=-1, keepdim=True)  # (1, T, 1)
        student_log_probs_at_teacher = torch.gather(student_logits, dim=-1, index=teacher_topk_ids) - log_Z
        student_log_probs_at_student = torch.gather(student_logits, dim=-1, index=student_topk_ids) - log_Z
        del log_Z
        student_probs_at_teacher = student_log_probs_at_teacher.exp()
        student_probs_at_student = student_log_probs_at_student.exp()
        teacher_probs_topk = teacher_topk_log_probs.exp()

        outputs: dict[str, torch.Tensor] = {}
        for j in _overlap_thresholds(topk):
            outputs[f"student_mass_at_{j}"] = student_probs_at_teacher[..., :j].sum(dim=-1)
            outputs[f"teacher_mass_at_{j}"] = teacher_probs_topk[..., :j].sum(dim=-1)

            # Symmetric top-j overlap via binary search — avoids the (1, T, j, j) bool.
            s_j = student_topk_ids[..., :j]
            t_j_sorted, sort_idx = teacher_topk_ids[..., :j].sort(dim=-1)
            teacher_lp_sorted_j = teacher_topk_log_probs[..., :j].gather(-1, sort_idx)
            pos = torch.searchsorted(t_j_sorted, s_j).clamp(max=j - 1)
            in_set = t_j_sorted.gather(-1, pos) == s_j
            in_set_f = in_set.float()
            outputs[f"overlap_ratio_at_{j}"] = in_set_f.sum(dim=-1) / j
            outputs[f"overlap_student_mass_at_{j}"] = (student_probs_at_student[..., :j] * in_set_f).sum(dim=-1)
            teacher_prob_at_s = teacher_lp_sorted_j.gather(-1, pos).exp()
            outputs[f"overlap_teacher_mass_at_{j}"] = (teacher_prob_at_s * in_set_f).sum(dim=-1)

    return outputs
