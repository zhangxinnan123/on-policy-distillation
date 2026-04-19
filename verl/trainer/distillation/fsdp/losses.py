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
import torch.nn.functional as F

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

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    outputs = {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}

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
        for j in _overlap_thresholds(topk):
            outputs[f"student_mass_at_{j}"] = student_probs_at_teacher[..., :j].sum(dim=-1)
            outputs[f"teacher_mass_at_{j}"] = teacher_probs_topk[..., :j].sum(dim=-1)
            s_j = student_topk_ids[..., :j]
            t_j_sorted, _ = teacher_topk_ids[..., :j].sort(dim=-1)
            pos = torch.searchsorted(t_j_sorted, s_j).clamp(max=j - 1)
            in_set = t_j_sorted.gather(-1, pos) == s_j
            outputs[f"overlap_ratio_at_{j}"] = in_set.float().sum(dim=-1) / j

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
      - student_mass_at_{j}:   Σ p_student(v) for v in teacher top-j
      - teacher_mass_at_{j}:   Σ p_teacher(v) for v in teacher top-j (cumulative cdf)
      - overlap_ratio_at_{j}:  symmetric |student top-j ∩ teacher top-j| / j

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

        # Mass diagnostics — log_softmax over full vocab, freed immediately after gather.
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_log_probs_at_teacher = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
        del student_log_probs  # free (1, T, V)
        student_probs_at_teacher = student_log_probs_at_teacher.exp()
        teacher_probs_topk = teacher_topk_log_probs.exp()

        outputs: dict[str, torch.Tensor] = {}
        for j in _overlap_thresholds(topk):
            outputs[f"student_mass_at_{j}"] = student_probs_at_teacher[..., :j].sum(dim=-1)
            outputs[f"teacher_mass_at_{j}"] = teacher_probs_topk[..., :j].sum(dim=-1)

            # Symmetric top-j overlap via binary search — avoids the (1, T, j, j) bool.
            s_j = student_topk_ids[..., :j]
            t_j_sorted, _ = teacher_topk_ids[..., :j].sort(dim=-1)
            pos = torch.searchsorted(t_j_sorted, s_j).clamp(max=j - 1)
            in_set = t_j_sorted.gather(-1, pos) == s_j
            outputs[f"overlap_ratio_at_{j}"] = in_set.float().sum(dim=-1) / j

    return outputs
