# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""In-flight update-consistency measurement callback.

Around each `_update_actor`, we run `compute_log_prob` on the **current
training batch** twice — once before the optimizer step and once after — and
diff the per-position student probabilities. No frozen probe set, no loss-fn
swap: the training loss (`k1_topk_overlap` + `use_policy_gradient=True`) is
already the REINFORCE single-sample RKL update the theory predicts, and its
logits-processor hook already emits `student_topk_probs` (at teacher's top-K
positions) and `student_s2 = ||π_θ||²` into model_output.

Candidate set per response position:  teacher_top_K ∪ {sampled_u}.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class UpdateConsistencyCallback:
    """Records per-(step, sequence_id, response_pos, candidate-token) rows."""

    def __init__(
        self,
        output_dir: str,
        measure_every: int = 1,
        single_token_mode: bool = False,
        seed: int = 0,
        noise_check: bool = False,
        lr: Optional[float] = None,
    ):
        self.output_dir = Path(output_dir)
        self.measure_every = max(1, int(measure_every))
        # Optimizer learning rate η — used to compute absolute predicted
        # magnitude (η·|A|·shape_factor). If None, predicted_absolute_magnitude
        # column will be NaN in the parquet and only the η-free
        # predicted_relative_magnitude (shape factor) is reliable.
        self.lr = float(lr) if lr is not None else None
        # When True, on the first measurement step take TWO snapshots back-to-back
        # (no optimizer step in between), compute their per-tensor differences,
        # and log determinism stats. If forward is deterministic, all diffs are
        # bitwise 0. If not, every Δπ we report later is contaminated by that
        # noise floor, and sign agreement is upper-bounded by it.
        self.noise_check = bool(noise_check)
        self._noise_check_done = False
        # When True: before super()._update_actor, override response_mask to be
        # 1-hot at ONE randomly-chosen valid position per batch, so gradient
        # comes from a single token only. Measurement still happens at all
        # valid positions — non-target positions then quantify NTK off-diagonal
        # coupling (how a single-token gradient leaks into other positions
        # through shared parameters).
        self.single_token_mode = bool(single_token_mode)
        self._rng = np.random.default_rng(int(seed))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.measurements_path = self.output_dir / "measurements.parquet"

        self._snapshot_before: Optional[dict[str, torch.Tensor]] = None
        # Per-sequence target j_i when single_token_mode=true; -1 means "no
        # valid position in this sequence" (-> contributed nothing to grad).
        self._target_pos_per_seq: Optional[np.ndarray] = None
        self._original_response_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------- snapshot machinery

    def _snapshot(self, trainer: "RayPPOTrainer", batch: "DataProto") -> dict[str, torch.Tensor]:
        """Run compute_log_prob on the training batch with the logits-processor
        hook enabled. Returns response-sliced per-position tensors.

        The training loss_fn (already installed on the actor worker) is
        k1_topk_overlap-style; its logits processor emits student_topk_probs
        (at teacher's top-K) and student_s2 into model_output.

        Returns:
            log_probs        : (B, R)   — π_S at sampled token (in log space)
            entropy          : (B, R)   — full-vocab entropy
            student_topk_probs : (B, R, K) — π_S at teacher_top_K positions
            student_s2       : (B, R)   — ||π_θ||²
        """
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

        # Build our own tensordict view of the batch — don't mutate the caller's
        # DataProto since super()._update_actor will reuse it.
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        # `_update_actor` normally sets `global_batch_size` before dispatching
        # (ray_trainer.py:1303); the loss_fn's final phase reads it from data
        # (distillation/losses.py:678). Our compute_log_prob → infer_batch
        # path bypasses that, so we set it here ourselves. `batch_num_tokens`
        # and `dp_size` are populated by the engine itself inside
        # forward_backward_batch (fsdp/transformer_impl.py:600-601).
        global_batch_size = int(batch.batch["input_ids"].shape[0])
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            compute_loss=True,            # logits processor only fires when loss_fn is non-None
            distillation_use_topk=True,   # tells the engine to call the logits processor
            global_batch_size=global_batch_size,
        )

        output = trainer.actor_rollout_wg.compute_log_prob(batch_td)

        log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)             # (B, R)
        entropy = no_padding_2_padding(tu.get(output, "entropy"), batch_td)                 # (B, R)
        student_topk_probs = no_padding_2_padding(tu.get(output, "student_topk_probs"), batch_td)  # (B, R, K)
        student_s2 = no_padding_2_padding(tu.get(output, "student_s2"), batch_td)           # (B, R)
        # log_Z is fp32 from the modified kernel — lets us recover unnormalized
        # logits (Δlogit_u = Δlog π_u + Δlog_Z). May not exist if the kernel
        # in use is older / different.
        log_Z_nested = tu.get(output, "student_log_Z", None)
        if log_Z_nested is not None:
            student_log_Z = no_padding_2_padding(log_Z_nested, batch_td).float()
        else:
            student_log_Z = None

        return {
            "log_probs": log_probs.float(),
            "entropy": entropy.float(),
            "student_topk_probs": student_topk_probs.float(),
            "student_s2": student_s2.float(),
            "student_log_Z": student_log_Z,
        }

    # ---------------------------------------------------------- trainer hooks

    def _should_measure(self, step: int) -> bool:
        return (step % self.measure_every) == 0 and step > 0

    def _install_single_token_mask(self, batch: "DataProto") -> None:
        """Replace batch.response_mask with a per-sequence 1-hot mask: for
        each sequence i in the batch, pick one random valid response position
        j_i and set mask[i, j_i] = 1, all other positions = 0.

        Why per-sequence rather than 1 target across the whole batch: with
        DP > 1, ranks that don't get the target sequence end up with
        response_mask.sum() == 0 in their loss computation, which crashes
        diagnostic reductions (.min() on an empty tensor). Per-sequence
        targeting keeps every rank with ≥1 valid token while keeping the
        gradient sparse — total contributing tokens = batch_size (e.g. 8)
        instead of ~250+ in the baseline run.

        Records the chosen target position per sequence in
        `self._target_pos_per_seq` (a (B,) int array, -1 where no valid
        position existed). The (i, j) → is_target check in _write_rows
        uses this per-row mapping.
        """
        original = batch.batch["response_mask"]
        self._original_response_mask = original.clone()

        B = original.shape[0]
        new_mask = torch.zeros_like(original)
        target_j = np.full(B, -1, dtype=np.int64)

        for i in range(B):
            valid_j = torch.nonzero(original[i].bool(), as_tuple=False).flatten().cpu().numpy()
            if len(valid_j) == 0:
                continue
            j_star = int(self._rng.choice(valid_j))
            new_mask[i, j_star] = 1
            target_j[i] = j_star

        if (target_j == -1).all():
            logger.warning("[update_consistency] no valid positions in any sequence; skipping single-token override")
            self._target_pos_per_seq = None
            return

        batch.batch["response_mask"] = new_mask
        self._target_pos_per_seq = target_j

    def _restore_response_mask(self, batch: "DataProto") -> None:
        if self._original_response_mask is not None:
            batch.batch["response_mask"] = self._original_response_mask
            self._original_response_mask = None

    def _run_noise_check(self, trainer: "RayPPOTrainer", batch: "DataProto") -> None:
        """Take TWO snapshots back-to-back with no parameter update in between.
        Log per-key max/mean abs diff. If forward is fully deterministic the
        diffs are all 0; non-zero diffs upper-bound the achievable Δπ sign
        resolution."""
        snap_a = self._snapshot(trainer, batch)
        snap_b = self._snapshot(trainer, batch)
        msg = ["[noise_check] same-batch two-forward determinism test:"]
        for key in snap_a.keys():
            a = snap_a[key]
            b = snap_b[key]
            diff = (b - a).abs()
            msg.append(
                f"  {key:>25s}: max|Δ|={diff.max().item():.3e}  "
                f"mean|Δ|={diff.mean().item():.3e}  "
                f"shape={tuple(a.shape)}"
            )
        # student_topk_probs and log_probs are the ones we actually use for Δπ.
        # If their max|Δ| > 1e-5 the per-token sign measurement is unreliable.
        print("\n".join(msg), flush=True)
        logger.info("\n".join(msg))

    def before_update(self, trainer: "RayPPOTrainer", batch: "DataProto") -> None:
        if not self._should_measure(trainer.global_steps):
            return
        try:
            if self.noise_check and not self._noise_check_done:
                self._run_noise_check(trainer, batch)
                self._noise_check_done = True
            self._snapshot_before = self._snapshot(trainer, batch)
            if self.single_token_mode:
                # Install single-token mask AFTER snapshot, so the snapshot
                # forward uses the full mask (mask doesn't affect logits, but
                # being explicit avoids accidental coupling).
                self._install_single_token_mask(batch)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            msg = f"[update_consistency] before_update failed at step {trainer.global_steps}: {e!r}\n{tb}"
            logger.warning(msg)
            print(msg, flush=True)
            self._snapshot_before = None
            self._restore_response_mask(batch)
            self._target_pos_per_seq = None

    def after_update(self, trainer: "RayPPOTrainer", batch: "DataProto") -> None:
        if self._snapshot_before is None:
            print(f"[update_consistency] after_update step {trainer.global_steps}: _snapshot_before is None, skipping", flush=True)
            return
        try:
            # Restore the original mask before the post-update snapshot so that
            # the snapshot covers ALL valid positions (including the target).
            self._restore_response_mask(batch)
            after = self._snapshot(trainer, batch)
            self._write_rows(trainer.global_steps, self._snapshot_before, after, batch)
            print(f"[update_consistency] step {trainer.global_steps}: wrote measurement rows", flush=True)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            msg = f"[update_consistency] after_update failed at step {trainer.global_steps}: {e!r}\n{tb}"
            logger.warning(msg)
            print(msg, flush=True)
        finally:
            self._restore_response_mask(batch)
            self._snapshot_before = None
            self._target_pos_per_seq = None

    # ----------------------------------------------------------- measurement

    def _write_rows(
        self,
        step: int,
        before: dict[str, torch.Tensor],
        after: dict[str, torch.Tensor],
        batch: "DataProto",
    ) -> None:
        import pandas as pd

        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

        # The training-batch teacher tensors are stored over the FULL sequence
        # (B, S, K) / (B, S). Slice them down to response-aligned form the same
        # way before/after did, so all indices line up.
        td_for_slicing = batch.to_tensordict()
        td_for_slicing = left_right_2_no_padding(td_for_slicing)

        def _slice(name):
            t = td_for_slicing.get(name, None)
            if t is None:
                return None
            return no_padding_2_padding(t, td_for_slicing).float()

        teacher_logprobs = _slice("teacher_logprobs")        # (B, R, K)
        teacher_ids_t = _slice("teacher_ids").long() if _slice("teacher_ids") is not None else None
        teacher_next_token_logprobs = _slice("teacher_next_token_logprobs")  # (B, R)

        assert teacher_logprobs is not None and teacher_ids_t is not None, (
            "Training batch must have teacher_logprobs/teacher_ids — check that the "
            "training distillation loss is a top-K family (e.g. k1_topk_overlap)."
        )
        assert teacher_next_token_logprobs is not None, (
            "Training batch must have teacher_next_token_logprobs."
        )

        responses = batch.batch["responses"].cpu().numpy()                       # (B, R)
        response_mask = batch.batch["response_mask"].cpu().numpy().astype(bool)  # (B, R)
        teacher_lp_np = teacher_logprobs.cpu().numpy()                            # (B, R, K)
        teacher_ids_np = teacher_ids_t.cpu().numpy()                              # (B, R, K)
        teacher_u_lp_np = teacher_next_token_logprobs.cpu().numpy()               # (B, R)

        log_prob_b = before["log_probs"].cpu().numpy()           # (B, R) — log π_S^before at sampled
        log_prob_a = after["log_probs"].cpu().numpy()            # (B, R) — log π_S^after at sampled
        topk_b = before["student_topk_probs"].cpu().numpy()       # (B, R, K) — π_S^before at teacher top-K
        topk_a = after["student_topk_probs"].cpu().numpy()        # (B, R, K)
        entropy_b = before["entropy"].cpu().numpy()              # (B, R)
        norm_sq = before["student_s2"].cpu().numpy()             # (B, R)

        pi_S_u_b = np.exp(log_prob_b)
        pi_S_u_a = np.exp(log_prob_a)
        pi_T_u = np.exp(teacher_u_lp_np)
        pi_T_topk = np.exp(teacher_lp_np)

        # A and τ per (i, j), both from BEFORE state
        A_per_pos = np.log(np.maximum(pi_T_u, 1e-30)) - np.log(np.maximum(pi_S_u_b, 1e-30))  # (B, R)
        tau_per_pos = norm_sq - pi_S_u_b                                                      # (B, R)
        sign_A = np.sign(A_per_pos)                                                           # (B, R)

        # Δπ at the K teacher candidates
        delta_topk = topk_a - topk_b                                                          # (B, R, K)
        # Sign predictions for c ≠ u: sign(A) · sign(τ - π_S(c))
        sign_tau_minus_pi = np.sign(tau_per_pos[:, :, None] - topk_b)                         # (B, R, K)
        predicted_sign_topk = (sign_A[:, :, None] * sign_tau_minus_pi).astype(np.int8)
        actual_sign_topk = np.sign(delta_topk).astype(np.int8)
        predicted_rel_mag_topk = topk_b * np.abs(tau_per_pos[:, :, None] - topk_b)
        # Absolute prediction: |Δπ(c)| ≈ η · |A| · π_S(c) · |τ − π_S(c)| for c ≠ u.
        # If lr is unknown (e.g. callback constructed without it), NaN-fill.
        if self.lr is not None:
            predicted_abs_mag_topk = self.lr * np.abs(A_per_pos[:, :, None]) * predicted_rel_mag_topk
        else:
            predicted_abs_mag_topk = np.full(predicted_rel_mag_topk.shape, np.nan, dtype=np.float32)
        is_above_watershed_topk = topk_b > tau_per_pos[:, :, None]                            # (B, R, K)

        # === LOGIT-LEVEL prediction (cleanest theory test, no softmax nonlinearity) ===
        # Theory:  Δℓ_u = +ηA·(1−π_S(u)),  Δℓ_c = −ηA·π_S(c)
        #   →  Δ(ℓ_u − ℓ_c) = ηA·(1 − π_S(u) + π_S(c))
        #   Since 0 ≤ π_S ≤ 1, (1 − π_S(u) + π_S(c)) ≥ 0 always,
        #   so sign(Δ(ℓ_u − ℓ_c)) = sign(A) for every c ≠ u, no τ dependence.
        # log_Z cancels in (ℓ_u − ℓ_c) so this is measurable from log_probs only:
        #   ℓ_u − ℓ_c = log π_S(u) − log π_S(c)
        # (B, R, K), per-candidate (each c is teacher_topk_ids[i,j,k])
        log_pi_u_b = log_prob_b[:, :, None]                       # (B, R, 1)
        log_pi_u_a = log_prob_a[:, :, None]
        # Clamp before log to avoid -inf when student probability is exactly 0
        log_pi_c_b = np.log(np.maximum(topk_b, 1e-45))            # (B, R, K)
        log_pi_c_a = np.log(np.maximum(topk_a, 1e-45))
        delta_logit_diff_topk = (log_pi_u_a - log_pi_c_a) - (log_pi_u_b - log_pi_c_b)         # (B, R, K)
        predicted_sign_logit_topk = np.broadcast_to(sign_A[:, :, None], delta_logit_diff_topk.shape).astype(np.int8)
        actual_sign_logit_topk = np.sign(delta_logit_diff_topk).astype(np.int8)

        # === ABSOLUTE LOGIT prediction (using student_log_Z to recover unnormalized logit) ===
        # Δlogit_u = Δlog π_u + Δlog_Z (since log π_u = logit_u − log_Z).
        # Theory: Δlogit_u = +ηA(1 − π_u) → sign(Δlogit_u) = sign(A).
        # Similarly Δlogit_c = Δlog π_c + Δlog_Z, theory: sign = −sign(A) for c ≠ u.
        # If student_log_Z is not present in the snapshot (e.g. older kernel),
        # NaN out the absolute-logit columns.
        if before.get("student_log_Z") is not None and after.get("student_log_Z") is not None:
            log_Z_b = before["student_log_Z"].cpu().numpy()                              # (B, R)
            log_Z_a = after ["student_log_Z"].cpu().numpy()
            delta_log_Z = log_Z_a - log_Z_b                                              # (B, R)
            delta_logit_u_abs = (log_prob_a - log_prob_b) + delta_log_Z                  # (B, R)
            delta_logit_c_abs = (log_pi_c_a - log_pi_c_b) + delta_log_Z[:, :, None]      # (B, R, K)
            predicted_sign_logit_abs_u = sign_A.astype(np.int8)                          # (B, R) — +sign(A)
            actual_sign_logit_abs_u = np.sign(delta_logit_u_abs).astype(np.int8)
            # Broadcast to (B, R, K) so per-candidate indexing [i, j, k] works.
            K_topk = topk_b.shape[-1]
            predicted_sign_logit_abs_c = np.broadcast_to(
                (-sign_A[:, :, None]).astype(np.int8), (sign_A.shape[0], sign_A.shape[1], K_topk),
            )
            actual_sign_logit_abs_c = np.sign(delta_logit_c_abs).astype(np.int8)
            have_abs_logit = True
        else:
            delta_log_Z = np.full(log_prob_b.shape, np.nan)
            delta_logit_u_abs = np.full(log_prob_b.shape, np.nan)
            delta_logit_c_abs = np.full(topk_b.shape, np.nan)
            predicted_sign_logit_abs_u = np.zeros_like(sign_A).astype(np.int8)
            actual_sign_logit_abs_u = np.zeros_like(sign_A).astype(np.int8)
            predicted_sign_logit_abs_c = np.zeros(topk_b.shape, dtype=np.int8)
            actual_sign_logit_abs_c = np.zeros(topk_b.shape, dtype=np.int8)
            have_abs_logit = False

        # Δπ at sampled u
        # Theory: Δπ_u ≈ ηA · π_u · (1 − 2π_u + ‖π‖²) = ηA · π_u · (1 + τ − π_u).
        # The factor (1 − 2π_u + ‖π‖²) = (1−π_u)² + Σ_{v≠u} π_v² ≥ 0 always, so the
        # sign is always sign(A) (no watershed at sampled u) and no abs() is needed
        # in the relative-magnitude prediction. Using `π_u · |τ − π_u|` here (the
        # c ≠ u formula) drops the +1 and is wrong for u — for typical π_u ≪ 1 it
        # underpredicts the magnitude by orders of magnitude.
        delta_u = pi_S_u_a - pi_S_u_b                                                         # (B, R)
        predicted_sign_u = sign_A.astype(np.int8)                                             # (B, R)
        actual_sign_u = np.sign(delta_u).astype(np.int8)
        predicted_rel_mag_u = pi_S_u_b * (1.0 - 2.0 * pi_S_u_b + norm_sq)
        # Absolute prediction: |Δπ(u)| ≈ η · |A| · π_u · (1 − 2π_u + ‖π‖²) (always positive).
        if self.lr is not None:
            predicted_abs_mag_u = self.lr * np.abs(A_per_pos) * predicted_rel_mag_u
        else:
            predicted_abs_mag_u = np.full(predicted_rel_mag_u.shape, np.nan, dtype=np.float32)
        # `is_above_watershed_u` has no theoretical meaning at the sampled token —
        # sign(Δπ_u) = sign(A) regardless of where π_u sits relative to τ. Kept for
        # schema parity; downstream analysis should not stratify sampled-u rows by it.
        is_above_watershed_u = pi_S_u_b > tau_per_pos

        # Per-sequence target position when single_token_mode; -1 (= invalid)
        # otherwise. is_target marks the single (i, j) per sequence that
        # contributed to the gradient.
        target_per_seq = self._target_pos_per_seq if self.single_token_mode else None

        rows: list[dict] = []
        valid_idx = np.nonzero(response_mask)
        # `B` here is the batch size; we use the batch row index as `sequence_id`.
        for i, j in zip(*valid_idx):
            i = int(i)
            j = int(j)
            target_j_for_i = -1 if target_per_seq is None else int(target_per_seq[i])
            is_target_pos = (j == target_j_for_i)

            # K teacher-topK candidates. When the sampled token u is inside
            # teacher's top-K we MUST skip that slot here: the c ≠ u formulas
            # for predicted_sign / predicted_relative_magnitude don't apply to u
            # (sign(Δπ_u) = sign(A), not sign(A)·sign(τ−π_u); magnitude uses the
            # 1 + τ − π_u factor). The dedicated sampled-u row below handles it.
            u_id = int(responses[i, j])
            for k in range(teacher_ids_np.shape[2]):
                c = int(teacher_ids_np[i, j, k])
                if c == u_id:
                    continue
                pi_S_b = float(topk_b[i, j, k])
                pi_S_a = float(topk_a[i, j, k])
                rows.append(
                    {
                        "step": int(step),
                        "sequence_id": i,
                        "response_pos": j,
                        "candidate_token_id": c,
                        "is_sampled_u": False,
                        "is_above_watershed": bool(is_above_watershed_topk[i, j, k]),
                        "pi_S_before": pi_S_b,
                        "pi_S_after": pi_S_a,
                        "pi_T": float(pi_T_topk[i, j, k]),
                        "delta_pi_actual": float(delta_topk[i, j, k]),
                        "predicted_sign": int(predicted_sign_topk[i, j, k]),
                        "actual_sign": int(actual_sign_topk[i, j, k]),
                        "predicted_relative_magnitude": float(predicted_rel_mag_topk[i, j, k]),
                        # Absolute prediction: η · |A| · π_S(c) · |τ − π_S(c)| for c ≠ u.
                        # NaN if lr was not passed to the callback.
                        "predicted_absolute_magnitude": float(predicted_abs_mag_topk[i, j, k]),
                        # c ≠ u is guaranteed by the `continue` above, so these
                        # always take the non-sampled branch.
                        "delta_logit_u_minus_c": float(delta_logit_diff_topk[i, j, k]),
                        "predicted_sign_logit": int(predicted_sign_logit_topk[i, j, k]),
                        "actual_sign_logit": int(actual_sign_logit_topk[i, j, k]),
                        # Absolute (unnormalized) logit Δ: requires student_log_Z
                        # from the modified kernel. For c ≠ u: Δlogit_c.
                        "delta_logit_abs": float(delta_logit_c_abs[i, j, k]),
                        "predicted_sign_logit_abs": int(predicted_sign_logit_abs_c[i, j, k]),
                        "actual_sign_logit_abs": int(actual_sign_logit_abs_c[i, j, k]),
                        "delta_log_Z": float(delta_log_Z[i, j]),
                        "tau": float(tau_per_pos[i, j]),
                        "A_value": float(A_per_pos[i, j]),
                        "norm_sq_S": float(norm_sq[i, j]),
                        "entropy_S": float(entropy_b[i, j]),
                        "target_j_for_seq": target_j_for_i,
                        "is_target": is_target_pos,
                    }
                )

            # 1 sampled-u candidate
            rows.append(
                {
                    "step": int(step),
                    "sequence_id": i,
                    "response_pos": j,
                    "candidate_token_id": int(responses[i, j]),
                    "is_sampled_u": True,
                    "is_above_watershed": bool(is_above_watershed_u[i, j]),
                    "pi_S_before": float(pi_S_u_b[i, j]),
                    "pi_S_after": float(pi_S_u_a[i, j]),
                    "pi_T": float(pi_T_u[i, j]),
                    "delta_pi_actual": float(delta_u[i, j]),
                    "predicted_sign": int(predicted_sign_u[i, j]),
                    "actual_sign": int(actual_sign_u[i, j]),
                    "predicted_relative_magnitude": float(predicted_rel_mag_u[i, j]),
                    # Absolute prediction: η · |A| · π_u · (1 − 2π_u + ‖π‖²) for c = u.
                    "predicted_absolute_magnitude": float(predicted_abs_mag_u[i, j]),
                    # c == u: relative-logit Δ trivially 0, so NaN
                    "delta_logit_u_minus_c": float("nan"),
                    "predicted_sign_logit": 0,
                    "actual_sign_logit": 0,
                    # Absolute logit at sampled u: this IS measurable and is
                    # the cleanest single-direction test of the theory.
                    "delta_logit_abs": float(delta_logit_u_abs[i, j]),
                    "predicted_sign_logit_abs": int(predicted_sign_logit_abs_u[i, j]),
                    "actual_sign_logit_abs": int(actual_sign_logit_abs_u[i, j]),
                    "delta_log_Z": float(delta_log_Z[i, j]),
                    "tau": float(tau_per_pos[i, j]),
                    "A_value": float(A_per_pos[i, j]),
                    "norm_sq_S": float(norm_sq[i, j]),
                    "entropy_S": float(entropy_b[i, j]),
                    "target_j_for_seq": target_j_for_i,
                    "is_target": is_target_pos,
                }
            )

        df_out = pd.DataFrame(rows)
        if len(df_out):
            mean_abs_dp = float(np.mean(np.abs(df_out["delta_pi_actual"].to_numpy())))
            logger.info(
                "[update_consistency] step=%d  rows=%d  mean|Δπ|=%.2e",
                step, len(df_out), mean_abs_dp,
            )

        if self.measurements_path.exists():
            existing = pd.read_parquet(self.measurements_path)
            combined = pd.concat([existing, df_out], ignore_index=True)
        else:
            combined = df_out
        combined.to_parquet(self.measurements_path, index=False)
