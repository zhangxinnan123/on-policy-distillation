# Update-Consistency Experiment (in-flight)

Measure per-token probability changes Δπ during real Verl on-policy
distillation steps, and compare against the sign + relative-magnitude
predictions of the reverse-KL theory.

## How it works

**Training loss is `k1_topk_overlap` + `use_policy_gradient=True`.** This is
exactly the REINFORCE single-sample reverse-KL update the theory predicts:

  - `compute_k1_topk_overlap` (`verl/trainer/distillation/losses.py:1172`) returns
    the per-token k1 KL at the sampled token.
  - The PG branch in `distillation_loss` turns `-distillation_losses` into the
    advantage signal, yielding `-A · ∇log π_θ(u)` (with PPO ratio ≈ 1 inside
    one optimizer step).
  - Because the loss is registered with `use_topk=True`, the teacher dispatch
    fetches teacher's top-K logprobs into `data["teacher_logprobs"]` /
    `data["teacher_ids"]`. Its logits-processor hook also emits
    `student_topk_probs` (π_θ at teacher's top-K) and `student_s2 = ||π_θ||²`
    into model_output.

The callback runs around `_update_actor`:

  1. **Before**: call `actor_rollout_wg.compute_log_prob(current_batch, …)`
     with `distillation_use_topk=True, compute_loss=True`. The actor's
     installed loss_fn (the training loss) fires both as logits processor
     (extracting student_topk_probs + student_s2) and as the final loss
     (which we discard). Returns model_output with:
       - `log_probs`               (B, R)  — log π_θ at sampled token
       - `entropy`                 (B, R)  — full-vocab entropy
       - `student_topk_probs`      (B, R, K) — π_θ at teacher top-K positions
       - `student_s2`              (B, R)  — ||π_θ||²
  2. `super()._update_actor(batch)` — the optimizer step
  3. **After**: same `compute_log_prob` call. Now with post-update weights.
  4. For each valid response position `(i, j)` and each candidate token
     `c ∈ teacher_top_K ∪ {sampled_u}` (if u happens to also lie in
     teacher's top-K we keep only the sampled_u row, not both):
       - π_S^before(c), π_S^after(c) come from the tensors above
       - Δπ_actual(c) = π_S^after(c) − π_S^before(c)
       - Predicted sign:
         - c ≠ u: `sign(A) · sign(τ − π_θ(c))`
         - c = u: `sign(A)` (the factor `1 + τ − π_u = (1−π_u)² + Σ_{v≠u}π_v²`
           is ≥ 0, so there is no watershed at the sampled token)
       - Predicted relative magnitude:
         - c ≠ u: `π_θ(c) · |τ − π_θ(c)|`
         - c = u: `π_u · (1 + τ − π_u) = π_u · (1 − 2π_u + ‖π‖²)`
     Append per-row to `measurements.parquet`.

**No frozen probe set, no loss swap, no extra teacher dispatch.** Per
measurement step the cost is just 2 extra `compute_log_prob` calls on the
training batch (no backward).

## Important assumption

Each trainer iteration must be exactly one optimizer step:

    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size = <train batch size>

The config pins `ppo_epochs=1`; size the mini-batch to match your training
batch yourself. If `_update_actor` performs multiple optimizer steps under
the hood, the before/after diff would span multiple steps and the figure no
longer corresponds to the theory.

## Candidate set caveat

The measurement candidate set per response position is
`teacher_top_K ∪ {sampled_u}` (K+1 ≈ 101 by default; K when u ∈ top_K
since we de-dup).

What this means for the analysis:
- `pi_S_before` / `pi_S_after` for non-sampled c are student's probabilities at
  *teacher's* top-K positions (computed by `compute_student_topk_overlap_k1`
  gathering student logits at `teacher_topk_ids`).
- `pi_T(c)` is teacher's own top-K logprob output. So `pi_T` and `pi_S` in the
  same row are at the same token id — `π_T(c) − π_S(c)` is well-defined.
- The candidate set is **biased toward teacher's preferences**: tokens
  teacher considers unlikely are excluded. A `sign(π_T − π_S)` baseline
  computed here over-samples positions where teacher has high mass.

## Files

    callback.py    — UpdateConsistencyCallback (before/after compute_log_prob + write)
    trainer.py     — ConsistencyRayPPOTrainer (subclass with hooks)
    main.py        — Hydra entry (always train_and_measure mode)
    analyze.py     — offline 2×3 figure + per-stratum CSV
    config/measure_during_opd.yaml
    run.sh

## Workflow

    bash recipe/update_consistency/run.sh

By hand:

    python -m recipe.update_consistency.main \
        --config-name=measure_during_opd \
        update_consistency.output_dir=data/measurements \
        update_consistency.measure_every=1 \
        trainer.total_training_steps=10 \
        actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
        # ... your usual data / teacher overrides ...

    python -m recipe.update_consistency.analyze \
        --measurements data/measurements/measurements.parquet \
        --output-dir figures/update_consistency

## Output: measurements parquet schema

One row per `(step, sequence_id, response_pos, candidate_token_id)`:

    step, sequence_id, response_pos, candidate_token_id
    is_sampled_u, is_above_watershed
    pi_S_before, pi_S_after, pi_T, delta_pi_actual
    predicted_sign, actual_sign, predicted_relative_magnitude
    tau, A_value, norm_sq_S, entropy_S

`sequence_id` is the row index within the rollout batch at that step.
`pi_T` is exact for both teacher-topK candidates and the sampled-u candidate.
Up to `K+1` rows per (response position, step).

## Sanity checks

- `train_and_measure`: per measurement event, mean `|Δπ|` is logged
- `analyze.py`: per-stratum sign agreement + Pearson/Spearman magnitude
  correlation, top-10 sign-disagreement outliers, the 2×3 figure
