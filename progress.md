# SDPO Progress Log

Research log for the SDPO fork of verl. Newest entry on top. Each entry captures what was run, what was observed, and what to try next.

Entry template:

```
## YYYY-MM-DD — <short title>

**Ran:**
- <experiment name / script / config diff> (job id, wandb run)

**Observed:**
- <key metrics, deltas vs. baseline, plots to look at>

**Interpretation:**
- <what it means, what it rules in/out>

**Next:**
- <concrete follow-up experiments or code changes>
```

---

## 2026-07-20 — IS-weight formulation ablation: shaping vs min-clip, entropy sanity

**Ran:**
- **Trainer changes** (`verl/trainer/ppo/sdpo_ray_trainer.py`)
  - Fixed `_build_is_rollout_weights` w-definition: reverted from LUFFY's
    `w = π_θ(y|x)` (numerator only) back to true IS ratio
    `w = π_θ(y|x) / π_θ_start(y|x, e)`. LUFFY paper drops the denominator
    because it doesn't know the behavior policy's logprob; we do (we sampled
    the hint rollouts ourselves), so no reason to throw the info away.
  - Added `mode` parameter to `_build_is_rollout_weights`:
    - `mode="shaping"` (default): `f(w) = w/(w+c)` — LUFFY-style smooth bound
    - `mode="clip"`: `g(w) = clip(w, low, high)` — hard PPO-style clip, no shaping
  - New config knobs `+sdpo.sdpo_grpo.is_correction.{mode,clip_low,clip_high}`.
    DAPO defaults `[0.8, 1.28]` (`ε_low=0.2, ε_high=0.28`).
  - Wandb metrics on hint tokens: `sdpo/is/mean_w`, `mean_shaped`, `min_w`, `max_w`,
    `clip_low_frac`, `clip_high_frac`.

- **Scripts** (`sdpo_experiment/hint_llm/`)
  - `run_qwen_isclip_hint_level3.sh` — hard `clip(w, 0.8, 1.28)` explicit mode,
    no shaping, no sign-flip. Simple `-A·clip(w)` (single term, not min-clip).
  - `run_qwen_ppo_isclip_hint_level3.sh` — **standard PPO min-clip** form:
    `pg_loss = max(-A·w, -A·clip(w, 0.8, 1.28))`. Uses trainer's implicit-ratio
    path (`fn_const=0` + `use_rollout_log_probs=True`) so PPO ratio auto-becomes
    the IS ratio and `compute_policy_loss_vanilla`'s standard clip triggers.
    DAPO asymmetric bounds (`clip_ratio_low=0.2, clip_ratio_high=0.28`).
  - All new EXP_NAMEs prefixed with mechanism: `luffyShape_`, `hardClip_`,
    `ppoMinClip_`, `signFlip_..._luffyShape_` (was: unclear generic names).

- **Jobs**
  - 9052 (killed) — Combo `signFlip_lpos0.1 + luffyShape_c0.1 + regCscale0.25`.
    Cancelled after 16h+ (wandb name too generic, redoing with clearer tag).
  - 9080 (running) — `luffyShape_c0.1_regCscale0.25`, no sign-flip.
    wandb: `qwen_deepmath_luffy_c0.1_n1_hintlevel_3_6_4` (old EXP_NAME).
  - 9096 (killed) — Explicit `hardClip_lo0.8_hi1.28` (`-A·clip(w)` form).
    Cancelled after 6m; not the intended min-clip form.
  - 9100 (running) — `ppoMinClip_epsLo0.2_epsHi0.28_regCscale0.25`, no sign-flip.
    Clean A/B partner for 9080 (only differs in w handling).

**Observed:**
- **LUFFY entropy issue confirmed**: with `luffyShape_c=0.1` on old jobs
  (9035 killed, 9051 killed, and current 9080 early steps), `actor/entropy`
  drops faster than baseline. Root cause is `f'(p) ≈ c/(p+c)²` amplifying
  gradient on rare (hint) tokens → hint-generated rare tokens' prob rises fast
  → distribution sharpens.
- **SDPO sign-flip alone (no LUFFY) — entropy INCREASES** with `sign_flip_λ=0.1`:
  the reweighting `(1-λ) + λ·exp(sign(A)·delta)` boosts learning on tokens where
  hint disagrees with unhinted policy, which spreads probability mass. Opposite
  of LUFFY's effect. Baseline 9030 (sign-flip only, completed at 200 steps)
  showed this trend.
- Regime C α=0.25 dampens the amplitude of Regime C updates 4×; without it the
  small-group std inflation compounded LUFFY's rare-token amplification.

**Interpretation:**
- **Sign-flip and LUFFY-shape push entropy in opposite directions** — this is
  actually useful for the combo: they could balance each other. But the
  Regime B (sign-flip) and Regime C (LUFFY) affect disjoint token populations,
  so the "balance" is mostly by coincidence not by design.
- Hard clip (9100 PPO min-clip) should behave differently from shaping:
  - LUFFY shaping: gradient reshape everywhere (soft)
  - PPO min-clip: pass through for `w ∈ [0.8, 1.28]`, hard bound outside
  - Expect PPO min-clip to be **less aggressive** on rare tokens (most hint
    tokens have `w << 0.8` → clip to 0.8, everyone gets same weight)
- Under DAPO `[0.8, 1.28]` clip, if IS ratio distribution is heavily left-skewed
  (typical for hint rollouts), most tokens hit `clip_low_frac ≈ 1`. Effectively
  hint rows just get a 0.8× loss scale, not real IS correction. Wandb
  `sdpo/is/clip_low_frac` on 9100 will verify.

**Next:**
- Wait 30-60 min for 9100 to hit step 3-5. Check `sdpo/is/clip_low_frac`.
  If > 0.9 → clip range too tight, expand to `[0.1, 10]` for a variant run.
- Compare 9080 vs 9100 at step 40-80: reward/mean, actor/entropy, val pass rate.
  Both share the α=0.25 fix so entropy shouldn't collapse; differences will
  come from smooth vs hard IS treatment.
- If 9080 (LUFFY) still shows entropy collapse with α=0.25: raise c to 0.3+ or
  add small `entropy_coeff` regularizer. If 9100 (min-clip) shows collapsed
  IS at low bound: widen clip.

### Job 9100 detailed algorithm (`ppoMinClip_epsLo0.2_epsHi0.28_regCscale0.25`)

**Config summary:**
- `algorithm.adv_estimator=sdpo_grpo`
- `algorithm.use_kl_in_reward=False`, `actor.use_kl_loss=False`, `entropy_coeff=0`
- `actor.clip_ratio_low=0.2, clip_ratio_high=0.28` → PPO ratio clip [0.8, 1.28] (DAPO asymmetric)
- `actor.use_rollout_log_probs=True` — KEY: don't detach old_log_probs
- `sdpo.hint_rollout.merge_into_group=True`, `n_hint=1`
- `sdpo.sdpo_grpo.sign_flip_lambda_{pos,neg}=0.0` — sign-flip OFF
- `sdpo.sdpo_grpo.regime_c_scale=0.25` — Regime C α dampener
- `sdpo.sdpo_grpo.is_correction.enabled=True, fn_const=0.0` — implicit-ratio mode

**Batch:** 128 prompts × (8 main + 1 hint) = 1152 rows/step.

**Per-token loss on hint tokens (Regime C only, elsewhere A_hint=0):**
```
w_t = π_θ(y_t|x) / π_θ_start(y_t|x, e)               (true IS ratio)
A_hint = regime_c_scale × GRPO_normalize({8 main + 1 hint})   (α=0.25 both signs)
loss_t = max(-A_hint · w_t, -A_hint · clip(w_t, 0.8, 1.28))    (PPO min-clip)
```

**Per-token loss on main tokens:**
```
ratio_t ≈ 1 (on-policy, ppo_epochs=1)
loss_t = -A_main_t              (PPO clip no-op)
```

**Key mechanism (implicit-ratio path in trainer):**
- Stage: `_swap_hint_rows_to_unhinted` before `_update_actor`.
- Swap replaces `batch["prompts"][hint_rows]` from `x+e` to `x` but leaves
  `batch["old_log_probs"][hint_rows] = log π_θ_start(y_hint|x, e)` unchanged.
- Actor forward on the swapped batch: `log_prob[hint] = log π_θ(y_hint|x)`.
- PPO ratio becomes `exp(log π_θ(y|x) − log π_θ_start(y|x,e)) = w`, and
  `compute_policy_loss_vanilla`'s standard clip auto-fires on w.
- No `rollout_is_weights` tensor written; IS correction rides on PPO ratio.

**Regime C scaling** (line 210-212 of `sdpo_ray_trainer.py`):
- `advantages[main_rows + hint_rows] *= regime_c_scale` — applied to BOTH signs
  symmetrically (main rows have A<0, hint rows have A>0; all shrunk to 25%).
- Preserves the GRPO zero-sum property within the group.
- Net effect: full Regime C update magnitude is 25% of a normal Regime A/B step.

**Correctness confidence:**
- ✅ Old logprob is hinted; log_prob (after swap) is unhinted → PPO ratio = true IS ratio.
- ✅ `use_rollout_log_probs=True` prevents the detach that would collapse ratio to 1.
- ✅ `ppo_epochs=1` (verl default) → old ≈ current, main ratio ≈ 1, only hint rows non-trivial.
- ⚠️ Most hint tokens likely have `w << 0.8` → clip pins them to 0.8 → hint pg_loss
  ≈ constant 0.8 × A scale, not true fine-grained IS. Wandb `actor/pg_clipfrac` should
  approach 1 for hint tokens; no explicit `sdpo/is/*` metrics in implicit path.
- ⚠️ Combined with α=0.25 in Regime C, effective hint-row loss magnitude is
  `≈ 0.25 × 0.8 × A_grpo = 0.20 × A_grpo` (20% of a normal step).

**Contrast with 9080 (`luffyShape_c0.1_regCscale0.25`):**
- 9080 uses explicit `rollout_is_weights = f(w) = w/(w+0.1)` (smooth, no PPO clip on w).
- 9100 uses PPO ratio auto-clip on w (min-clip form, hard bounds).
- Same α=0.25, same batch structure, same regimes — only the w-handling differs.

### Amplify + fixed Regime C (job 9461, running)

**Motivation:**
- All prior hint-augmented experiments underperformed pure GRPO baseline.
- Two prior failure modes:
  - Sign-flip: entropy up over-shoot
  - LUFFY shaping / off-policy hint: entropy down collapse
- 9279/9306 (decay): partially controlled but still < pure GRPO
- Key hypothesis: the GRPO normalization in Regime C creates
  spurious negative advantage on failed main rows (~-0.09 with α=0.25).
  Since "all main fail" means "student doesn't know how", punishing those
  rollouts has no meaningful signal.

**Trainer changes:**
- **Regime C now uses fixed advantage instead of GRPO normalization:**
  - `A_hint = regime_c_scale × 1.0` (positive learning signal only)
  - `A_main = 0` (no gradient, no punishment)
  - GRPO_normalize replaced with direct assignment in `sdpo_ray_trainer.py:205-218`
  - **BREAKING CHANGE**: this affects ALL merge-based scripts (luffy, isclip,
    ppo_isclip, combo, sdpo_decay). Old scripts, if rerun, will now behave
    differently. Tagged via EXP_NAME `fixedRegC` suffix on new scripts to
    disambiguate wandb runs.

- **New `_build_is_rollout_weights` mode="amplify"**: LUFFY gradient modifier
  `4γ²/(p+γ)²` applied to positive-advantage hint tokens only (via
  `apply_mask = is_hint & (adv > 0)`).
  - `p = π_θ(y|x)` = raw deployment prob per hint token
  - `amplifier > 1` for p < γ (boost rare hint tokens)
  - `amplifier = 1` at p = γ (inflection)
  - `amplifier → 0` for p >> γ (dampen common tokens)
  - Multiplies onto PPO min-clipped surrogate (via `rollout_is_weights`), so
    PPO clip's gradient stop remains intact. This is different from LUFFY
    paper's shaping (`f(w) = w/(w+c)`) which only rescales magnitude and doesn't
    trigger clip semantics.

- **`_sdpo_is_correction_params()` extended** to return `(mode, clip_low,
  clip_high, gamma)`. Config: `+sdpo.sdpo_grpo.is_correction.mode=amplify`,
  `+sdpo.sdpo_grpo.is_correction.gamma=0.1`.

**Correctness note on PPO clip semantics for hint IS:**
- `fn_const=0` implicit path: standard PPO min-clip — `torch.clamp` gives zero
  gradient outside `[0.8, 1.28]`, so hint tokens with extreme IS ratios truly
  stop updating. Correct PPO semantics.
- `fn_const>0` explicit LUFFY path: `rollout_is_weights = f(w) = w/(w+c)` is
  a detached scalar multiplier onto pg_losses — only rescales magnitude, doesn't
  trigger gradient stop. Weaker guarantee than PPO clip and partially explains
  why 9080/9187 had unbounded entropy drift.
- `amplify` mode: gradient modifier `4γ²/(p+γ)²` is also detached, but it
  multiplies onto the ALREADY-clipped PPO surrogate (from `compute_policy_loss_vanilla`
  line 1354 → 1358). So clip stops still hold for extreme IS ratios; amplify
  only reshapes magnitude on tokens that survived the clip.

**Config summary (job 9461):**
- Merge = Regime C only (no always_merge)
- Regime C: fixed A_hint=+1.0, A_main=0 (NO GRPO normalization, NO punishment)
- Amplify: mode=amplify, γ=0.1, applied to hint & adv>0 tokens
- PPO min-clip [0.8, 1.28] on IS ratio (via use_rollout_log_probs=True)
- Sign-flip OFF (λ=0)
- regime_c_scale = 1.0 (no additional dampening needed)
- save_freq=200 (save only final checkpoint)

**EXP_NAME:** `qwen_deepmath_amplify_gamma0.1_ppoMinClip_epsLo0.2_epsHi0.28_fixedRegC1.0_n1_hintlevel_3_6_4`

**Expected hint gradient magnitude** (Regime C only):
| p (raw hint prob) | A_hint | amplifier | effective A |
|---|---|---|---|
| 0.001 | +1.0 | 3.31 | **3.31** |
| 0.01  | +1.0 | 3.31 (clamped by exp(-10)) | 3.31 |
| 0.1 (inflection γ) | +1.0 | 1.0 | 1.0 |
| 0.5   | +1.0 | 0.11 | 0.11 |
| 1.0   | +1.0 | 0.03 | 0.03 |

**Status:** submitted 2026-07-22 as job 9461, running on ip-10-1-39-123.

### hint_reg (job 9564, running): true gradient-through-shape LUFFY loss

- First implementation with gradient flowing through the shape (via
  dp_actor's fresh log_prob), unlike shaping/amplify which were detached.
- Formula: `loss = -ratio/(ratio + γ)` on Regime C hint tokens only.
  A_hint=0 in pg_loss → hint gradient comes ONLY from hint_reg (fully isolated).
- **Observed**: entropy still increases, but slower than amplify variants
  (9461, 9505). Training score not as good as pure GRPO baseline so far.
- Verdict: shape-gradient LUFFY doesn't fix the fundamental problem. The
  saturating shape `ratio/(ratio+γ)` has derivative `γ/(ratio+γ)²` which
  peaks at rare tokens — same "boost rare, suppress common" mechanism that
  drives entropy up in amplify. Direction of gradient flow through shape
  didn't invert the pathology.

### Correctness note: shaping mode's gradient behavior (semantic misalignment with LUFFY paper)

- Existing `mode="shaping"` computes `weight = w/(w+c)` where `w = exp(unhinted - hinted)`.
- **`_build_is_rollout_weights` itself has NO explicit `.detach()` call.** The
  function receives `unhinted_logprob` and `hinted_logprob` from the trainer
  dispatcher, which got them via `_compute_old_log_prob → actor_rollout_wg.compute_log_prob`
  (Ray cross-process call).
- **The autograd graph is broken at the source**: `dp_actor.compute_log_prob`
  wraps its forward pass in `with torch.no_grad():` (line 482 of dp_actor.py).
  So log_probs returned to the trainer are no-grad leaf tensors from birth.
- Downstream `torch.exp(unhinted - hinted)` and `w/(w+c)` produce no-grad
  tensors because inputs have `requires_grad=False`. `rollout_is_weights` stored
  in the batch is thus a de-facto detached scalar.
- **Consequence**: at loss time in `compute_policy_loss_vanilla`,
  `pg_loss = -A · ratio_ppo · rollout_is_weights`, the gradient flows ONLY
  through `ratio_ppo` (which has fresh gradient from dp_actor's own forward).
  The shape `f(w)` acts as a detached per-token magnitude multiplier.
- **This differs from LUFFY paper**, which lets gradient flow through the
  shape: `∂[p/(p+γ)]/∂θ = γ/(p+γ)² · ∂p/∂θ`. In LUFFY paper's setup, rare
  tokens (small p) get amplified gradient via the shape's own derivative.
- In our `mode="shaping"`, the shape can only rescale magnitude, not amplify
  gradient. Rare tokens (small w) get COMPRESSED gradient (`w/(w+c) ≈ w/c`),
  the opposite of LUFFY's intent.

**Implication**: the previous shaping/luffy-family experiments (9080, 9052,
9187 etc.) that "underperformed vs GRPO" may have failed partly because the
loss form was structurally different from LUFFY paper — shape acted as a
detached weight, not as a gradient-flowing objective. The new `hint_reg`
(job 9564) is the first implementation where gradient truly flows through
the shape via dp_actor's fresh forward log_prob.

### Ablation: amplify vs amplify_linear (jobs 9461, 9505) — both killed

- 9461: `4γ²/(p+γ)²` (square). 9505: `2γ/(p+γ)` (no square). γ=0.1, fixed
  Regime C, no sign-flip, PPO min-clip.
- **Both cause LARGE entropy INCREASE, uncontrolled**. Both killed early
  after entropy trajectory made clear divergence from healthy training.
- 9505 amp_mean > 9461 (surprising), suggests hint tokens are NOT dominantly
  in the rare-p region — amplify's peak boost is rarely triggered.
- Suppressing common tokens (amp < 1) flattens the distribution rather than
  sharpening toward y_hint → entropy ↑, contrary to LUFFY-shape's ↓ pattern.
- Verdict: amplify family abandoned. Consistent with the running theme —
  no hint-augmented method has beaten vanilla GRPO on this student/dataset.


---

### SDPO decay experiments: 9279 (slow) vs 9306 (fast) — both COMPLETED 200/200

**Setup:** Both use identical config except decay speed.
- Merge = Regime C only (no always_merge)
- sign-flip base λ = 0.1, cosine decay to λ = 0.001 (final_scale = 0.01)
- PPO min-clip on hint IS ratio [0.8, 1.28] via implicit path
  (`use_rollout_log_probs=True`, `fn_const=0`)
- regime_c_scale = 0.25

| Job | Decay total_steps | λ at step 100 | λ at step 200 | EXP_NAME suffix |
|-----|-------------------|---------------|---------------|-----------------|
| 9279 (slow) | 200 | ~0.051 | ~0.001 | `_lpos0.1to0.01_` |
| 9306 (fast) | 100 | ~0.001 | ~0.001 | `_decayTo100_` |

**Observed (wandb):**
- **9279 (slow decay): entropy rises throughout training, performance
  eventually degrades**. Sign-flip pressure persists at meaningful strength
  (λ ~0.05 at step 100) too long → cumulative entropy inflation dominates.
- **9306 (fast decay): entropy rises briefly early (steps 0-100 while λ still
  meaningful), then FALLS after step 100 because Regime C off-policy hint loss
  dominates once sign-flip is essentially off (λ<0.001)**. Net result: better
  training curve.

**Mechanism explanation (matches earlier prediction):**
- **Sign-flip is entropy-UP**: per-token scaling that systematically boosts
  hint-specific rare tokens' probability across many main rollouts → spreads
  probability mass → entropy ↑.
- **Off-policy hint (Regime C merge) is entropy-DOWN**: MLE-style pull toward
  one specific y_hint sequence → concentrates probability mass → entropy ↓.
- These two mechanisms operate on **disjoint token populations** (main rows vs
  hint rows in Regime C), so they don't cancel per-token — they compose as a
  net effect on the whole policy.

**Verdict on decay schedule:**
- **Fast decay (100 steps) works better** — matches sign-flip's early bump
  with a clean cutoff, then hint's entropy-down effect stabilizes late training.
- Slow decay (200 steps) leaves sign-flip alive during the entropy-up phase
  for too long → net entropy trends upward → over-shoots exploration budget.

**HOWEVER — critical negative result:**
- **Both 9279 and 9306 underperform pure on-policy GRPO (no hint at all)**.
- Best hint-augmented run so far does NOT beat vanilla GRPO baseline on this
  setup (Qwen3-1.7B, deepmath_diff6to8, 200 steps).
- Implication: on this student/dataset combination, hint injection — in any of
  the forms we've tried — is net-negative:
  - Sign-flip pushes entropy up destructively (9187).
  - LUFFY-shape / off-policy hint pushes entropy down destructively (9080, 9232).
  - Combinations with decay (9279, 9306) can balance the two directions but the
    net signal is still worse than pure on-policy.
- Suggested reasons:
  - **Signal too weak**: at 1.7B / diff6-8, Regime C is rare (student rarely
    stuck AND hint successful). Most steps hint contributes nothing.
  - **Signal too destructive when it fires**: the rare-but-strong hint gradient
    creates instabilities that pure GRPO's on-policy multi-sample diversity
    doesn't produce.
  - **Distribution mismatch**: hint rollouts are sampled from π(·|x, e) but
    trained against π(·|x); even with IS correction, the gap is large enough
    that gradient variance / bias undermines training.

**Suggested next directions (if continuing hint work):**
- **Scale up student model** (Qwen3-4B or 7B). At 1.7B, base pass rate on
  diff6-8 is near 0, and hint doesn't rescue it meaningfully.
- **Easier dataset** (MATH, GSM8K). If hint helps at all it should show up on
  problems where student is on the edge of solving.
- **Warmup with pure GRPO, then introduce hint late** — reverse of our current
  approach. Only add hint after student has some baseline competency.
- **Consider abandoning hint approach** for this student/dataset and focus on
  pure GRPO variants (better data curriculum, longer training, etc.).

**Correctness note on IS path (verified this turn):**
- The `fn_const=0` implicit path IS standard PPO min-clip on IS ratio:
  `pg_loss = max(-A·ratio, -A·clamp(ratio, 0.8, 1.28))`, `torch.clamp` gives
  zero gradient outside bounds → hint tokens with extreme IS ratio truly stop
  updating (not just magnitude-scaled).
- The `fn_const>0` explicit path (LUFFY shaping) is different: it multiplies
  `rollout_is_weights = f(w) = w/(w+c)` into pg_loss as a detached constant,
  which only rescales magnitude — log_prob's gradient always flows. This is
  a weaker guarantee than PPO clip and partially explains why earlier LUFFY
  runs (9080, 9187) had unbounded entropy drift.

---

### always_merge_hint_into_group failure (job 9232 killed)

- **9232** (`ppoMinClip_alwaysMerge_epsLo0.2_epsHi0.28_lpos0.0_regCscale0.25`)
  — new trainer knob `always_merge_hint_into_group=True` puts hint rows into
  the GRPO group in every regime (except Regime A, which keeps hint out to
  avoid failed hint polluting group stats).
- Setup was IS-correct: hint rows are swapped to unhinted prompts and PPO
  ratio auto-becomes `w = π_θ(y|x)/π_θ_start(y|x, e)` with DAPO min-clip
  [0.8, 1.28]. No sign-flip. Regime C α=0.25.
- Result: **reward rose briefly, then entropy dropped fast and performance
  degraded**. Killed early.
- Root cause:
  - Regime B group of 3 main-succ + 5 main-fail + 1 hint-succ (9 rows).
    Hint row gets `A ≈ +1.12`, same magnitude as a main-succ row.
  - Hint response `y_hint` is ONE specific sequence. All hint tokens
    push student toward that one path.
  - Main-succ rows come from N independent samplings → gradient spreads over
    diverse successful paths.
  - Net effect: hint gradient is *entropy-reducing*, main gradient is
    *entropy-preserving*. In Regime B (frequent regime) this pull toward a
    single specific sequence dominates → distribution sharpens too fast.
  - IS ratio `w` clip does NOT rescue: `w` is usually small (student without
    hint rarely generates y_hint), so `pg_loss = -A·w` is naturally small,
    but the direction is consistent every step → cumulative sharpening.
- Contrast with vanilla merge (only Regime C uses hint):
  - Regime C is rare (student mostly not stuck fully) → hint gradient sparse
    → no cumulative sharpening from hint.
  - Regime C's own α=0.25 further dampens.

**Updated verdict on merging hint into loss:**
- Only-Regime-C rescue is safer: hint contributes gradient exactly when
  student needs it, otherwise silent.
- always_merge amplifies hint's "single-path pull" across most steps →
  entropy collapse.
- **Trainer changes staying (bug-free), but `always_merge_hint_into_group`
  knob defaults to False and is not recommended.**

### Combo failure: sign-flip + PPO min-clip IS (job 9187 killed)

- **9187** (`signFlip_lpos0.1_ppoMinClip_epsLo0.2_epsHi0.28_regCscale0.25`)
  — combined the two mechanisms in one run.
- Result: **initial steps looked GOOD (both reward and entropy healthy), then
  entropy kept climbing, and performance dropped**. Killed after the
  regression became clear.
- Trajectory shape (from wandb):
  - Early steps: reward/mean ↑, entropy stable → looked like the combo was
    working better than baseline
  - Mid-training: entropy started monotonically increasing
  - Later: performance rolled over and dropped; the training had over-diversified
- The initial improvement suggests hint signal IS useful at first — the failure
  mode is that sign-flip's continuous entropy pressure eventually **over-shoots**,
  spreading probability mass beyond what the underlying reward signal can pull
  back. Once entropy grows too far, the policy loses the sharp modes it had
  learned and reward collapses.
- Root cause (analytical): sign-flip on Regime B main rows and PPO min-clip on
  Regime C hint rows push in incompatible directions:
  - Sign-flip reweight `(1-λ) + λ·exp(sign(A)·delta)` in Regime B tends to
    **spread probability mass** across tokens where hint disagrees with the
    unhinted policy → entropy ↑.
  - PPO min-clip on hint IS ratio in Regime C leaves the hint gradient at the
    clipped 0.8× scale on rare tokens. Not enough to counter Regime B's
    entropy pressure at 1.7B scale.
- Also on `diff6to8` the base pass rate is very low → most steps are Regime A
  (hint fails) or Regime C (student fails). Regime B is the shrinking middle,
  and sign-flip there disperses probability without buying accuracy.

**Cross-experiment entropy direction summary:**

| Job | Mechanism | Entropy | Performance |
|-----|-----------|---------|-------------|
| 9030 (completed) | Sign-flip only, no merge | ↑ | ≈ baseline |
| 9080 (killed) | LUFFY-shape only, α=0.25 | ↓ fast | poor |
| 9129 (killed) | PPO min-clip only, α=1.0 | ↓ fast | poor |
| 9100 (OOM at 11h) | PPO min-clip only, α=0.25 | stable | inconclusive |
| 9187 (killed) | Sign-flip + PPO min-clip, α=0.25 | ↑ | worse |

**Emerging pattern:**
- On diff6-8 with Qwen3-1.7B, both entropy-up (sign-flip) and entropy-down
  (LUFFY, un-damped PPO min-clip) directions harm training.
- The one stable-entropy candidate (9100) OOM'd before conclusive metrics.
- Combining sign-flip and hint IS is worse than either alone → they don't
  compose; sign-flip's Regime B action is destructive at this scale/difficulty.
- Next candidate: rerun 9100-style config (PPO min-clip α=0.25, no sign-flip)
  with reduced token budget (45056) to avoid OOM, and let it run 200 steps.

### α ablation: `regime_c_scale=1.0` vs `0.25` (job 9129 killed)

- **9129** (`ppoMinClip ... regCscale1.0`) — same as 9100 but α=1.0 (no dampening).
- Result: **entropy dropped very fast, killed early**. Confirms Regime C's raw
  |A| is too large without α.
- Root cause (analytical): in Regime C the GRPO group is 8 fails + 1 success.
  With `norm_by_std=True`:
  ```
  mean = 1/9 ≈ 0.111
  var  = (8·0.111² + 0.889²) / 9 ≈ 0.099
  std  ≈ 0.314
  A_hint = (1 - 0.111) / 0.314 ≈ +2.83
  A_main = (0 - 0.111) / 0.314 ≈ -0.354
  ```
  |A_hint| ≈ 2.83 is much larger than typical Regime A/B |A| ~1 (where std is
  bigger because success/fail is more balanced). At α=1, this ~3× oversized
  gradient signal (compounded by log-space instability of exp) drives entropy
  down in a few steps.
- α=0.25 brings the effective update magnitude in line with Regime A/B; entropy
  behavior on 9100 is stable so far.
- **Takeaway**: Regime C α damping is not optional when running with hint merge.
  Alternative would be to disable std-normalization for Regime C (leave raw
  centered scores) or use a fixed-clip advantage bound.

---

## 2026-07-19 — LUFFY reshape wired, Regime C α scaling, first ablation runs

**Ran:**
- **Data pipeline**
  - Downloaded `XinnanZhang/deepmath-diff6to8-verified` (56,209 rows) → `train.parquet` on cluster (274 MB, `data_source=math_dapo`, ~46% num / 26% Yes-No / 26% other GT format).
  - Extracted 3-level hints (level_1/2/3) for each row via Bedrock Claude Haiku 4.5, prompt in `sdpo_experiment/data_process/hint_generation.jinja` (rewrote from original to "direct hint" style — dropped "don't quote/paraphrase" clauses, added word budgets, real jinja slots for `{{ question }}` / `{{ solution }}`).
  - Extractor: `sdpo_inference/extract_hints_bedrock.py`, ~$180 total, ~1h52m at concurrency=32. 56,209 / 56,209 cells populated, 0 real errors, 0 boxed-answer leaks.
  - Merged into training parquet via `sdpo_inference/merge_hints_into_train.py` → `train_with_hints.parquet` (135 MB, adds `level_1/2/3` string columns).

- **Trainer changes** (`verl/trainer/ppo/sdpo_ray_trainer.py`)
  - Hint prompt template `EXPERT_GUIDANCE_TEMPLATE` rewritten from "expert trajectory" (for full solutions) to short-hint direct style: `"Hint (a suggested approach — use it to guide your reasoning, but derive every step yourself): {expert}"`.
  - `_build_is_rollout_weights` rewritten to LUFFY semantics: `w = π_θ(y|x)` (raw prob, not IS ratio), `f(p) = p/(p+c)`. Old ratio-based `w` was numerically confusing and inconsistent with LUFFY paper. Config knob unchanged (`+sdpo.sdpo_grpo.is_correction.fn_const=c`), but effect now matches LUFFY `p_div_p_c` mode.
  - Added `regime_c_scale` param to `compute_sdpo_grpo_advantage`. Scales Regime C advantage by α (default 1.0). Config: `+sdpo.sdpo_grpo.regime_c_scale=0.25`. Motivation: Regime C group has only 1 succ / 8 fail → std small → |advantage| inflated → entropy collapses.

- **Ops fixes** (blocked earlier runs)
  - Installed `math_verify` in cluster `opd` env (was missing → reward always returned 0 → all-zero metrics on 9018).
  - Installed `boto3` / `jinja2` in `opd` for hint extraction.
  - Removed `CUDA_VISIBLE_DEVICES` export from sbatch (conflicted with Slurm's `ROCR_VISIBLE_DEVICES`); `unset ROCR_VISIBLE_DEVICES` instead.
  - Synced `verl/trainer/config/data/legacy_data.yaml` (was missing on cluster, blocked 9011).

- **Scripts created** under `sdpo_experiment/hint_llm/`:
  - `run_qwen_sdpo_grpo_signflip_hint_level3.sh` — Sign-flip only (no merge, no LUFFY)
  - `run_qwen_luffy_hint_level3.sh` — LUFFY only (merge + `is_correction.fn_const=0.1`, λ=0)
  - `run_qwen_combo_hint_level3.sh` — Sign-flip + LUFFY (both on)
  - Corresponding sbatch wrappers (job_name=seller)

- **Jobs**
  - 9030 (running) — sign-flip only, λ=0.1, no merge. wandb: `qwen_deepmath_sdpo_signflip_n1_hintlevel_3_lpos0.1_6_4`
  - 9035 (killed) — LUFFY only c=0.1, α=1.0. Entropy collapsed too quickly.
  - 9038 (running) — Combo λ=0.1 + LUFFY c=0.1, α=1.0. wandb: `qwen_deepmath_combo_lpos0.1_c0.1_n1_hintlevel_3_6_4`
  - 9051 (running, new) — LUFFY only c=0.1, **α=0.25**. Testing if α fixes 9035's entropy collapse.

**Observed:**
- On 9035 (LUFFY-only, no α): entropy dropped much faster than baseline; performance bad. Confirms `f'(p) = c/(p+c)²` amplifies gradient on rare (hint) tokens too aggressively in Regime C.
- On earlier 9018 (LUFFY-only pre-`math_verify`): all metrics 0 including `critic/score/max` and `sdpo/hint/mean_reward`. Root cause was reward silent-failing, not model.
- OOM at `STUDENT_MAX_TOKEN_LEN_PER_GPU=57000`; backward peak +16 GB on top of ~62 GB forward. Reverted to 49152 (+20% over baseline 40960).
- Bedrock cost estimate `$3.3k` for Sonnet 5 was ~10× overshoot; actual with Haiku 4.5 was $180 and quality equivalent for this extractive task.

**Interpretation:**
- LUFFY (`p/(p+c)`) reshape works but interacts badly with tiny Regime C group std (only 1 success). Two mitigations available: raise `c` (flattens f(p)) or scale advantage by α. Trying α=0.25 first (cleaner ablation).
- Full Regime C activation is rare (student on diff6-8 mostly fails, hint mostly fails too on 1.7B). Effective LUFFY influence bounded by regC frequency — need to check `sdpo/regime/regC_stuck_student_frac` in wandb.

**Next:**
- Compare 9051 (α=0.25) vs 9035 (α=1.0, killed early) entropy curve. If α fixes it, apply α=0.25 to combo (9038 currently α=1.0).
- If regC frequency < 5%: LUFFY signal too sparse to matter. Consider stronger hint injection or easier eval split.
- Longer term: check if `HINT_LEVEL=level_2` (weaker hint) reduces entropy pressure vs `level_3`.

---

## 2026-07-18 — Log started

**Ran:**
- N/A — bootstrapping the log.

**Observed:**
- Current SDPO trainer lives in `verl/trainer/ppo/sdpo_ray_trainer.py` (~3084 lines).
- Latest experiment scripts under `sdpo_experiment/6_4/`.

**Interpretation:**
- Starting point for future entries.

**Next:**
- Add an entry after the next training run with wandb link + observed reward/KL curves.
