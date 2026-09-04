# openthoughts3-400k SFT + OPD results

Eval: `~/data/dapo_17k_aime2426-suffix/test.parquet` (aime24/25/26 = 30 Q each, amc23 = 40 Q),
16k max response, n=8, T=0.6 / top_p=0.95 / top_k=20, thinking mode.
All numbers **mean@8 / pass@8** in percent; `avg 4` = mean of the four datasets.

Every number is read from wandb (`rl_agent/verl_opd_dapo`) via
`opd_inference/dump_wandb_results.py`, and each row carries its run id so it can be re-checked:

- **mean@8** = `val-core/<ds>/acc/mean@8`
- **pass@8** = `val-core/<ds>/acc/best@8/mean`

> **pass@8 changed meaning in this revision.** The previous version of this file used an exact
> max-over-8-samples pass@8, which runs 2–4pp higher than verl's `best@8/mean` estimator and
> lands on clean multiples of 1/30. Example: 4B SFT aime24 was recorded as 73.33 (= 22/30), but
> the eval log for that same run prints `val-core/aime/acc/best@8/mean: 0.6837`. mean@8 is
> unaffected and matches to the last digit. **Do not compare pass@8 in this file against the
> older revision.**

Only **finished** runs with a **correct** configuration are recorded. Two caveats gate what
counts as correct — see [rope_theta](#caveat-1-rope_theta) and [inert mask](#caveat-2-inert-mask).

**Noise floor** (three identical repeats per size): mean@8 spread 1.07pp @1.7B, 1.63pp @4B;
pass@8 spread 1.97pp @1.7B, 5.68pp @4B. **Differences below ~1.6pp mean@8 or ~6pp pass@8 are
not interpretable.**

---

## 1. SFT checkpoints

`XinnanZhang/openthoughts3-math-50k8` (50k × 8 = 400k), boxed suffix on every user turn.
Official `openthinker3.yaml` hp: LR 8e-5, batch 512, 1 epoch, cutoff 20000, packing + liger + fa2.

| model | run | aime24 | aime25 | aime26 | amc23 | **avg 4** |
|---|---|---|---|---|---|---|
| Qwen3-8B *(untuned ref)* | `jiycpm8z` | 67.50 / 78.23 | 53.33 / 73.68 | 55.83 / 71.37 | 91.56 / 99.77 | **67.06 / 80.76** |
| `sft_qwen3_8b_ot400k` | `mdw104c0` | 46.25 / 70.60 | 37.08 / 53.34 | 46.25 / 62.62 | 82.19 / 93.61 | **52.94 / 70.04** |
| `sft_qwen3_4b_ot400k` | `eval4bot400k` | 42.08 / 68.37 | 34.17 / 54.87 | 37.08 / 56.09 | 74.69 / 90.27 | **47.00 / 67.40** |
| `xinnan_400k_1ep` (1.7B) | `eq9rv3if` | 10.83 / 26.05 | 13.75 / 20.69 | 9.17 / 22.68 | 46.56 / 68.35 | **20.08 / 34.44** |

**SFT is net harmful at every size**: same-size 52.94 vs 67.06 = **−14.1pp**. Survives the
rope_theta fix, so not a measurement artifact.

Cause: the 400k data is **70.6% truncated** — only 29.4% of targets close `</think>` and emit
`\boxed{}`. The rest hit a ~16k cap mid-reasoning and the template then appends `<|im_end|>`,
teaching the model to emit EOS mid-sentence. Inherited from upstream (OpenThoughts3-1.2M math
is 30.2% complete). Ruled out: prompt format and boxed-suffix alignment are byte-exact.

*Excluded:* `xinnan_400k_sft` (1.7B local) — saved by transformers 5.2.0, evaluated with
`rope_theta=10000`; its numbers measure the bug. Needs re-eval.

---

## 2. SFT students — three sizes

Students = the SFT checkpoints from §1. Common: prompt 2048 + **response 16384**, batch 128,
rollout_n=1, LR 1e-6, topk=16, 100 steps, thinking mode, 1 node × 8 GPU.

`pg%` = final `actor/distillation/pg_token_ratio` (share of tokens on the PG arm).
`baseline` = `k1_topk_overlap`; **`aopd` / `tent` / `opdt4` rows are `k1_pg_fkl_topk`, i.e. the
mask really routes.** The same names under `k1_topk_overlap` were inert (Caveat 2) and appear
here as `baseline rep 2/3`.

> **`pg%` is not defined for the baseline rows.** For non-hybrid losses the metric is only a
> diagnostic computed over an unset mask, so it reads ≈0 (0.04–0.27) rather than the 100 the
> previous revision of this file claimed. Baseline pg% is listed as n/a; do not read the
> baseline as "0% PG".

### Baseline vs SFT start, by size

| student | avg 4 SFT start | avg 4 after OPD | Δ mean@8 | Δ pass@8 |
|---|---|---|---|---|
| **8B** `ot400k` | 52.94 / 70.04 | **59.95 / 79.05** | **+7.01** | **+9.01** |
| 4B `ot400k` | 47.00 / 67.40 | 50.80 / 67.19 | +3.80 | −0.21 |
| 1.7B `400k1ep` | 20.08 / 34.44 | 20.86 / 35.13 | +0.78 | +0.69 |

**The baseline gain scales with student size.** 8B is the only size where it is large on both
metrics, and it has not plateaued by step 20 (which read 58.07). 4B gains mean@8 while pass@8
stands still. 1.7B's baseline gain is inside the 1.07pp floor. The teacher is 3.3× stronger than
the 1.7B student but only 14.1pp above the 8B one, where they also share architecture and
tokenizer.

Note this ordering is about the **baseline** loss. The best *method* per size does not follow it:
aopd is the best 1.7B run (+2.13) and the worst 4B run (−4.00).

### 8B SFT student

| method | run | pg% | aime24 | aime25 | aime26 | amc23 | **avg 4** | Δ base |
|---|---|---|---|---|---|---|---|---|
| *SFT start* | `mdw104c0` | — | 46.25 / 70.60 | 37.08 / 53.34 | 46.25 / 62.62 | 82.19 / 93.61 | **52.94 / 70.04** | — |
| baseline | `lhltlx0e` | n/a | 55.00 / 76.59 | 44.58 / 68.08 | 55.83 / 77.55 | 84.38 / 93.99 | **59.95 / 79.05** | — |
| **aopd** thr=0 | `6xx9cvjp` | 75.9 | 59.58 / 69.89 | 44.17 / 65.56 | 52.08 / 70.34 | 86.56 / 94.10 | **60.60 / 74.97** | +0.65 |
| **tent** τ=1.6 | `rk2s5yn2` | 98.0 | 60.83 / 76.22 | 40.83 / 60.23 | 47.08 / 62.56 | 81.88 / 90.79 | **57.66 / 72.45** | −2.29 |
| **opdt4** v0.5 cov0.1 eps5 | `1y6vjzlq` | 99.4 | 59.58 / 79.77 | 41.67 / 62.31 | 52.08 / 71.56 | 85.31 / 94.06 | **59.66 / 76.92** | −0.29 |
| **opdt4** v0.3 cov0.1 eps5 tp0.95 | `d0rp7mbi` | 98.8 | 54.58 / 72.81 | 41.25 / 62.42 | 50.00 / 67.23 | 84.06 / 95.36 | **57.47 / 74.46** | −2.48 |
| **opdt4** v0.3 cov0.1 eps0.5 tp0.95 ‡ | `j4ik04uj` | 89.7 | 54.58 / 71.04 | 43.75 / 70.08 | 51.67 / 71.27 | 80.94 / 93.47 | **57.73 / 76.46** | +0.91 |

‡ job 16187, **still running** — step-84 reading, not a final result.

aopd gives the **highest 8B mean@8 of any run (60.60)** but costs 4.08pp of pass@8, so it is a
mean/pass trade, not a clean win — and +0.65 is inside the noise floor anyway. tent's −2.29 is
just past the floor.

### 4B SFT student — the only size with a full mask sweep

| method | run | pg% | aime24 | aime25 | aime26 | amc23 | **avg 4** | Δ base |
|---|---|---|---|---|---|---|---|---|
| *SFT start* | `eval4bot400k` | — | 42.08 / 68.37 | 34.17 / 54.87 | 37.08 / 56.09 | 74.69 / 90.27 | **47.00 / 67.40** | — |
| baseline | `8t0h14sp` | n/a | 47.50 / 66.27 | 37.08 / 57.32 | 40.83 / 54.76 | 77.81 / 90.41 | **50.80 / 67.19** | — |
| baseline rep 2 | `k21kw9vr` | n/a | 45.00 / 69.13 | 40.42 / 59.73 | 41.67 / 61.53 | 75.00 / 90.97 | **50.52 / 70.34** | −0.28 |
| baseline rep 3 | `9rdh0a20` | n/a | 44.17 / 65.13 | 35.42 / 49.52 | 40.83 / 55.15 | 76.25 / 88.86 | **49.17 / 64.66** | −1.63 |
| **aopd** thr=0 | `zyt1z46m` | 70.6 | 41.25 / 65.80 | 33.33 / 53.60 | 41.67 / 55.57 | 70.94 / 89.96 | **46.80 / 66.23** | **−4.00** |
| **opdt4** vote0.3 | `vdxcshlt` | 82.0 | 47.92 / 71.86 | 35.83 / 54.94 | 40.00 / 58.48 | 79.38 / 94.31 | **50.78 / 69.90** | −0.02 |
| **tent** τ=1.6 | `c99wrtur` | 97.6 | 44.58 / 67.64 | 42.08 / 58.77 | 42.08 / 52.94 | 77.50 / 91.30 | **51.56 / 67.66** | +0.76 |
| **opdt4** vote0.3 eps10 | `umu7p78g` | 97.9 | 43.75 / 66.10 | 37.50 / 58.73 | 42.50 / 57.15 | 80.00 / 93.16 | **50.94 / 68.78** | +0.14 |
| **opdt4** v0.3 cov0.1 eps5 tp0.95 | `yfggmwma` | 97.2 | 48.75 / 70.41 | 37.50 / 57.25 | 40.83 / 56.85 | 78.44 / 91.30 | **51.38 / 68.95** | +0.57 |
| **opdt4** v0.3 cov0.1 eps0.5 tp0.95 | `2xjfay9b` | 83.5 | 42.08 / 64.47 | 38.75 / 56.51 | 42.50 / 60.38 | 77.50 / 91.37 | **50.21 / 68.18** | −0.60 |

> **Correction:** an earlier revision listed this run at **51.82 / 68.60 (+1.02)**. That was its
> **step-60 peak**, read while the job was still running. The final step-100 value is
> **50.94 (+0.14)**, i.e. indistinguishable from baseline. Its curve is
> 47.63 → 46.48 → 51.82 → 50.83 → 50.94 over steps 20–100, so mid-run readings on this run swing
> by up to 5.3pp — **do not quote a number from a live job.**

**Within 4B, avg-4 rises with `pg%`**: 70.6 → 46.80, 82.0 → 50.78, 97.6 → 51.56, 97.9 → 50.94.
The more tokens routed to the FKL arm, the worse the model. But once the eps10 run's final value
replaced its mid-run peak, the top three settings (50.78 / 51.56 / 50.94) collapsed into a
0.78pp band around the 50.80 baseline — so the pattern reduces to a single fact: **aopd is bad
at 4B (−4.00pp, 2.5× the noise floor) and everything else ties the baseline.**

**This is a 4B-only pattern, and 4B is the outlier.** Lining up the three sizes at nearly the
same routing fraction:

| student | aopd run | pg% | Δ baseline mean@8 |
|---|---|---|---|
| 1.7B SFT | `6a38xpb4` | 67.3 | **+2.13** |
| 4B SFT | `zyt1z46m` | 70.6 | **−4.00** |
| 8B SFT | `6xx9cvjp` | 75.9 | +0.65 |

All three route roughly a quarter to a third of tokens to FKL, yet 1.7B gains and 4B loses —
a 6.1pp swing across a 3.3pp difference in pg%. So the effect is **neither monotone in pg% nor
monotone in student size**, and "less FKL is better" is not a rule. The 4B result is the one
that needs explaining; it is the only negative aopd run among the three SFT students.

### 1.7B SFT student

| method | run | pg% | aime24 | aime25 | aime26 | amc23 | **avg 4** | Δ base |
|---|---|---|---|---|---|---|---|---|
| *SFT start* | `eq9rv3if` | — | 10.83 / 26.05 | 13.75 / 20.69 | 9.17 / 22.68 | 46.56 / 68.35 | **20.08 / 34.44** | −0.78 |
| **aopd** thr=0 | `6a38xpb4` | 67.3 | 12.50 / 26.17 | 18.75 / 29.81 | 15.42 / 29.60 | 45.31 / 67.38 | **22.99 / 38.24** | **+2.13** |
| **opdt4** vote0.3 eps10 | `zo9tt461` | 95.5 | 11.67 / 27.39 | 16.25 / 33.20 | 13.33 / 28.98 | 44.69 / 64.42 | **21.48 / 38.50** | +0.62 |
| baseline | `2q7t70uu` | n/a | 13.33 / 26.95 | 16.67 / 28.44 | 8.75 / 18.43 | 44.69 / 66.70 | **20.86 / 35.13** | — |
| **tent** τ=1.6 | `v01gyicv` | 96.6 | 12.92 / 26.46 | 15.00 / 23.17 | 8.33 / 23.75 | 45.00 / 67.60 | **20.31 / 35.24** | −0.55 |
| baseline rep 2 | `43axjj72` | n/a | 11.67 / 27.45 | 14.17 / 31.26 | 6.67 / 23.69 | 47.81 / 65.90 | **20.08 / 37.08** | −0.78 |
| **opdt4** v0.5 cov0.1 eps5 | `bp1y00pz` | 96.3 | 17.08 / 26.63 | 15.00 / 27.26 | 11.67 / 20.74 | 44.38 / 65.06 | **22.03 / 34.92** | +1.17 |
| **opdt4** v0.3 cov0.1 eps5 tp0.95 | `sl8wjf53` | 93.6 | 13.33 / 26.96 | 15.00 / 24.30 | 11.67 / 21.51 | 46.88 / 68.31 | **21.72 / 35.27** | +0.86 |
| **opdt4** v0.3 cov0.1 eps0.5 tp0.95 | `416nu6jm` | 77.5 | 13.75 / 28.10 | 15.42 / 31.22 | 10.83 / 25.05 | 44.06 / 60.78 | **21.02 / 36.29** | +0.16 |
| baseline rep 3 | `8akt842o` | n/a | 11.67 / 22.95 | 13.75 / 27.80 | 11.25 / 28.01 | 42.50 / 61.68 | **19.79 / 35.11** | −1.07 |

**aopd is the only result at 1.7B that clears the noise floor** (+2.13pp vs a 1.07pp floor), and
its val curve rises monotonically with no dip: 20.05 → 20.16 → 21.74 → 22.37 → 22.99 at steps
20/40/60/80/100, with pg% steady at 63–68. Every other 1.7B run shares a common dip at step 40
(17.0–18.2) because `data.shuffle=False` gives them all the same batch order; aopd is the only
one that does not dip.

Everything else at 1.7B is inside the floor: opdt4 +0.62, tent −0.55.

*Partial (died before step 100, listed for completeness only):* `xglsupnb` opdt4 vote0.3 @32 →
20.73 / 36.67 (CUDA OOM — single-node, no respool); `p1cljd2l` tent @32 → 20.05 / 33.91.

Note `43axjj72` is named "aopd" but ran `k1_topk_overlap`, so it is an inert-mask baseline
repeat (Caveat 2), not aopd. The real aopd row is `6a38xpb4` (job 15811, 10h57m, respool
recipe). Nine earlier 1.7B aopd attempts died first, all on CUDA OOM in the FKL top-k backward
(13.4–15.2 GiB requested against 10–13.7 GiB free) — aopd routes ~33% of tokens to FKL, roughly
10× tent's share, so it is the heaviest memory case at this length.

### Training-rollout truncation — aopd lengthens generations at every size

`clip%` = `response_length/clip_ratio`, the share of *training* rollouts that hit the response
cap; `complete%` = `response/complete_ratio_non_aborted`. `val_reslen` is the mean *eval*
response length, which is what the scores are actually computed on.

| student / method | clip% | complete% | reslen | val_reslen | avg 4 mean@8 |
|---|---|---|---|---|---|
| 1.7B aopd | **93.8** | 6.2 | 16182 | 14448 | 22.99 |
| 1.7B opdt4 | 78.1 | 21.9 | 15382 | 14617 | 21.48 |
| 1.7B baseline | 80.5 | 19.5 | 15487 | 14702 | 20.86 |
| 1.7B tent | 75.0 | 25.0 | 15010 | 14640 | 20.31 |
| 4B aopd | 56.2 | 43.8 | 13682 | 13229 | 46.80 |
| 4B baseline | 46.9 | 53.1 | 12882 | 12768 | 50.81 |
| 8B aopd | 38.3 | 61.7 | 12140 | 12348 | 60.60 |
| 8B baseline | 37.5 | 62.5 | 12142 | 11925 | 59.95 |

**aopd raises truncation over baseline at every size** (+13.3pp @1.7B, +9.3pp @4B, +0.8pp @8B),
and the size ordering of that increase matches where aopd helps or hurts least — but not its
sign, so this does not explain the 4B reversal either.

**The 1.7B aopd gain is not a length artifact.** Its training rollouts are 93.8% truncated and
only 6.2% terminate on their own, which is alarming on its own terms, but at *eval* time it
produces the **shortest** responses of any 1.7B run (14448 vs baseline 14702). Eval length is
essentially constant across the four 1.7B runs (14448–14720, a 1.9% spread), so the +2.13pp
cannot be attributed to "aopd just writes longer answers when scored."

What the 93.8% does mean: aopd's own distillation signal was computed almost entirely on
truncated sequences, i.e. on reasoning that never reaches an answer. That it still improves the
eval score is the surprising part, and it makes the 1.7B result worth reproducing before being
built on.

Baseline truncation also tracks student size (80.5% → 46.9% → 37.5% for 1.7B → 4B → 8B), which
is an independent reason §2's cross-size comparisons are not clean: the 1.7B student is running
into the length cap four times as often as the 8B one.

### Cost

aopd runs ~635 s/step vs tent ~333 s/step, tracking pg%: the FKL arm must materialize top-k
logits, so routing ~30% of tokens there roughly doubles step time. The 1.7B aopd run took
10h57m for 100 steps under the 2-node respool recipe.

### The 2-node recipe (needed for 1.7B/4B at reslen 16384)

```
TEACHER_RESOURCE_POOL=True                 # student 8 GPU (node0) + teacher 8 GPU (node1)
+distillation.teacher_model.inference.enable_sleep_mode=False   # "+" required: not in the YAML struct
free_cache_engine                          # leave at default True
SP=2 ; ppo_max_token_len_per_gpu=9216      # (2048+16384)/2, satisfies seqlen_balancing.py:384
gpu_memory_utilization=0.5 (both)
sbatch: --nodes=2 --gpus-per-node=8, explicit ray start head/worker, srun --overlap,
        and `unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES` *inside* every srun
```

The peak is the `(packed_tokens, vocab)` bf16 logits term (~15.4 GiB measured, needing ~3× that
for forward + grad + temp). SP=2 halves the per-rank token count, bringing it to ~7.7 GiB and
leaving a ~2.5× margin.

Seven failure modes, all now ruled out: colocated OOM (~1.1 GiB short); `expandable_segments:True`
racing vLLM sleep-mode for the CUDA VMM API; teacher `raw_delete` crash from
`CUDAPluggableAllocator` still being registered; `free_cache_engine=False` alone (guards
`sleep()` but **not** `wake_up()`, creating a wake-without-sleep asymmetry); a 4+4 same-node
split (halved GPUs but `max_num_seqs=128` unchanged, doubling per-instance KV demand); Ray
seeing only 8 of 16 GPUs without an explicit multi-node cluster; and a **stale
`/tmp/ray/ray_current_cluster`** left by a previous multi-node job on the same node, which makes
a later single-node launch try to rejoin a dead head (`ConnectionError: Failed to connect to Ray
cluster at <old_ip>:6379`, job 15997). Fix: `unset RAY_ADDRESS; ray stop --force;
rm -rf /tmp/ray/ray_current_cluster` before launching.

---

## 3. Base students — three sizes

Students = untuned `Qwen/Qwen3-{1.7B,4B,8B}-Base`, teacher Qwen3-8B, **response 4096**
(not 16384), **non-think**, 200 steps. Not comparable to §2: different length cap, no SFT warm
start, non-think prompts. `reslen` = final training `response_length/mean` (cap 4096).

| student | method | run | pg% | clip% | reslen | aime24 | aime25 | aime26 | amc23 | **avg 4** |
|---|---|---|---|---|---|---|---|---|---|---|
| 8B-Base | v4 vote0.3 eps0.5 † | `zrbek8zd` | 92.9 | **4.7** | 1521 | 22.50 / 41.77 | 17.50 / 31.60 | 18.75 / 33.91 | 61.56 / 83.89 | **30.08 / 47.79** |
| 8B-Base | v4 vote0.3 eps5 | `171sfne3` | 99.4 | **3.9** | 1434 | 21.67 / 36.56 | 17.92 / 27.02 | 17.92 / 29.47 | 62.81 / 85.91 | **30.08 / 44.74** |
| 8B-Base | v4 vote0.2 | `n3cutrcf` | 91.9 | ~100 | 4096 | 20.00 / 36.59 | 16.25 / 29.24 | 17.08 / 32.32 | 61.56 / 83.60 | **28.72 / 45.44** |
| 8B-Base | v4 vote0.5 | `nm12kcea` | 97.9 | ~100 | 4033 | 21.25 / 33.88 | 16.67 / 29.53 | 15.00 / 28.56 | 61.56 / 84.31 | **28.62 / 44.07** |
| 8B-Base | v4 vote0.3 eps10 | `u5vseqzs` | 99.5 | 100 | 4096 | 22.50 / 37.32 | 17.08 / 27.25 | 15.00 / 21.78 | 59.06 / 78.67 | **28.41 / 41.26** |
| 8B-Base | aopd thr=0 | `ocz57y41` | 82.5 | 100 | 4096 | 22.50 / 38.01 | 16.25 / 20.99 | 12.50 / 24.72 | 60.31 / 84.87 | **27.89 / 42.15** |
| 8B-Base | tent τ=1.6 | `9f0qmx2y` | 97.4 | 100 | 4096 | 20.83 / 33.12 | 15.42 / 28.70 | 14.17 / 24.58 | 57.81 / 79.77 | **27.06 / 41.54** |
| 4B-Base | aopd thr=0 | `vibe1ovk` | 76.7 | — | 1987 | 15.83 / 34.10 | 16.25 / 25.91 | 10.83 / 22.29 | 55.94 / 83.89 | **24.71 / 41.55** |
| 1.7B-Base | all-PG | `n1mxr8w7` | 100 | 89.1 | 3759 | 8.75 / 18.72 | 5.42 / 11.70 | 6.67 / 11.90 | 36.56 / 65.76 | **14.35 / 27.02** |
| 1.7B-Base | aopd thr=0 | `i1ypph73` | 66.7 | 42.2 | 3043 | 6.25 / 19.45 | 5.42 / 17.16 | 4.58 / 12.55 | 33.12 / 58.09 | **12.34 / 26.81** |

The 1.7B-Base all-PG row is the **degeneration reference** (`pg_token_ratio` = 1.0000 throughout).
Checkpoints kept at `global_step_{80,160,200}` under
`/fsx/xinnanzh/ckpts/opd_degen_1p7b_base_8b_nonthink` (7.6 GB each, `save_contents=[model,extra]`)
for generation-level inspection.

### §3 does not support method conclusions — five of six 8B-Base runs collapsed

`clip%` = `response_length/clip_ratio`, the share of training rollouts hitting the 4096 cap.

| 8B-Base run | clip% | complete% | reslen | val_reslen | avg 4 |
|---|---|---|---|---|---|
| v4 vote0.3 eps0.5 † `zrbek8zd` | **4.7** | **95.3** | 1521 | 3059 | **30.08** |
| v4 vote0.3 eps5 `171sfne3` | **3.9** | **96.1** | 1434 | 3153 | **30.08** |
| v4 vote0.3 eps10 `u5vseqzs` | 100.0 | 0.0 | 4096 | 8192 | 28.41 |
| aopd `ocz57y41` | 100.0 | 0.0 | 4096 | 8177 | 27.89 |
| tent `9f0qmx2y` | 100.0 | 0.0 | 4096 | 8192 | 27.06 |

**The runs split cleanly into two healthy and five collapsed.** The collapsed ones sit at
**100% truncation and 0% completion**, with eval responses pinned against the 8192 val cap — they
never emit `\boxed{}` because they never stop. vote0.2 (reslen 4096) and vote0.5 (4033) are in the
same state.

So the earlier reading was too generous. It is not that "vote0.3 is best-observed but inside
noise"; it is that **the five collapsed runs measure degeneration, not the mask.** They span
27.06–28.72 (1.66pp), while both healthy runs land at **exactly 30.08**.

That two independent healthy runs agree to 0.01pp on mean@8 — with different `eps_low` (none vs 5)
and different pg% (92.9 vs 99.4) — while their pass@8 differs by 3.05pp (47.79 vs 44.74) suggests
30.08 is close to what this student reaches at 4096 regardless of routing, i.e. the mask is not
the binding constraint here. **Length collapse is the only effect §3 actually resolves.**

† `eps_low` was not passed by that script, so it used the code default **0.5**
(`hybrid_masks.py:1116`), i.e. R2 fires on `π_T/π_S > 1.5`. The three settings are therefore
ratio thresholds of **1.5 / 6 / 11** for eps_low 0.5 / 5 / 10, and pg% orders accordingly
(92.9 → 99.4 → 99.5): a lower threshold routes more tokens to FKL.

`eps_low=5` stayed healthy (3.9% clip) while `eps_low=10` collapsed (100%), but **this is not
attributable to eps_low** — their pg% differs by only 0.1pp (99.4 vs 99.5), and there are no
repeats at 8B-Base, so run-to-run variance cannot be ruled out. An earlier revision claimed the
collapse was driven by *which* tokens reach the FKL arm rather than how many; that was
over-reading two single runs and is retracted. All that is established: one collapsed, one did
not, cause unknown.

**The 1.7B-Base pair is confounded too, in the opposite direction.** aopd (12.34) truncates
*less* than all-PG (42.2% vs 89.1%) and generates far shorter eval responses (4300 vs 8005), yet
scores 2.01pp lower. So the one comparison I previously called "strictly comparable — same
student, same length, same steps" is not length-matched at all, and the −2.01pp cannot be
attributed to the mask either. Retracted.

What survives from §3: routing demonstrably reaches the loss (see the `pg_token_ratio` trace
below), and the 4096 cap is too tight for non-think base students at this batch size — every
setting except vote0.3 runs away to the cap within 200 steps. Any future base-student sweep needs
a longer cap or a length penalty before its numbers mean anything.

One structural observation does survive the length confound, because it is about routing rather
than scores: **aopd's `pg%` rises monotonically with base-student size** (66.7 → 76.7 → 82.5 for
1.7B → 4B → 8B). aopd routes to FKL where `−k1 < 0`, i.e. where the teacher assigns *more*
probability than the student, and a stronger student hits that condition less often. So aopd's
intervention shrinks exactly as the student improves — its FKL arm is most active where it is
least trustworthy. The same ordering holds on the SFT students (67.3 → 70.6 → 75.9).

**`eps_low=10` nearly switches the FKL arm off.** It drives pg% to 99.5 at 8B-Base and 95–98.5
at 4B/1.7B SFT, so those runs are approximately the all-PG baseline. On 8B-Base that is worse
than vote0.3 (28.41 vs 30.08, but both length-confounded); on the SFT students it is marginally
the best setting at 4B (51.82, partial) and 1.7B (21.48) — both inside noise.

*Still missing:* tent for 1.7B/4B-Base, all-PG control for 4B/8B-Base. (8B-Base tent is no
longer missing — `9f0qmx2y` did finish; see Caveat 3 for why it was misfiled as a failed launch.)

### `opd_theory_guided4` — the router used in the v4 rows

Two rules (`verl/trainer/distillation/hybrid_masks.py:1061`) under `k1_pg_fkl_topk`:

- **R1** `Σ_{c ∈ teacher top-0.9} π_S(c) < 0.2` → FKL
- **R2** teacher-prob-weighted vote of `[π_T(c)/π_S(c) > 1 + eps_low]` over the nucleus > `vote` → FKL
- else → PG (k1)

R1 at 0.2 fires on 1.07% of positions but adds just 0.01% beyond R2 (measured on 5.06M saved
positions), so it is a floor, not a co-equal rule.

**Routing did work** — direct evidence the mask reaches the loss:

| first → final | vote0.2 | vote0.3 | vote0.5 |
|---|---|---|---|
| `pg_token_ratio` | 0.696 → 0.919 | 0.742 → 0.929 | 0.810 → 0.979 |
| `opd_low_coverage_ratio` (R2 fires) | 0.302 → 0.081 | 0.254 → 0.071 | 0.186 → 0.021 |

Neither 1.0 nor constant, and correctly ordered by threshold. Offline prediction for vote0.3 was
19.7% FKL vs 25.8% measured at step 1. The decay to 7% is expected — `top_p_coverage_mean` rises
0.875 → 0.973 as the student converges, so a fixed threshold fires progressively less. A
percentile threshold would hold the ratio constant.

---

## 4. In flight

### The `opd_theory_guided4` sweep is finished, and the family does not work

Eleven runs across three students, `eps_low` from 10 down to 0.5, FKL share from 0.6% to 22.5%.
**Not one clears its size's noise floor.** The single closest is 1.7B `bp1y00pz` at +1.17 against
a 1.07pp floor — 0.10pp over the line.

Sorted by FKL share, with Δ against each student's own baseline:

| FKL share | student / setting | Δ mean@8 |
|---|---|---|
| 0.6% | 8B eps5 v0.5 | −0.29 |
| 1.2% | 8B eps5 tp0.95 | −2.48 |
| 2.1% | 4B eps10 | +0.14 |
| 2.8% | 4B eps5 tp0.95 | +0.57 |
| 3.7% | 1.7B eps5 v0.5 | +1.17 |
| 4.5% | 1.7B eps10 | +0.62 |
| 6.4% | 1.7B eps5 tp0.95 | +0.86 |
| 10.3% | 8B eps0.5 tp0.95 ‡ | +0.91 |
| 16.5% | 4B eps0.5 tp0.95 | −0.60 |
| 18.0% | 4B eps0.5 | −0.03 |
| 22.5% | 1.7B eps0.5 tp0.95 | +0.16 |

Grouping: FKL < 7% averages **+0.08**, FKL > 10% averages **−0.65**. The 0.73pp gap is smaller
than every noise floor here, so even that weak ordering is not established.

**The decisive point is what "low FKL" means.** At 0.6% FKL, 99.4% of tokens take exactly the
baseline k1 path, so opdt4's best results are the ones where it barely intervenes — they are
baseline reproductions bought at roughly 2x the step time (~635 s/step with an active FKL arm vs
~333 s/step for plain k1). Pushing `eps_low` higher converges to the baseline exactly. That is
not a recipe, it is evidence the routing criterion adds nothing.

**Contrast with aopd, the one mask that does something.** aopd runs at 24–33% FKL and produces
the only effect above any noise floor in this whole file: **+2.14pp at 1.7B**. It routes a
comparable share of tokens to 1.7B `416nu6jm` (32.7% vs 22.5%) yet scores +2.14 against +0.16.
So the difference is not *how many* tokens reach the FKL arm but *which*:

- **aopd** tests the token that was actually sampled: `log π_T(u) − log π_S(u) < 0`, i.e. route
  when the student was over-confident on its own choice.
- **opdt4 R2** tests the shape of the teacher's nucleus via a probability-weighted vote over all
  candidates, independent of what was sampled.

A per-token, post-hoc criterion works; a per-position, distribution-shape criterion does not.
Any further work on this router should change the criterion, not its thresholds.

‡ still running at the time of writing (step 84 of 100).

---

**Size-matched opdt4 trio** — identical mask settings (`vote=0.5, cov=0.1, eps_low=5`) across all
three SFT students, so the set isolates student size. All log **online** to wandb (verified
reachable from a compute node), so no post-hoc sync is needed.

| job | student | run | nodes | pg% | status |
|---|---|---|---|---|---|
| 16034 | 1.7B SFT | `bp1y00pz` | 2 | — | running, 79% |
| 16035 | 4B SFT | `yhm6ynir` | 1 | 98.5 | running, 55% (47.61 @step 55) |
| 16033 | 8B SFT | `1y6vjzlq` | 2 | 99.2 | running, 43% (55.13 @step 42) |

Why this setting: aopd (pg 67.3) is the only 1.7B result above the noise floor while being the
worst 4B result at nearly the same routing fraction, and every high-pg% setting tried at 1.7B
(tent 96.6, opdt4 eps10 95.5) sat inside the floor. `eps_low=5` lands between those regimes.
16033 is also the first opdt4 run on the 8B SFT student at all.

> **The premise was wrong: `eps_low=5` does not keep R2 firing.** Measured pg% on the trio is
> **98.6 → 99.2 @8B** (`1y6vjzlq`) and **98.5 @4B** (`yhm6ynir`), i.e. ~1% of tokens reach the FKL
> arm — the same near-inert regime as `eps_low=10` (95.5–99.5%). Combined with `vote=0.5` (a
> higher vote threshold is *harder* to trigger), the FKL arm barely fires, so all three runs are
> approximately the baseline with extra cost. Confirmed by `171sfne3`: `eps_low=5` at 8B-Base also
> reads 99.4%.
>
> To actually exercise the router, the next sweep needs a much smaller `eps_low` (≤1, i.e. ratio
> > 2, versus the current > 6) and/or a lower `vote`. For reference, aopd — the only setting that
> beat the noise floor anywhere — runs at pg 67–76%, so ~25–33% FKL is the regime worth targeting.

Unrelated, also running:

Both of the previous batch have now **finished** and are folded into the tables above:

| job | run | what | outcome |
|---|---|---|---|
| 15921 | `umu7p78g` | 4B SFT opdt4 vote0.3 eps_low=10 | 50.94 / 68.79 (+0.14, noise) |
| 16032 | `171sfne3` | 8B-Base opdt4 vote0.3 eps_low=5 | 30.08 / 44.74, stayed healthy (3.9% clip) |

---

## Caveat 1: rope_theta

Training env transformers 5.2.0 nests `rope_theta` under `rope_parameters`; inference env
4.55.2 doesn't know the field and silently falls back to **10000** instead of **1000000**.
Generations start coherent then collapse to character garbage. No error raised.

Same 4B checkpoint, eval differing only by the fix: aime24 mean@8 **1.25 → 42.08**,
avg 4 **11.68 → 47.00**, degenerate-repetition **17.4% → 2.4%**.

Fix: `opd_inference/fix_llamafactory_tokenizer_config.py <ckpt> --also_generation_config`
(idempotent; also repairs `extra_special_tokens` list→dict and a `generation_config.json`
missing `<|im_end|>` in `eos_token_id`). Wired into the `8_28`/`8_29` sbatch scripts.

**Consequence:** an earlier round measured 74–76% degenerate repetition on 8B SFT checkpoints
and concluded the SFT data destroyed the model. Those were all 5.2.0-saved — that was the rope
bug; a correct eval gives 1.5–2.4%. The separate finding that SFT is net harmful does survive.

## Caveat 2: inert mask

`hybrid_mask_strategy` is only consumed by loss modes registered `use_hybrid=True`:

```
k1_pg_fkl_topk      use_hybrid=True    <- real hybrid
k1_pg_jsd_topk      use_hybrid=True
rkl_all_fkl_masked  use_hybrid=True
k1_topk_overlap     (not hybrid)       <- what the "rep 2/3" rows ran
```

`compute_k1_topk_overlap` (`losses.py:1139`) computes plain k1 over all tokens; it never reads
`hybrid_mask_strategy`, never computes FKL, never splits arms. The mask only reaches
`_compute_pg_diag_for_non_hybrid`, which exists to "surface the same routing **diagnostic**".
So `hybrid_mask_strategy`, `pg_loss_coef` and `supervised_loss_coef` were silently ignored —
every `aopd`/`tent` run under `k1_topk_overlap` is a baseline repeat, which is why they are
relabelled `baseline rep 2/3` above and reused as the noise-floor estimate.

Corrected runs use `k1_pg_fkl_topk` and carry a `fklhybrid_` prefix in `EXP_NAME`.
`USE_POLICY_GRADIENT` stays `True` but is inert: dispatch is `if use_hybrid: ... elif
use_policy_gradient: ...`.

## Caveat 3: wandb bookkeeping

Offline run dirs are triaged by `files/wandb-summary.json`, never by scanning the binary
`.wandb` datastore — `scan_record()` stops early on many files, so a 200-step run can read back
as 0 steps and be mistaken for a crashed launch. Keep rule for syncing:
`has_val OR _step >= 20` (`has_val` is required because eval-only runs legitimately finish at
`_step = 0`). Tooling: `opd_inference/list_offline_wandb_runs.py`,
`sync_good_wandb_runs.sh`, `delete_junk_wandb_runs.py` (dry-run by default),
`dump_wandb_results.py`, `wandb_val_curve.py`, `wandb_length_stats.py`.

**That keep rule lost three completed runs.** `6a38xpb4` (1.7B aopd, job 15811, 100 steps) has
**no `wandb-summary.json`, no `output.log`, no `config.yaml`** — only `requirements.txt` and a
20 MB `run-6a38xpb4.wandb`. wandb never wrote the files dir, so summary-based triage saw nothing
and classified a finished 11-hour run as a failed launch. Adding an `inspect` flag for
"summary missing but datastore ≥ 5 MB" immediately surfaced two more of the same kind:

| run | job | what | elapsed | result |
|---|---|---|---|---|
| `6a38xpb4` | 15811 | 1.7B SFT aopd | 10h57m | 22.99 / 38.24 — best 1.7B run |
| `9f0qmx2y` | 15807 | 8B-Base tent | 13h03m | 27.06 / 41.54 — was listed as missing |
| `i1ypph73` | 15804 | 1.7B-Base aopd | 06h05m | 12.34 / 26.81 — was the un-attributed row |

All three report `COMPLETED` in `sacct` and `Training Progress: 100%` in their slurm logs. So:
**a large `.wandb` with a missing summary is a "needs inspection" case, not junk** — and
`sacct -j <jobid>`, not the wandb directory, is the authority on whether a job finished. The
`--inspect_mb` flag in `list_offline_wandb_runs.py` now enforces this.

Deleting an online run **tombstones its id**: a later `wandb sync` of the same dir prints
`... done.` but the server discards the data and `api.run()` still 404s. Recovery requires
`wandb sync --id <fresh_id> <dir>` — which is why the 4B SFT eval now lives under the manual id
`eval4bot400k` instead of its original `m0e9df26`.
