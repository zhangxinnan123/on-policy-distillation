# 11_7 — OPD / EOPD / AOPD / opd_theory_guided4 on a coding task

Four distillation routing methods × two student sizes, on CodeR1-12k, evaluated on
HumanEval+, MBPP+ and LiveCodeBench.

| | |
|---|---|
| Students | `Qwen3-1.7B-Base`, `Qwen3-4B-Base` |
| Teacher | `Qwen3-8B` (colocated, TP=2) |
| Train data | `ganler/code-r1-12k` → 12452 prompts |
| Eval data | HumanEval+ 164 · MBPP+ 378 · LiveCodeBench **v6 shard** 175 = **717** |
| Response length | 4096 (train and val) |
| Mode | **non-thinking**, matching the math arms |
| Cluster | poe-skywalker, `ml-p4d-24xlarge` (A100 40GB), 1 node × 8 GPU per arm |
| Steps | 200, `test_freq=50` |

## The four arms

| Arm | `loss_mode` | `hybrid_mask_strategy` | Routes? |
|---|---|---|---|
| `opd` | `k1_topk_overlap` | `all` | no — 100% PG by construction |
| `eopd` | `k1_pg_fkl_topk` | `teacher_topk_entropy` (thr 1.0) | yes |
| `aopd` | `k1_pg_fkl_topk` | `aopd` (thr 0.0) | yes |
| `opdt4` | `k1_pg_fkl_topk` | `opd_theory_guided4` | yes |

`opdt4` uses the best math configuration (`teacher_top_p=0.9`,
`coverage_threshold=0.2`, `eps_low=2`, `fkl_vote_threshold=0.3` — run `f11inhls`),
carried over unchanged so this tests transfer rather than a re-tuned fit.

Only `k1_pg_fkl_topk` / `k1_pg_jsd_topk` / `rkl_all_fkl_masked` are registered
`use_hybrid=True`, so only those actually route. The `opd` arm is the one case
where a non-hybrid mode is intended.

## Running

```bash
# 1. Build the data (once; needs HF, which poe-skywalker can reach)
conda activate verl && cd /fsx/xinnanzh/on-policy-distillation
python3 jx_experiment/11_7/prepare_code_data.py --out ~/data/code_opd \
    --prompt-mode nonthink --lcb-version v6_only

# 2. Verify the eval specs by scoring the datasets' own reference solutions
python3 jx_experiment/11_7/validate_specs.py --data ~/data/code_opd

# 3. Smoke-test one arm (3 steps, validation on)
RUN=jx_experiment/11_7/run_1p7b_base_code_opdt4_p4.sh
sbatch --export=ALL,RUN=$RUN,TOTAL_STEPS=3,TEST_FREQ=2,EXP_SUFFIX=_smoke \
  --job-name=code_smoke jx_experiment/11_7/submit_code_p4.sbatch

# 4. Submit the set
for m in opd eopd aopd opdt4; do for s in 1p7b 4b; do
  RUN=jx_experiment/11_7/run_${s}_base_code_${m}_p4.sh
  sbatch --export=ALL,RUN=$RUN --job-name=${m}_${s} jx_experiment/11_7/submit_code_p4.sbatch
done; done
```

Harness unit tests (no GPU, no cluster): `python3 jx_experiment/11_7/test_code_reward_on_cpu.py`

## Running on agb05 (MSI, no Slurm)

8x A100-40GB, 16 CPUs, glibc 2.28. `launch_agb05.sh` wraps an unchanged arm:
`seed` conda env + this repo first on `PYTHONPATH` + flash-attn 2.8.3 built from
source for sm80 in `/projects/standard/mhong/zhan9359/pyextra/opd_seed` (the
prebuilt wheel needs glibc 2.32), Ray told 64 logical CPUs, one code sandbox per
reward worker.

```bash
ssh agb05 "cd $REPO && setsid nohup bash jx_experiment/11_7/launch_agb05.sh \
  jx_experiment/11_7/run_1p7b_base_code_opd_p4.sh > logs/code_opd_1p7b.log 2>&1 < /dev/null &"
```

## Online eval in the opd-test-time-scaling (TTS) format

[Geraldxm/opd-test-time-scaling](https://github.com/Geraldxm/opd-test-time-scaling)
evaluates code offline; this reproduces its eval inside verl validation:

```bash
python3 jx_experiment/11_7/prepare_code_eval_tts.py --tts-repo <their clone> \
    --out ~/data/code_opd_tts                                   # test_tts.parquet, 717
python3 jx_experiment/11_7/subsample_eval.py --data ~/data/code_opd_tts \
    --src test_tts.parquet --n 50                               # optional subset
CODE_TEST_FILE=~/data/code_opd_tts/test_tts.parquet VAL_MAX_RESPONSE_LENGTH=8192 ...
```

- Prompts: their canonical problems + prompt files, verbatim (HE+/MBPP+ "Return only
  the complete Python solution, without Markdown fences"; LCB system + `### Format`).
- Extraction: theirs (`spec["extract"]`): HE+/MBPP+ first complete fence + header
  prefix, LCB last complete fence. Verified identical to their converter on 400
  real generations.
- LCB judge: their vendored official `lcb_runner` (`vendor_lcb/`, SHA-256 matches
  their manifest), public + private tests of `test6.jsonl` at their pinned revision,
  6s per test.
- Differences: one val temperature for all benchmarks (theirs: 0.7/0.95 HE+/MBPP+,
  1.0/0.8 LCB); n=8 (theirs 512/128); EvalPlus via the HF `test` harness.
- `compare_with_tts_judge.py` re-judges a validation dump with their EvalPlus /
  lcb_runner for per-sample agreement (runs generated code without a sandbox).
- Their code runs: Qwen3-1.7B (instruct) student, batch 128, lr 5e-7 cosine,
  4096 response, checkpoints to step 500.

Branch note: the `aopd` and `opd_theory_guided4` arms need masks that exist only
on `zxn-opd-mask-sweep`; the `opd` (`all`) and `eopd` arms run on `zxn-dev`.

## Harness fixes (2026-11-07)

- Solution stdout/stderr no longer share the verdict channel. Before, a module-level
  `print(...)` in a HumanEval+/MBPP+/LCB-functional answer was flushed *after*
  the verdict, so the parent misread it (acc 0) or crashed (`json.loads("3")` → int).
- LCB stdio: `sys.stdin.buffer` and `open(0)` now see the test input.
- `compute_score` never raises; train-set samples are not executed.
- Code extraction prefers the block that DEFINES the target function (a trailing
  "### Test Case:" block made MBPP+ look like it collapsed 42% -> 5%).
- Regression tests in `test_code_reward_on_cpu.py` (57 total).

## Files

| File | Role |
|---|---|
| `_common_code_p4.sh` | All shared config. Sourced by every arm. |
| `run_{1p7b,4b}_base_code_{opd,eopd,aopd,opdt4}_p4.sh` | The 8 arms — each sets only what differs |
| `submit_code_p4.sbatch` | Slurm wrapper, generic via `RUN=` |
| `prepare_code_data.py` | Builds train/test parquet + `MANIFEST.json` |
| `code_reward.py` | Executing reward, batch entry point for verl |
| `code_exec_runner.py` | Per-sample isolated child interpreter |
| `validate_specs.py` | Scores canonical solutions — the spec-correctness check |
| `test_code_reward_on_cpu.py` | 30 harness unit tests |
| `debug_canonical_fails.py` | Lists which canonical solutions fail and why |

Departure from convention: the arms share `_common_code_p4.sh` instead of each
being a self-contained ~230-line copy as in `8_29`/`9_7`/`9_16`. Eight hand-kept
copies cannot be held byte-identical, and `8_29/RESULTS.md` records that the only
effects which survived on math were smaller than the drift one stray config
difference would introduce.

## Eval design and its cost

Validation is wired through verl so `val-core/<data_source>/acc/mean@8` lands in
wandb and the existing `opd_inference/` tooling reads it unchanged. The reward
manager is the default `naive` of `verl/experimental/reward_loop/` (`batch` does
not exist in that registry); it runs the per-sample `compute_score` concurrently
across `reward.num_workers` actors and forwards dict scores into
`reward_extra_info`. See the docstring of `code_reward.py`.

**Eval set: 717 prompts** = HumanEval+ 164 (all) + MBPP+ 378 (all) +
LiveCodeBench 175. The LCB part is the **v6 shard only** (`--lcb-version
v6_only`, i.e. `test6.jsonl`), not the cumulative `release_v6` (~1055 problems).
Contests 2025-01-04 → 2025-04-06; 80 hard / 52 medium / 43 easy; 112 stdin / 63
functional.

**Cost.** 717 × n=8 = **5736 generations per validation**, about 45 training
steps' worth of rollout, against 128 generations per training step. At
`test_freq=50` the run validates at step 0, 50, 100, 150, 200 — **5 points**,
roughly 2.1× the job's total generation cost.

Five points matters because `8_29/RESULTS.md` reports "best @ step", which is not
a meaningful selection over 3. This only became affordable by narrowing LCB to
one shard: the full cumulative `release_v5` suite (879 problems, 1421 total) cost
11368 generations per validation and forced `test_freq=100` / 3 points.

## Caveats that apply to every number this produces

These are also written into `~/data/code_opd/MANIFEST.json`.

1. **Not official benchmark numbers.** This is a self-contained harness for
   comparing the four methods against each other. Do not put these in a table
   next to published HumanEval+/MBPP+/LCB figures.
2. **LCB is the v6 shard only, and it is HARD for these students.** 80 of 175
   problems are rated hard, and the window (Jan-Apr 2025) is chosen to be
   contamination-free rather than easy. A 1.7B base student may score at or near
   zero on it, in which case LCB contributes no signal for separating the four
   methods and only HumanEval+/MBPP+ are informative. Check the step-0
   `val-core/livecodebench/acc/mean@8` from `val_before_train` before reading
   anything into later LCB movement.
3. **LiveCodeBench uses public tests only.** Private tests are
   base64+zlib+**pickle**; decoding them runs arbitrary code from a downloaded
   artifact, so it is gated behind `--lcb-tests all`. Public tests are the
   examples printed in the problem statement, so LCB accuracy here is
   substantially **higher** than official.
4. **LCB capped at 25 test cases per problem.** Truncation can only help a
   solution, so LCB accuracy is an upper bound.
5. **2 of 542 canonical solutions fail** (99.4% HumanEval+, 99.7% MBPP+), measured by
   `validate_specs.py`:
   - `HumanEval/32` (`find_zero`) — upstream EvalPlus harness quirk, its test
     calls `candidate(*inp)` with a float. Not a mapping error; left alone rather
     than special-cased.
   - `mbppplus/599` — exceeds the 15s `RLIMIT_CPU` in the sandbox.

   (The limit was raised from 8s to 15s precisely because at 8s the canonical
   solutions of `HumanEval/139` and `mbppplus/599` were both being SIGKILLed —
   the harness was scoring correct reference code as wrong.)

   This is a ~0.4% underestimate applied identically to all four arms.
6. **`acc` is strict all-tests-pass.** `case_pass_rate` is logged alongside as an
   early-training progress signal and is **not** a benchmark number.
7. **Overlong prompts were dropped at build time**, not at train time, so the
   denominator is fixed across runs. Measured: train max 1858 tokens, eval max
   434 (LCB median is the longest at ~429), against `max_prompt_length=2048` —
   nothing was dropped from eval.
8. **No task reward in training.** `use_task_rewards=False`. reward_loop still
   calls `compute_score` on every training rollout, so `code_r1_12k` samples
   short-circuit to 0 without executing (`OPD_CODE_REWARD_SCORE_TRAIN=1` turns
   execution back on). CodeR1's `functional` specs are stored with
   `entry_point=None` and are unverified; check them before enabling task rewards.

## Reading the results

Before comparing any two arms:

- **Check `actor/distillation/pg_token_ratio`.** For the three hybrid arms this is
  the FKL/PG split and the thing that makes the arm a method rather than a
  baseline repeat. On math, the `opd_theory_guided4` family's best settings were
  the ones that barely routed at all — a near-zero FKL share means the score
  difference is noise whatever its sign. For the `opd` arm this metric reads ~0
  and is meaningless (diagnostic over an unset mask); that arm is 100% PG.
- **There is no code noise floor yet.** The math floor (≈1.6pp mean@8, ≈6pp
  pass@8 at these sizes) does not transfer. Nothing here is interpretable as an
  effect until at least two same-config repeats of one arm have been run. Budget
  for that before drawing conclusions.
- **Watch `[code_reward]` lines in the job log.** One line per benchmark per
  validation with `acc`, `case_pass_rate` and the top error kinds. A benchmark at
  acc 0 whose errors are all `NameError` or `unknown spec type` is a broken spec,
  not a weak student — and the wandb curve alone cannot tell those apart.
