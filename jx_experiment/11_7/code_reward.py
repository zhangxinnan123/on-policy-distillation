#!/usr/bin/env python3
"""Executing reward function for HumanEval+, MBPP+ and LiveCodeBench validation.

Wired in via::

    reward.custom_reward_function.path=<repo>/jx_experiment/11_7/code_reward.py
    reward.custom_reward_function.name=compute_score

and nothing else -- the default `reward.reward_manager.name=naive` is correct.

WHICH REWARD MANAGER, AND WHY IT IS NOT `batch`
-----------------------------------------------
These runs set `trainer.use_legacy_worker_impl=disable`, which routes reward
computation through `verl/experimental/reward_loop/`. That subsystem has its OWN
manager registry (`verl/experimental/reward_loop/reward_manager/registry.py`),
holding only naive / dapo / gdpo / remote / visual / rate_limited. `batch` exists
only in the legacy registry under `verl/workers/reward_manager/`, so asking for
it there fails at startup with `ValueError: Unknown reward manager: batch`
(observed, job 14359).

That turned out to be the better path anyway. reward_loop's `naive` is not the
serial per-sample loop the legacy `naive` is:

  * `run_single` is async and a *sync* compute_score is dispatched with
    `loop.run_in_executor(None, ...)`, so many samples are in flight at once
    inside one worker;
  * `reward.num_workers` (8) RewardLoopWorker Ray actors each run their own
    event loop, so the batch is sharded 8 ways on top of that;
  * a dict result has every key copied into `reward_extra_info`, which is what
    makes `acc` reach `val-core/<data_source>/acc/mean@8` -- the key every table
    in `8_29/RESULTS.md` and `opd_inference/dump_wandb_results.py` reads.

So the entry point below is PER SAMPLE, and the concurrency this file has to
manage is its own: without a bound, 8 actors x the default executor's 32 threads
would put 256 compiler-grade subprocesses on a 96-vCPU node. See _SEM.

Ground truth travels as a JSON string in `reward_model.ground_truth`, tagged with
a `type` that selects the harness. See `prepare_code_data.py`.

Scoring is strict pass/fail per sample (`acc` in {0, 1}), which is what pass@k
and mean@k are defined over. The fraction of individual test cases passed is
reported separately as `case_pass_rate`: a useful progress signal early in
training, when almost nothing fully passes, but NOT the benchmark number.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_RUNNER = str(Path(__file__).resolve().parent / "code_exec_runner.py")

# Per-sample wall-clock cap. Deliberately ABOVE the 15s RLIMIT_CPU the child sets
# for itself: a child that busy-loops is then stopped by its own CPU limit, which
# still lets the parent read a verdict, and only a child that blocks on I/O or
# forks reaches this outer kill, which loses the verdict.
_TIMEOUT_S = float(os.environ.get("OPD_CODE_REWARD_TIMEOUT", "25"))

# Concurrent child interpreters allowed IN THIS PROCESS.
#
# The trainer runs reward.num_workers=8 RewardLoopWorker actors, each a separate
# OS process, each dispatching sync compute_score calls onto a default thread
# executor of min(32, cpu+4) threads. Unbounded that is up to 256 concurrent
# subprocesses against 96 vCPUs, which starves the Ray driver and the resident
# vLLM engines. 6 per process x 8 actors = 48, half the node.
_PER_PROC = int(os.environ.get("OPD_CODE_REWARD_CONCURRENCY", "6"))
_SEM = threading.Semaphore(_PER_PROC)

# Periodic aggregate logging. One line per sample would be ~5.7k lines per
# validation; one line per _LOG_EVERY samples per process keeps the systematic
# failure modes visible without drowning the log.
_LOG_EVERY = int(os.environ.get("OPD_CODE_REWARD_LOG_EVERY", "250"))
_stats_lock = threading.Lock()
_stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "acc": 0.0, "cases": 0.0, "errs": defaultdict(int)})

_FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(completion: str) -> str:
    """Pull the program out of a chat-style completion.

    Takes the LAST fenced block, not the first: these models routinely sketch a
    partial approach in an early block and emit the real answer last, so taking
    the first block scores a draft. An unterminated trailing fence is accepted
    because a response truncated at max_response_length mid-block is common at
    4k and its code up to the cut is still worth executing.
    """
    blocks = _FENCE.findall(completion)
    if blocks:
        return blocks[-1]
    idx = max(
        completion.rfind("```python"),
        completion.rfind("```py"),
        completion.rfind("```"),
    )
    if idx != -1:
        tail = completion[idx:]
        nl = tail.find("\n")
        if nl != -1:
            return tail[nl + 1 :]
    # No fence at all -- treat the whole completion as code, which scores the
    # occasional bare-code response instead of zeroing it.
    return completion


def _run(completion: str, ground_truth) -> dict:
    """Execute one completion against one spec in a fresh isolated interpreter."""
    out = {"score": 0.0, "acc": 0.0, "case_pass_rate": 0.0, "err": ""}

    try:
        spec = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (TypeError, ValueError) as exc:
        out["err"] = f"bad ground_truth: {exc}"
        return out
    if not spec:
        out["err"] = "empty spec"
        return out

    code = extract_code(completion or "")
    if not code.strip():
        out["err"] = "empty code"
        return out

    payload = json.dumps({"code": code, "spec": spec})
    with _SEM:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", _RUNNER],
                input=payload,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
                # Somewhere the generated program cannot reach the repo or the
                # datasets by a relative path.
                cwd="/tmp",
            )
        except subprocess.TimeoutExpired:
            out["err"] = "timeout"
            return out
        except Exception as exc:  # noqa: BLE001
            out["err"] = f"spawn failed: {type(exc).__name__}: {exc}"
            return out

    line = proc.stdout.strip().split("\n")[-1] if proc.stdout.strip() else ""
    if not line:
        # Killed by SIGXCPU / SIGSEGV / OOM before it could report.
        out["err"] = f"no verdict (rc={proc.returncode})"
        return out
    try:
        verdict = json.loads(line)
    except ValueError:
        out["err"] = "unparseable verdict"
        return out

    n_total = int(verdict.get("n_total") or 0)
    n_pass = int(verdict.get("n_pass") or 0)
    passed = bool(verdict.get("passed"))
    out["score"] = 1.0 if passed else 0.0
    out["acc"] = out["score"]
    out["case_pass_rate"] = (n_pass / n_total) if n_total else 0.0
    out["err"] = str(verdict.get("error") or "")[:200]
    return out


def _record(data_source: str, res: dict) -> None:
    """Accumulate and periodically print a per-benchmark summary.

    This is the only place a systematic harness failure is visible. A benchmark
    sitting at acc 0 whose errors are all `NameError` or `unknown spec type` is a
    broken spec, not a weak student, and the wandb curve alone cannot tell those
    two apart.
    """
    with _stats_lock:
        st = _stats[str(data_source)]
        st["n"] += 1
        st["acc"] += res["acc"]
        st["cases"] += res["case_pass_rate"]
        if res["acc"] == 0.0 and res["err"]:
            st["errs"][res["err"].split(":")[0][:40]] += 1
        if st["n"] % _LOG_EVERY == 0:
            top = sorted(st["errs"].items(), key=lambda kv: -kv[1])[:4]
            print(
                f"[code_reward] {data_source}: n={st['n']} "
                f"acc={st['acc'] / st['n']:.4f} "
                f"case_pass_rate={st['cases'] / st['n']:.4f} top_errors={top}",
                flush=True,
            )


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
) -> dict:
    """Per-sample entry point for reward_loop's NaiveRewardManager.

    Returns a dict so every key lands in `reward_extra_info`; `score` drives the
    reward tensor and `acc` is what `_val_metrics_update` promotes to
    `val-core/<ds>/acc/...` (it prefers a variable literally named `acc` over
    `reward`).

    `err` is deliberately stripped: reward_extra_info values are aggregated
    numerically for logging, and a string column there breaks
    process_validation_metrics.
    """
    res = _run(solution_str or "", ground_truth)
    _record(data_source, res)
    return {k: v for k, v in res.items() if k != "err"}


def compute_score_batch(
    data_sources: list,
    solution_strs: list,
    ground_truths: list,
    extra_infos=None,
    **kwargs,
) -> list[dict]:
    """Whole-batch entry point for the LEGACY `batch` reward manager.

    Only reachable with `trainer.use_legacy_worker_impl` left enabled; the 11_7
    arms do not use this path. Kept because it is the natural way to score a
    dumped set of generations offline, where there is no reward_loop to provide
    concurrency and this file must supply its own.
    """
    pairs = list(zip(solution_strs, ground_truths))
    if not pairs:
        return []
    with ThreadPoolExecutor(max_workers=min(48, len(pairs))) as pool:
        results = list(pool.map(lambda p: _run(p[0], p[1]), pairs))
    for src, res in zip(data_sources, results):
        _record(src, res)
    return [{k: v for k, v in r.items() if k != "err"} for r in results]


if __name__ == "__main__":
    spec = json.dumps(
        {
            "type": "evalplus",
            "entry_point": "add",
            "test_code": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        }
    )
    print(compute_score("humanevalplus", "```python\ndef add(a, b):\n    return a + b\n```", spec))
