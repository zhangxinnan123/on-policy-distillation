#!/usr/bin/env python3
"""Validate the built eval parquet by scoring the datasets' OWN reference solutions.

    conda activate verl
    python3 jx_experiment/11_7/validate_specs.py --data ~/data/code_opd

This is the check that catches a wrong field mapping. A model scoring 0 on a
benchmark is ambiguous -- weak student or broken spec -- but a *canonical*
solution scoring 0 can only mean the spec is broken. Anything below ~98% here
means do not trust that benchmark's numbers yet.

LiveCodeBench ships no reference solutions in the lite release, so it gets a
structural check instead: specs well-formed, and the stdio/functional split
matching what the records declare.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from code_reward import compute_score_batch as compute_score  # noqa: E402

FENCE_OPEN = "```python\n"
FENCE_CLOSE = "\n```"


def fence(code: str) -> str:
    return FENCE_OPEN + code + FENCE_CLOSE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/data/code_opd")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset

    data = Path(args.data).expanduser()
    test = pd.read_parquet(data / "test.parquet")
    gt = {
        (r["data_source"], str(r["extra_info"]["task_id"])): r["reward_model"]["ground_truth"]
        for _, r in test.iterrows()
    }

    rc = 0

    # ---------------------------------------------------------------- HumanEval+
    h = list(load_dataset("evalplus/humanevalplus", split="test"))
    if args.limit:
        h = h[: args.limit]
        h = [x for x in h if ("humanevalplus", str(x["task_id"])) in gt]
    comps = [fence(x["prompt"] + x["canonical_solution"]) for x in h]
    gts = [gt[("humanevalplus", str(x["task_id"]))] for x in h]
    res = compute_score(["humanevalplus"] * len(gts), comps, gts)
    rate = sum(r["acc"] for r in res) / max(1, len(res))
    print(f"\nHumanEval+ canonical pass rate: {rate:.4f}  (n={len(res)})")
    if rate < 0.98:
        bad = [x["task_id"] for x, r in zip(h, res) if r["acc"] == 0.0][:10]
        print(f"  FAIL -- spec is wrong. first failing task_ids: {bad}")
        rc = 1

    # -------------------------------------------------------------------- MBPP+
    m = list(load_dataset("evalplus/mbppplus", split="test"))
    m = [x for x in m if ("mbppplus", str(x["task_id"])) in gt]
    if args.limit:
        m = m[: args.limit]
    comps = [fence(x["code"]) for x in m]
    gts = [gt[("mbppplus", str(x["task_id"]))] for x in m]
    res = compute_score(["mbppplus"] * len(gts), comps, gts)
    rate = sum(r["acc"] for r in res) / max(1, len(res))
    print(f"\nMBPP+ canonical pass rate: {rate:.4f}  (n={len(res)})")
    if rate < 0.98:
        bad = [x["task_id"] for x, r in zip(m, res) if r["acc"] == 0.0][:10]
        print(f"  FAIL -- spec is wrong. first failing task_ids: {bad}")
        rc = 1

    # ------------------------------------------------------------ LiveCodeBench
    lcb = test[test["data_source"] == "livecodebench"]
    kinds: dict[str, int] = {}
    n_empty = 0
    for _, r in lcb.iterrows():
        spec = json.loads(r["reward_model"]["ground_truth"])
        kinds[spec["type"]] = kinds.get(spec["type"], 0) + 1
        if not spec.get("inputs") or not spec.get("outputs"):
            n_empty += 1
        elif len(spec["inputs"]) != len(spec["outputs"]):
            n_empty += 1
    print(f"\nLiveCodeBench structural check: n={len(lcb)} kinds={kinds} malformed={n_empty}")
    if n_empty:
        print("  FAIL -- specs with no/mismatched test cases")
        rc = 1
    declared = set(lcb["extra_info"].apply(lambda e: e.get("testtype")))
    if not declared <= {"stdio", "functional"}:
        print(f"  FAIL -- unexpected testtype values: {declared}")
        rc = 1

    # A single hand-written LCB probe: the harness must be able to pass at least
    # one real stdio problem, otherwise the stdin plumbing is broken for the
    # whole benchmark and every LCB number would be a flat zero.
    probe = None
    for _, r in lcb.iterrows():
        spec = json.loads(r["reward_model"]["ground_truth"])
        if spec["type"] == "stdio":
            probe = (r, spec)
            break
    if probe is not None:
        _, spec = probe
        # Echo back the expected output for case 0 -- verifies stdin is wired,
        # stdout is captured and comparison works on real LCB data.
        lit = repr(spec["outputs"][0])
        res = compute_score(
            ["livecodebench"],
            [fence(f"import sys\n_ = sys.stdin.read()\nsys.stdout.write({lit})")],
            [json.dumps({**spec, "inputs": spec["inputs"][:1], "outputs": spec["outputs"][:1]})],
        )
        ok = res[0]["acc"] == 1.0
        print(f"LiveCodeBench stdio plumbing probe: {'ok' if ok else 'FAIL'}")
        if not ok:
            rc = 1

    print("\nOK" if rc == 0 else "\nFAILURES -- do not run the sweep yet")
    return rc


if __name__ == "__main__":
    sys.exit(main())
