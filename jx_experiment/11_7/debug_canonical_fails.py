#!/usr/bin/env python3
"""Print which canonical solutions fail and why. Diagnostic for validate_specs.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import code_reward  # noqa: E402

FENCE_OPEN = "```python\n"


def main() -> int:
    import pandas as pd
    from datasets import load_dataset

    data = Path(sys.argv[1] if len(sys.argv) > 1 else "~/data/code_opd").expanduser()
    test = pd.read_parquet(data / "test.parquet")
    gt = {
        (r["data_source"], str(r["extra_info"]["task_id"])): r["reward_model"]["ground_truth"]
        for _, r in test.iterrows()
    }

    jobs = []
    for x in load_dataset("evalplus/humanevalplus", split="test"):
        k = ("humanevalplus", str(x["task_id"]))
        if k in gt:
            jobs.append((k, FENCE_OPEN + x["prompt"] + x["canonical_solution"] + "\n```", gt[k]))
    for x in load_dataset("evalplus/mbppplus", split="test"):
        k = ("mbppplus", str(x["task_id"]))
        if k in gt:
            jobs.append((k, FENCE_OPEN + x["code"] + "\n```", gt[k]))

    # _run, not compute_score_batch: the public entry points strip `err` (a
    # string column breaks reward_extra_info aggregation) and `err` is the whole
    # point of this script.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda j: code_reward._run(j[1], j[2]), jobs))
    for (k, _, g), r in zip(jobs, results):
        if r["acc"] == 0.0:
            spec = json.loads(g)
            print(f"{k}  err={r['err']!r}  entry={spec.get('entry_point')}")
    print(f"\n{sum(1 for r in results if r['acc'] == 0.0)} / {len(results)} canonical failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
