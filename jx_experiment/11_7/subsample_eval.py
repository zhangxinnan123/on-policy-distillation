#!/usr/bin/env python3
"""Write a smaller eval parquet: a fixed, stratified subsample of test.parquet.

    python3 jx_experiment/11_7/subsample_eval.py --data ~/data/code_opd --n 100
    -> <data>/test_100.parquet  (+ test_100.json listing the kept task_ids)

Stratified by data_source, proportional to each benchmark's size (largest
remainder), with a fixed seed so every arm validates on the SAME prompts.
For 100 of 717: humanevalplus 23, mbppplus 53, livecodebench 24.

Smaller n makes per-benchmark numbers noisier (24 LCB prompts x 8 samples);
compare arms on this set only with each other, never with full-set numbers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/data/code_opd"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default="test.parquet", help="eval parquet inside --data to subsample")
    args = ap.parse_args()

    src = Path(args.data) / args.src
    df = pd.read_parquet(src)
    sizes = df.data_source.value_counts()
    exact = sizes / sizes.sum() * args.n
    quota = exact.astype(int)
    for ds in (exact - quota).sort_values(ascending=False).index[: args.n - quota.sum()]:
        quota[ds] += 1

    parts = [
        df[df.data_source == ds].sample(n=int(k), random_state=args.seed).sort_index()
        for ds, k in quota.items()
    ]
    out = pd.concat(parts).reset_index(drop=True)
    for i in range(len(out)):
        out.at[i, "extra_info"] = {**out.at[i, "extra_info"], "index": i}

    stem = Path(args.src).stem
    dst = Path(args.data) / f"{stem}_{args.n}.parquet"
    out.to_parquet(dst)
    ids = {ds: [str(e["task_id"]) for e in out[out.data_source == ds].extra_info] for ds in quota.index}
    (Path(args.data) / f"{stem}_{args.n}.json").write_text(
        json.dumps({"source": str(src), "seed": args.seed, "counts": quota.to_dict(), "task_ids": ids}, indent=1)
    )
    print(f"wrote {dst}  n={len(out)}  {quota.to_dict()}")


if __name__ == "__main__":
    main()
