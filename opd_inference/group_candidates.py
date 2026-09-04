"""Group a flat generation jsonl into one record per prompt with K candidates.

Input is the flat jsonl written by run_generate_parquet.py (one line per
(row_idx, sample_idx)). Output is one record per prompt holding all K
candidates, optionally scored with the math_boxed scorer, plus per-prompt
avg accuracy and pass@k.

Writes <out_prefix>.jsonl and <out_prefix>.parquet.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_math_boxed():
    base = Path(__file__).parent.parent / "verl/utils/reward_score"
    spec = importlib.util.spec_from_file_location("math_boxed", base / "math_boxed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _question_of(prompt: Any) -> str:
    """Last user message content from a chat-format prompt."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return str(prompt)


def main() -> None:
    p = argparse.ArgumentParser(description="Group flat generation jsonl into one record per prompt.")
    p.add_argument("--input", required=True, help="Flat results .jsonl from run_generate_parquet.py.")
    p.add_argument(
        "--out_prefix",
        default=None,
        help="Output path prefix (default: <input without .jsonl> + '_grouped').",
    )
    p.add_argument("--k", type=int, default=None, help="Expected candidates per prompt; warns on mismatch.")
    p.add_argument("--score", action="store_true", help="Score each candidate with math_boxed.")
    p.add_argument("--math_verify", action="store_true", help="Use math_verify symbolic fallback when scoring.")
    args = p.parse_args()

    in_path = Path(args.input)
    # NOTE: filenames here contain dots (e.g. "..._t0.6_p0.95_results.jsonl"), so
    # never use Path.with_suffix() on them — it would eat everything after "_p0".
    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
    else:
        out_prefix = in_path.parent / (in_path.name[: -len(".jsonl")] + "_grouped")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    math_boxed = _load_math_boxed() if args.score else None

    # Collect candidates per prompt, preserving first-seen prompt metadata.
    groups: Dict[int, Dict[str, Any]] = {}
    for line in in_path.open(encoding="utf-8"):
        rec = json.loads(line)
        row_idx = rec["row_idx"]
        g = groups.setdefault(
            row_idx,
            {
                "row_idx": row_idx,
                "data_source": rec.get("data_source"),
                "question": _question_of(rec.get("prompt")),
                "prompt": rec.get("prompt"),
                "ground_truth": rec.get("ground_truth"),
                "_cands": [],
            },
        )
        g["_cands"].append(rec)

    records: List[Dict[str, Any]] = []
    per_source = defaultdict(lambda: {"n_prompts": 0, "correct": 0, "total": 0, "solved": 0, "truncated": 0})

    for row_idx in sorted(groups):
        g = groups[row_idx]
        cands = sorted(g.pop("_cands"), key=lambda r: r["sample_idx"])
        if args.k is not None and len(cands) != args.k:
            print(f"[WARN] row_idx={row_idx} has {len(cands)} candidates, expected {args.k}")

        g["k"] = len(cands)
        g["responses"] = [c.get("response", "") for c in cands]
        # finish_reason/num_tokens are only present if generation recorded them.
        if any("finish_reason" in c for c in cands):
            g["finish_reasons"] = [c.get("finish_reason") for c in cands]
            g["num_truncated"] = sum(1 for c in cands if c.get("finish_reason") == "length")
        if any("num_tokens" in c for c in cands):
            g["num_tokens"] = [c.get("num_tokens") for c in cands]

        if math_boxed is not None:
            gt = g.get("ground_truth")
            scores, preds = [], []
            for resp in g["responses"]:
                if not gt:
                    scores.append(0)
                    preds.append(None)
                    continue
                res = math_boxed.compute_score(resp, gt, use_math_verify=args.math_verify)
                scores.append(int(bool(res["acc"])))
                preds.append(res.get("pred"))
            g["scores"] = scores
            g["preds"] = preds
            g["n_correct"] = sum(scores)
            g["avg_acc"] = sum(scores) / len(scores) if scores else 0.0
            g["pass_at_k"] = int(any(scores))

        src = g["data_source"]
        s = per_source[src]
        s["n_prompts"] += 1
        s["total"] += g["k"]
        s["truncated"] += g.get("num_truncated", 0)
        if math_boxed is not None:
            s["correct"] += g["n_correct"]
            s["solved"] += g["pass_at_k"]

        records.append(g)

    jsonl_path = Path(str(out_prefix) + ".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    parquet_path = Path(str(out_prefix) + ".parquet")
    try:
        import pandas as pd

        pd.DataFrame(records).to_parquet(parquet_path, index=False)
    except Exception as e:  # pragma: no cover - parquet is a convenience artifact
        parquet_path = None
        print(f"[WARN] parquet write skipped: {e}")

    print(f"\n[OK] {len(records)} prompts → {jsonl_path}")
    if parquet_path:
        print(f"[OK] {len(records)} prompts → {parquet_path}")

    header = f"\n{'Source':<16} {'Prompts':>8} {'Cands':>8} {'Trunc':>8}"
    if math_boxed is not None:
        header += f" {'avg@k':>8} {'pass@k':>8}"
    print(header)
    print("-" * len(header.strip("\n")))
    tot = {"n_prompts": 0, "total": 0, "correct": 0, "solved": 0, "truncated": 0}
    for src in sorted(per_source):
        s = per_source[src]
        for key in tot:
            tot[key] += s[key]
        line = f"{src:<16} {s['n_prompts']:>8} {s['total']:>8} {s['truncated']:>8}"
        if math_boxed is not None:
            line += f" {s['correct'] / s['total']:>7.1%} {s['solved'] / s['n_prompts']:>7.1%}"
        print(line)
    print("-" * len(header.strip("\n")))
    line = f"{'Overall':<16} {tot['n_prompts']:>8} {tot['total']:>8} {tot['truncated']:>8}"
    if math_boxed is not None:
        line += f" {tot['correct'] / tot['total']:>7.1%} {tot['solved'] / tot['n_prompts']:>7.1%}"
    print(line)


if __name__ == "__main__":
    main()
