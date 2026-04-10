"""Evaluate a results jsonl file using math_boxed scorer."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to results .jsonl file")
    p.add_argument("--math_verify", action="store_true", help="Use math_verify as fallback for symbolic equivalence.")
    args = p.parse_args()

    import importlib.util, sys
    base = Path(__file__).parent.parent / "verl/utils/reward_score"
    spec = importlib.util.spec_from_file_location("math_boxed", base / "math_boxed.py")
    math_boxed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_boxed)

    per_source = defaultdict(lambda: {"correct": 0, "total": 0})
    overall = {"correct": 0, "total": 0}

    with open(args.input) as f:
        for line in f:
            record = json.loads(line)
            gt = record.get("ground_truth")
            response = record.get("response", "")
            data_source = record.get("data_source", "unknown")
            if not gt:
                continue

            res = math_boxed.compute_score(response, gt, use_math_verify=args.math_verify)
            acc = res["acc"]
            per_source[data_source]["correct"] += int(acc)
            per_source[data_source]["total"] += 1
            overall["correct"] += int(acc)
            overall["total"] += 1

    print(f"\n{'Source':<15} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 45)
    for src, stats in sorted(per_source.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"{src:<15} {stats['correct']:>8} {stats['total']:>8} {acc:>8.1%}")
    print("-" * 45)
    acc = overall["correct"] / overall["total"] if overall["total"] else 0
    print(f"{'Overall':<15} {overall['correct']:>8} {overall['total']:>8} {acc:>8.1%}")


if __name__ == "__main__":
    main()
