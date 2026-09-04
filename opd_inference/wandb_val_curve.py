"""Print the step-by-step validation curve for one or more wandb runs.

Answers "is the final number a trend or a lucky last eval?", which the summary alone cannot.

Usage:
    python opd_inference/wandb_val_curve.py zo9tt461 2q7t70uu v01gyicv
"""

import sys

import wandb

DATASETS = ["aime", "aime25", "aime26", "amc23"]
ENTITY_PROJECT = "rl_agent/verl_opd_dapo"

MEAN_KEYS = [f"val-core/{d}/acc/mean@8" for d in DATASETS]
PASS_KEYS = [f"val-core/{d}/acc/best@8/mean" for d in DATASETS]
EXTRA_KEYS = ["actor/distillation/pg_token_ratio", "response_length/mean"]


def main():
    ids = sys.argv[1:]
    if not ids:
        print(__doc__)
        return

    api = wandb.Api()
    for rid in ids:
        run = api.run(f"{ENTITY_PROJECT}/{rid}")
        print(f"=== {rid}  {run.name}")
        print(f"{'step':>5} {'avg4 mean@8':>12} {'avg4 pass@8':>12} {'pg%':>7} {'reslen':>8}")

        # Do NOT pass keys= to scan_history. The backend rejects it for some runs with
        # "error scanning step range: Step column '_step' not found in schema", and even when it
        # works it drops rows missing any requested key -- while a single step's metrics can be
        # split across several rows. Scan everything and merge per step instead.
        merged = {}
        for row in run.scan_history(page_size=500):
            step = row.get("_step")
            if step is None:
                continue
            slot = merged.setdefault(step, {})
            for k, v in row.items():
                if v is not None:
                    slot.setdefault(k, v)

        for step in sorted(merged):
            row = merged[step]
            means = [row.get(k) for k in MEAN_KEYS]
            passes = [row.get(k) for k in PASS_KEYS]
            if any(v is None for v in means):
                continue
            pg = row.get("actor/distillation/pg_token_ratio")
            reslen = row.get("response_length/mean")
            avg_m = sum(means) / 4 * 100
            avg_p = sum(v for v in passes if v is not None) / max(
                1, sum(1 for v in passes if v is not None)
            ) * 100
            pg_s = f"{pg * 100:.1f}" if pg is not None else "-"
            rl_s = f"{reslen:.0f}" if reslen is not None else "-"
            print(f"{step:>5} {avg_m:>12.2f} {avg_p:>12.2f} {pg_s:>7} {rl_s:>8}")
        print()


if __name__ == "__main__":
    main()
