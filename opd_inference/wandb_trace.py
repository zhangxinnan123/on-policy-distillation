"""Print a per-step trace of chosen scalar metrics (not just validation steps).

Validation only runs every test_freq steps, so val-based tools cannot show what happened in
the step where a job died. This walks every history row.

Usage:
    python opd_inference/wandb_trace.py <run_id> [--last N] [--keys k1,k2]
"""

import argparse

import wandb

ENTITY_PROJECT = "rl_agent/verl_opd_dapo"
DEFAULT_KEYS = [
    "actor/distillation/pg_token_ratio",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "actor/entropy",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--last", type=int, default=12)
    ap.add_argument("--keys", default="")
    args = ap.parse_args()

    keys = args.keys.split(",") if args.keys else DEFAULT_KEYS

    api = wandb.Api()
    run = api.run(f"{ENTITY_PROJECT}/{args.run_id}")
    print(f"=== {args.run_id}  {run.name}")

    # scan_history without keys= : the keys= form is rejected by the backend for some runs
    # ("Step column '_step' not found in schema") and drops rows missing any requested key.
    merged = {}
    for row in run.scan_history(page_size=500):
        step = row.get("_step")
        if step is None:
            continue
        slot = merged.setdefault(step, {})
        for k, v in row.items():
            if v is not None:
                slot.setdefault(k, v)

    steps = sorted(merged)
    short = [k.split("/")[-1][:9] for k in keys]
    print("step  " + " ".join(f"{s:>10}" for s in short))
    for step in steps[-args.last :]:
        row = merged[step]
        cells = []
        for k in keys:
            v = row.get(k)
            cells.append("-" if v is None else (f"{v:10.4f}" if isinstance(v, float) else f"{v:>10}"))
        print(f"{step:>4}  " + " ".join(cells))
    print(f"\n(total steps logged: {len(steps)}, last = {steps[-1] if steps else 'none'})")


if __name__ == "__main__":
    main()
