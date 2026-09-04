"""Dump the full per-dataset validation record at one specific step of a run.

The summary only holds the final step, and dump_wandb_results.py reports that; this walks the
history to a chosen step so an intermediate checkpoint can be quoted per-dataset.

Usage:
    python opd_inference/wandb_step_detail.py <run_id> <step> [<run_id> <step> ...]
"""

import sys

import wandb

DATASETS = ["aime", "aime25", "aime26", "amc23"]
ENTITY_PROJECT = "rl_agent/verl_opd_dapo"

EXTRA = [
    "actor/distillation/pg_token_ratio",
    "response_length/mean",
    "response_length/clip_ratio",
    "response/complete_ratio_non_aborted",
    "val/response_length/mean",
    "actor/entropy",
]


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        print(__doc__)
        return

    api = wandb.Api()
    for rid, step_s in zip(args[::2], args[1::2]):
        want = int(step_s)
        run = api.run(f"{ENTITY_PROJECT}/{rid}")
        keys = ["_step"]
        for d in DATASETS:
            keys += [f"val-core/{d}/acc/mean@8", f"val-core/{d}/acc/best@8/mean"]
        keys += EXTRA

        # Do NOT pass keys= to scan_history: it silently drops rows missing any requested key,
        # and a single step's metrics can be split across several history rows. Merge instead.
        hit = {}
        for row in run.scan_history(page_size=500):
            if row.get("_step") != want:
                continue
            for k, v in row.items():
                if v is not None:
                    hit.setdefault(k, v)

        print(f"=== {rid} @ step {want}   {run.name}")
        if not any(k.startswith("val-core/") for k in hit):
            print("   no validation record at that step")
            continue

        means, passes = [], []
        for d in DATASETS:
            m = hit.get(f"val-core/{d}/acc/mean@8")
            p = hit.get(f"val-core/{d}/acc/best@8/mean")
            if m is None:
                print(f"   {d:<7} (missing)")
                continue
            m *= 100
            p = p * 100 if p is not None else float("nan")
            means.append(m)
            passes.append(p)
            print(f"   {d:<7} {m:6.2f} / {p:6.2f}")
        if len(means) == len(DATASETS):
            print(f"   {'avg 4':<7} {sum(means) / 4:6.2f} / {sum(passes) / 4:6.2f}")
        for k in EXTRA:
            v = hit.get(k)
            if v is not None:
                print(f"   {k} = {v:.4f}" if isinstance(v, float) else f"   {k} = {v}")
        print()


if __name__ == "__main__":
    main()
