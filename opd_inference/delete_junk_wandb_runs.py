"""Delete specific online wandb runs by id.

Used to clean up empty runs that were uploaded before the "did it actually train?" filter
was applied (see sync_good_wandb_runs.sh). Run ids are passed explicitly — no globbing, no
pattern matching — so nothing can be deleted by accident.

Prints each run's step/history size before deleting, and refuses any run that looks like it
has real data unless --force is given.

Usage:
    python opd_inference/delete_junk_wandb_runs.py --entity rl_agent \
        --ids dedoz8zq,m0e9df26 --inventory /fsx/xinnanzh/logs/wandb_inventory.json [--apply]
"""

import argparse
import json

import wandb

MAX_SAFE_STEP = 20  # anything at or beyond this is treated as real data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated run ids")
    ap.add_argument("--inventory", required=True, help="json from list_offline_wandb_runs.py")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--force", action="store_true", help="delete even if the run has real data")
    args = ap.parse_args()

    want = [i.strip() for i in args.ids.split(",") if i.strip()]
    inv = {r["dir"].rsplit("-", 1)[-1]: r for r in json.load(open(args.inventory))}

    api = wandb.Api()
    deleted, skipped, missing = 0, 0, 0

    for rid in want:
        meta = inv.get(rid)
        if meta is None:
            print(f"{rid}: not in inventory, skipping")
            missing += 1
            continue
        project = meta["project"]
        path = f"{args.entity}/{project}/{rid}"
        try:
            run = api.run(path)
        except Exception as e:
            print(f"{rid}: not found online ({type(e).__name__}), skipping")
            missing += 1
            continue

        step = run.summary.get("_step")
        n_hist = getattr(run, "lastHistoryStep", None)
        looks_real = (meta.get("val_pts", 0) > 0) or (
            isinstance(step, (int, float)) and step >= MAX_SAFE_STEP
        )
        tag = "REAL DATA" if looks_real else "empty"
        print(f"{rid}: {tag}  _step={step} lastHistoryStep={n_hist} project={project}")
        print(f"    {run.name}")

        if looks_real and not args.force:
            print("    -> refusing (use --force to override)")
            skipped += 1
            continue
        if not args.apply:
            print("    -> would delete (dry run)")
            continue
        run.delete()
        print("    -> DELETED")
        deleted += 1

    print(f"\ndeleted={deleted} skipped={skipped} missing={missing} requested={len(want)}")


if __name__ == "__main__":
    main()
