"""Inventory offline wandb run dirs: display name, project, final step, val presence.

Used to decide which offline runs are worth syncing online. Failed launches (OOM,
EngineDead, config error) still leave a wandb dir behind, but they never reach a training
step, so a missing/zero `_step` is a reliable "this run is garbage" signal.

The step count comes from `files/wandb-summary.json`, NOT from scanning the binary
`.wandb` datastore. Scanning is unreliable: `scan_record()` stops early on some files, so a
run with 200 real steps can read back as `last_step=None`. The summary file is written by
the run itself and always reflects the true final step.

Usage:
    python opd_inference/list_offline_wandb_runs.py <wandb_dir> [--since 20260829]
"""

import argparse
import glob
import json
import os

VAL_KEY_HINT = "val-core/"


def read_summary(run_dir):
    """Return (last_step, has_val) from files/wandb-summary.json."""
    p = os.path.join(run_dir, "files", "wandb-summary.json")
    if not os.path.exists(p):
        return None, False
    try:
        d = json.load(open(p))
    except Exception:
        return None, False
    step = d.get("_step")
    has_val = any(str(k).startswith(VAL_KEY_HINT) for k in d)
    return step, has_val


def datastore_mb(run_dir):
    """Size of the run's .wandb datastore in MB (0.0 if absent)."""
    files = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not files:
        return 0.0
    return max(os.path.getsize(f) for f in files) / 1e6


def read_meta(run_dir):
    """Return (display_name, project) from files/config.yaml + wandb-metadata.json."""
    name = project = None
    p = os.path.join(run_dir, "files", "wandb-metadata.json")
    if os.path.exists(p):
        try:
            name = json.load(open(p)).get("name")
        except Exception:
            pass
    # project only lives in the datastore header or config.yaml's wandb block; the sync
    # command re-derives it anyway, so a miss here is harmless.
    p = os.path.join(run_dir, "files", "config.yaml")
    if os.path.exists(p):
        try:
            for line in open(p):
                if "project_name" in line:
                    project = line.split(":", 1)[1].strip().strip("'\"")
                    break
        except Exception:
            pass
    return name, project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wandb_dir")
    ap.add_argument("--since", default="", help="keep dirs whose date part is >= this (YYYYMMDD)")
    ap.add_argument("--min_step", type=int, default=1, help="a run is real if _step >= this")
    ap.add_argument(
        "--inspect_mb",
        type=float,
        default=5.0,
        help="a summary-less run with a .wandb datastore at least this big is flagged for "
        "inspection rather than treated as a failed launch",
    )
    ap.add_argument("--json_out", default="")
    ap.add_argument("--print_good_dirs", action="store_true", help="print only kept dir names")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.wandb_dir, "offline-run-*")))
    if args.since:
        dirs = [d for d in dirs if os.path.basename(d).split("-")[2][:8] >= args.since]

    rows = []
    for d in dirs:
        step, has_val = read_summary(d)
        name, project = read_meta(d)
        mb = datastore_mb(d)
        try:
            step_i = int(float(step)) if step is not None else -1
        except (TypeError, ValueError):
            step_i = -1
        # A run can finish without wandb ever writing files/: job 15811 (1.7B aopd) completed
        # 100 steps over 11h and left only requirements.txt plus a 20 MB .wandb datastore, so
        # summary-based triage called it a failed launch. Treat a fat datastore with no summary
        # as "inspect" -- check `sacct -j <job>` -- never as junk.
        inspect = step_i < args.min_step and mb >= args.inspect_mb
        rows.append(
            {
                "dir": os.path.basename(d),
                "id": os.path.basename(d).rsplit("-", 1)[-1],
                "name": name,
                "project": project,
                "last_step": step_i,
                "has_val": has_val,
                "wandb_mb": round(mb, 1),
                "inspect": inspect,
                "keep": step_i >= args.min_step or has_val or inspect,
            }
        )

    if args.print_good_dirs:
        for r in rows:
            if r["keep"]:
                print(r["dir"])
        return

    kept = [r for r in rows if r["keep"]]
    flagged = [r for r in rows if r["inspect"]]
    print(
        f"total={len(rows)}  keep={len(kept)}  drop={len(rows) - len(kept)}  "
        f"inspect={len(flagged)}"
    )
    print()
    print(f"{'step':>5} {'val':>4} {'MB':>7} {'keep':>5} {'insp':>5}  {'id':<10} name")
    for r in sorted(rows, key=lambda r: (-r["last_step"], r["dir"])):
        print(
            f"{r['last_step']:>5} {str(r['has_val']):>4} {r['wandb_mb']:>7} "
            f"{str(r['keep']):>5} {str(r['inspect']):>5}  {r['id']:<10} {r['name']}"
        )
    if flagged:
        print("\nflagged for inspection (summary missing but datastore is large) -- run")
        print("`sacct -j <jobid>` for these before dismissing them:")
        for r in flagged:
            print(f"  {r['id']}  {r['wandb_mb']} MB  {r['dir']}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
