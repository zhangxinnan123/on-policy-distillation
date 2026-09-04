"""Dump final eval metrics for wandb runs so RESULTS.md can be updated from one source.

mean@8 = val-core/<ds>/acc/mean@8 ; pass@8 = val-core/<ds>/acc/best@8/mean.

Usage:
    python opd_inference/dump_wandb_results.py --entity rl_agent --project verl_opd_dapo \
        --since 2026-08-29
"""

import argparse
from datetime import datetime

DATASETS = ["aime", "aime25", "aime26", "amc23"]


def pct(v):
    return None if v is None else round(float(v) * 100, 2)


def main():
    import wandb

    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="rl_agent")
    ap.add_argument("--project", default="verl_opd_dapo")
    ap.add_argument("--since", default="", help="ISO date; keep runs created at/after this")
    ap.add_argument("--name_filter", default="", help="substring that must appear in run name")
    args = ap.parse_args()

    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))

    rows = []
    for r in runs:
        created = getattr(r, "created_at", "") or ""
        if args.since and created[:10] < args.since:
            continue
        if args.name_filter and args.name_filter not in (r.name or ""):
            continue
        s = r.summary
        rec = {
            "id": r.id,
            "name": r.name,
            "created": created[:16],
            "state": r.state,
            "step": s.get("_step"),
            "pg": pct(s.get("actor/distillation/pg_token_ratio")),
            "reslen": s.get("response_length/mean"),
            "val_reslen": s.get("val/response_length/mean"),
        }
        vals = []
        for ds in DATASETS:
            m = pct(s.get(f"val-core/{ds}/acc/mean@8"))
            p = pct(s.get(f"val-core/{ds}/acc/best@8/mean"))
            rec[ds] = (m, p)
            if m is not None:
                vals.append((m, p))
        if len(vals) == 4:
            rec["avg_m"] = round(sum(v[0] for v in vals) / 4, 2)
            rec["avg_p"] = round(sum(v[1] for v in vals) / 4, 2)
        else:
            rec["avg_m"] = rec["avg_p"] = None
        rows.append(rec)

    rows.sort(key=lambda r: (r["avg_m"] is None, -(r["avg_m"] or 0)))
    for r in rows:
        head = (
            f"{r['created']}  {r['id']:<12} step={str(r['step']):>4} "
            f"pg={str(r['pg']):>6} reslen={str(r['reslen'])[:7]:>7} state={r['state']}"
        )
        print(head)
        print(f"    {r['name']}")
        if r["avg_m"] is None:
            print("    (incomplete val)")
        else:
            cells = " | ".join(
                f"{ds}: {r[ds][0]:.2f}/{r[ds][1]:.2f}" for ds in DATASETS
            )
            print(f"    {cells}  ||  avg4: {r['avg_m']:.2f}/{r['avg_p']:.2f}")
        print()

    print(f"total runs listed: {len(rows)}   ({datetime.now().isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
