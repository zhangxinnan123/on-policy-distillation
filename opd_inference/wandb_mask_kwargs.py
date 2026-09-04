"""Print the hybrid-mask kwargs actually used by each run, read from its wandb config.

Run names only carry the params whoever wrote the script chose to put in EXP_NAME, so a name
without `epslow` does not mean eps_low was unset -- it means the code default (0.5) applied.
This reads the recorded config instead of inferring from names.

Usage:
    python opd_inference/wandb_mask_kwargs.py <id> [<id> ...]
    python opd_inference/wandb_mask_kwargs.py --all_opdt4
"""

import sys

import wandb

ENTITY_PROJECT = "rl_agent/verl_opd_dapo"
KEYS = ["eps_low", "fkl_vote_threshold", "coverage_threshold", "teacher_top_p", "prob_floor"]


def flat_get(cfg, leaf):
    """Config keys may be flattened ('a.b.c') or nested; try both."""
    for k, v in cfg.items():
        if k.endswith(leaf):
            return v
    node = cfg.get("distillation")
    if isinstance(node, dict):
        loss = node.get("distillation_loss", {})
        kw = loss.get("hybrid_mask_kwargs", {}) if isinstance(loss, dict) else {}
        if isinstance(kw, dict) and leaf in kw:
            return kw[leaf]
    return None


def main():
    args = sys.argv[1:]
    api = wandb.Api()

    if args and args[0] == "--all_opdt4":
        runs = [r for r in api.runs(ENTITY_PROJECT) if "opdt4" in (r.name or "")
                or "theory_guided4" in (r.name or "")]
    else:
        runs = [api.run(f"{ENTITY_PROJECT}/{a}") for a in args]

    hdr = f"{'run':<12}{'eps_low':>9}{'vote':>7}{'cov':>6}{'top_p':>7}{'avg4':>8}  name"
    print(hdr)
    print("-" * 74)
    rows = []
    for r in runs:
        cfg = r.config
        vals = [flat_get(cfg, k) for k in KEYS]
        s = r.summary
        means = [s.get(f"val-core/{d}/acc/mean@8") for d in ["aime", "aime25", "aime26", "amc23"]]
        avg = None if any(m is None for m in means) else sum(means) / 4 * 100
        rows.append((r.id, vals, avg, r.name, s.get("_step")))

    for rid, vals, avg, name, step in sorted(rows, key=lambda x: (x[2] is None, -(x[2] or 0))):
        eps, vote, cov, top_p, _ = vals
        a = "-" if avg is None else f"{avg:.2f}"
        short = (name or "").replace("fsdp/student-Qwen/", "").replace("opd/", "")
        print(f"{rid:<12}{str(eps):>9}{str(vote):>7}{str(cov):>6}{str(top_p):>7}{a:>8}  "
              f"step={step} {short[:66]}")


if __name__ == "__main__":
    main()
