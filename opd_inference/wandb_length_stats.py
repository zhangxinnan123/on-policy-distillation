"""Compare training-rollout truncation across runs.

`response_length/clip_ratio` is the fraction of training rollouts that hit the response cap;
`response/complete_ratio_non_aborted` is the fraction that terminated on their own. A run with
clip_ratio near 1 has stopped producing finishable reasoning, which confounds any eval-score
comparison against runs that still terminate.

Usage:
    python opd_inference/wandb_length_stats.py <id>:<label> [<id>:<label> ...]
"""

import sys

import wandb

DATASETS = ["aime", "aime25", "aime26", "amc23"]
ENTITY_PROJECT = "rl_agent/verl_opd_dapo"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    api = wandb.Api()
    header = f"{'method':<14}{'clip%':>7}{'complete%':>11}{'reslen':>9}{'val_reslen':>12}{'avg4 m@8':>10}"
    print(header)
    print("-" * len(header))

    for arg in args:
        rid, _, label = arg.partition(":")
        label = label or rid
        s = api.run(f"{ENTITY_PROJECT}/{rid}").summary
        clip = s.get("response_length/clip_ratio")
        comp = s.get("response/complete_ratio_non_aborted")
        reslen = s.get("response_length/mean")
        val_reslen = s.get("val/response_length/mean")
        means = [s.get(f"val-core/{d}/acc/mean@8") for d in DATASETS]

        def fmt(v, scale=100.0, nd=1):
            return "-" if v is None else f"{v * scale:.{nd}f}"

        avg = "-" if any(m is None for m in means) else f"{sum(means) / 4 * 100:.2f}"
        print(
            f"{label:<14}{fmt(clip):>7}{fmt(comp):>11}{fmt(reslen, 1, 0):>9}"
            f"{fmt(val_reslen, 1, 0):>12}{avg:>10}"
        )


if __name__ == "__main__":
    main()
