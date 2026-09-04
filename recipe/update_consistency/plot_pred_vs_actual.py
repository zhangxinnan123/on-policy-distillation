"""Predicted vs actual |Δlogπ| for sampled tokens, two candidate predictors.

Panel (a): predicted |A·π(1−π)|
Panel (b): predicted |A·π(1−π)²|

Target (--target): |Δπ| = |delta_pi_actual|  (default)
                   |Δlogπ| = |log π_after − log π_before|
A = log π_T(u) − log π_S(u).  Sampled tokens only (is_sampled_u).

Available predictors (--pred-a / --pred-b), all times |A|:
  p1mp      π(1−π)            Δπ shape, first order
  p1mp2     π(1−π)²           Δπ shape, assumes ‖π‖²≈π²  (only valid for π≳0.9)
  1mp       (1−π)             Δlogπ shape, first order
  1mp2      (1−π)²            Δlogπ shape, assumes ‖π‖²≈π²
  exact_lp  (1−2π+‖π‖²)       Δlogπ shape, exact first order
  exact_p   π(1−2π+‖π‖²)      Δπ shape, exact first order
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

COLS = ["is_sampled_u", "pi_S_before", "pi_S_after", "A_value", "norm_sq_S",
        "delta_pi_actual"]

# shape factor -> (callable(p, nsq), latex label)
PREDICTORS = {
    "p1mp":     (lambda p, n: p * (1 - p),            r"A\,\pi(1-\pi)"),
    "p1mp2":    (lambda p, n: p * (1 - p) ** 2,       r"A\,\pi(1-\pi)^2"),
    "1mp":      (lambda p, n: (1 - p),                r"A\,(1-\pi)"),
    "1mp2":     (lambda p, n: (1 - p) ** 2,           r"A\,(1-\pi)^2"),
    "exact_lp": (lambda p, n: 1 - 2 * p + n,          r"A\,(1-2\pi+\|\pi\|^2)"),
    "exact_p":  (lambda p, n: p * (1 - 2 * p + n),    r"A\,\pi(1-2\pi+\|\pi\|^2)"),
}
LINE_COLOR = "#F58518"
REF_COLOR = "#333333"
INK, MUTED = "#1A1A1A", "#6B6B6B"


def load(data_dir, pairs, steps):
    frames = []
    allp = sorted(d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and d != "figures")
    use = allp if pairs == ["all"] else pairs
    for p in use:
        for f in sorted(glob.glob(os.path.join(data_dir, p, "step_*.parquet"))):
            if steps and int(os.path.basename(f).split("_")[1].split(".")[0]) not in steps:
                continue
            df = pd.read_parquet(f, columns=COLS)
            frames.append(df[df.is_sampled_u].drop(columns="is_sampled_u"))
    out = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(out):,} sampled tokens from {len(use)} pair(s)")
    return out


def panel(ax, pred, act, xlabel, ylabel, title, nbins, gridsize):
    ok = np.isfinite(pred) & np.isfinite(act) & (pred > 0) & (act > 0)
    x, y = np.log10(pred[ok]), np.log10(act[ok])

    hb = ax.hexbin(x, y, gridsize=gridsize, cmap="Blues", norm=LogNorm(),
                   mincnt=1, linewidths=0, zorder=2)

    # y = x reference
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color=REF_COLOR, linestyle="--", linewidth=1.2,
            label="$y=x$", zorder=4)

    # binned median of actual vs predicted
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 50:
            med[b] = np.median(y[m])
    ax.plot(centers, med, color=LINE_COLOR, linewidth=2.0, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.7,
            label="Median actual per bin", zorder=5)

    r_log = np.corrcoef(x, y)[0, 1]
    ratio = np.median(act[ok] / pred[ok])
    ax.text(0.03, 0.97, f"$r_{{\\log}}$ = {r_log:+.3f}\nmedian(act/pred) = {ratio:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#DDDDDD", linewidth=0.6))

    ax.set_xlabel(xlabel, fontsize=10.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=11.5, color=INK, pad=8)
    ax.tick_params(labelsize=9.5, colors=MUTED)
    ax.grid(True, color="#DDDDDD", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CCCCCC")
    ax.legend(fontsize=9, frameon=False, loc="lower right", handlelength=1.8)
    return hb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/fsx/xinnanzh/data/update_consistency_multi_pair")
    ap.add_argument("--pairs", nargs="+", default=["all"])
    ap.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--nbins", type=int, default=45)
    ap.add_argument("--gridsize", type=int, default=70)
    ap.add_argument("--target", default="dpi", choices=["dpi", "dlogpi"],
                    help="y-axis: |Δπ| (default) or |Δlogπ|")
    ap.add_argument("--pred-a", default="p1mp", choices=list(PREDICTORS),
                    help="shape factor for panel (a)")
    ap.add_argument("--pred-b", default="p1mp2", choices=list(PREDICTORS),
                    help="shape factor for panel (b)")
    ap.add_argument("--a-clip", type=float, default=None,
                    help="keep only |A| <= this (mirrors training-time A clamp, e.g. 10)")
    ap.add_argument("--out", default="/fsx/xinnanzh/eval_out/pred_vs_actual_dlogpi.png")
    args = ap.parse_args()

    df = load(args.data_dir, args.pairs, set(args.steps) if args.steps else None)
    if args.a_clip is not None:
        n0 = len(df)
        df = df[df.A_value.abs() <= args.a_clip]
        print(f"A-clip |A|<={args.a_clip}: kept {len(df):,}/{n0:,} ({100*len(df)/n0:.2f}%)")

    p = df.pi_S_before.to_numpy(np.float64)
    A = df.A_value.to_numpy(np.float64)
    nsq = df.norm_sq_S.to_numpy(np.float64)
    if args.target == "dpi":
        act = np.abs(df.delta_pi_actual.to_numpy(np.float64))
        tgt_tex = r"|\Delta\pi|"
    else:
        act = np.abs(np.log(df.pi_S_after.to_numpy(np.float64)) - np.log(p))
        tgt_tex = r"|\Delta\log\pi|"
    ylabel = f"Actual ${tgt_tex}$   ($\\log_{{10}}$)"

    fn_a, lab_a = PREDICTORS[args.pred_a]
    fn_b, lab_b = PREDICTORS[args.pred_b]
    pred_a = np.abs(A * fn_a(p, nsq))
    pred_b = np.abs(A * fn_b(p, nsq))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    hb1 = panel(axes[0], pred_a, act,
                f"Predicted $|{lab_a}|$   ($\\log_{{10}}$)",
                ylabel,
                f"(a) Predicted $|{lab_a}|$ vs actual ${tgt_tex}$",
                args.nbins, args.gridsize)
    hb2 = panel(axes[1], pred_b, act,
                f"Predicted $|{lab_b}|$   ($\\log_{{10}}$)",
                ylabel,
                f"(b) Predicted $|{lab_b}|$ vs actual ${tgt_tex}$",
                args.nbins, args.gridsize)
    for ax, hb in zip(axes, (hb1, hb2)):
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("Token count", fontsize=9.5, color=MUTED)
        cb.ax.tick_params(labelsize=8.5, colors=MUTED)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    print(f"wrote {os.path.splitext(args.out)[0]}.pdf")

    for nm, pr in [(f"(a) {args.pred_a}", pred_a), (f"(b) {args.pred_b}", pred_b)]:
        ok = np.isfinite(pr) & np.isfinite(act) & (pr > 0) & (act > 0)
        lx, ly = np.log(pr[ok]), np.log(act[ok])
        print(f"  {nm:16s} n={ok.sum():,}  r_log={np.corrcoef(lx,ly)[0,1]:+.4f}"
              f"  slope={np.polyfit(lx,ly,1)[0]:.3f}"
              f"  median(act/pred)={np.median(act[ok]/pr[ok]):.4g}")


if __name__ == "__main__":
    main()
