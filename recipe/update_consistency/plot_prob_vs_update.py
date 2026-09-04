"""Two-panel dual-axis binned plot: rollout-probability distribution vs realized update.

Panel (a): token-count histogram + mean realized Δlogπ
Panel (b): token-count histogram + mean realized Δπ

Data: /fsx/xinnanzh/data/update_consistency_multi_pair/<pair>/step_0000NN.parquet
Tokens: is_sampled_u == True  (the token actually emitted by the rollout policy)
  p      = pi_S_before
  Δlogπ  = log(pi_S_after) - log(pi_S_before)
  Δπ     = delta_pi_actual
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

COLS = ["is_sampled_u", "pi_S_before", "pi_S_after", "delta_pi_actual", "A_value"]

# --- styling -----------------------------------------------------------------
BAR_COLOR  = "#4C78A8"   # blue  — token counts
LINE_COLOR = "#F58518"   # orange — mean realized update
INK        = "#1A1A1A"
MUTED      = "#6B6B6B"


def load(data_dir, pairs, steps):
    frames = []
    all_pairs = sorted(d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d != "figures")
    use_pairs = all_pairs if pairs == ["all"] else pairs
    for p in use_pairs:
        files = sorted(glob.glob(os.path.join(data_dir, p, "step_*.parquet")))
        if steps is not None:
            files = [f for f in files
                     if int(os.path.basename(f).split("_")[1].split(".")[0]) in steps]
        for f in files:
            df = pd.read_parquet(f, columns=COLS)
            df = df[df.is_sampled_u].drop(columns="is_sampled_u")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(out):,} sampled tokens from {len(use_pairs)} pair(s)")
    return out


def binstats(p, y, nbins):
    """Uniform bins over [0,1]; return centers, counts, mean(y) per bin."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins).astype(float)
    sums = np.bincount(idx, weights=y, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return edges, centers, counts, means


def panel(ax, edges, centers, counts, means, right_label, title, min_count, log_counts,
          nonneg=False):
    width = edges[1] - edges[0]

    # blue count histogram (left axis)
    ax.bar(centers, counts, width=width * 0.92, color=BAR_COLOR,
           edgecolor="white", linewidth=0.4, label="Token count", zorder=2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Token probability under rollout policy", fontsize=10.5, color=INK)
    ax.set_ylabel("Token count", fontsize=10.5, color=INK)
    if log_counts:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=9.5, colors=MUTED)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CCCCCC")

    # orange mean-update line (right axis)
    ax2 = ax.twinx()
    m = np.where(counts >= min_count, means, np.nan)
    if not nonneg:
        ax2.axhline(0.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=2)
    ax2.plot(centers, m, color=LINE_COLOR, linewidth=2.0, marker="o",
             markersize=4.5, markeredgecolor="white", markeredgewidth=0.7,
             label=right_label, zorder=3)
    # headroom above the curve so the legend never overlaps it
    lo = 0.0 if nonneg else np.nanmin(np.append(m, 0.0))
    hi = np.nanmax(np.append(m, 0.0))
    span = (hi - lo) or 1.0
    ax2.set_ylim(lo if nonneg else lo - 0.06 * span, hi + 0.26 * span)
    ax2.set_ylabel(right_label, fontsize=10.5, color=LINE_COLOR)
    ax2.tick_params(axis="y", labelsize=9.5, colors=LINE_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(LINE_COLOR)

    ax.set_title(title, fontsize=11.5, color=INK, pad=8)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False,
              loc="upper center", ncol=2, handlelength=1.6)
    return ax2


def panel_pred(ax, pred, act, nbins, min_count, log_counts, xlabel, ylabel, title):
    """Panel (c): bin by the *predictor* magnitude (log-spaced) instead of by pi."""
    ok = np.isfinite(pred) & np.isfinite(act) & (pred > 0)
    x, y = pred[ok], act[ok]
    lx = np.log10(x)

    # trim the extreme tails so the bins cover the bulk of the mass
    lo, hi = np.percentile(lx, [0.5, 99.5])
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = (lx >= lo) & (lx <= hi)
    idx = np.clip(np.digitize(lx[keep], edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins).astype(float)
    sums = np.bincount(idx, weights=y[keep], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)

    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width * 0.92, color=BAR_COLOR,
           edgecolor="white", linewidth=0.4, label="Token count", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=10.5, color=INK)
    ax.set_ylabel("Token count", fontsize=10.5, color=INK)
    if log_counts:
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$"))
    ax.tick_params(axis="both", labelsize=9.5, colors=MUTED)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CCCCCC")

    ax2 = ax.twinx()
    m = np.where(counts >= min_count, means, np.nan)
    # Proportional (slope-1) reference, anchored on the top two decades of the
    # predictor, where the realized update is well clear of the bf16 round-off
    # floor.  A median over all tokens is meaningless here: most tokens have a
    # vanishing predictor but a floored, non-zero realized update.
    sel = x > 10.0 ** (hi - 2)
    c = np.median(y[sel] / x[sel]) if sel.sum() > 1000 else np.median(y / x)
    ax2.plot(centers, c * 10.0 ** centers, color=MUTED, linestyle="--",
             linewidth=1.1, label=f"slope 1 ($c$={c:.2f})", zorder=2)
    ax2.plot(centers, m, color=LINE_COLOR, linewidth=2.0, marker="o",
             markersize=4.5, markeredgecolor="white", markeredgewidth=0.7,
             label=ylabel, zorder=3)
    ax2.set_yscale("log")
    # scale the axis to the realized curve, letting the reference line clip
    fin = np.isfinite(m)
    ax2.set_ylim(np.nanmin(m[fin]) * 0.35, np.nanmax(m[fin]) * 12.0)
    ax2.set_ylabel(ylabel, fontsize=10.5, color=LINE_COLOR)
    ax2.tick_params(axis="y", labelsize=9.5, colors=LINE_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(LINE_COLOR)

    pos = y > 0
    slope = np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)[0]
    r_log = np.corrcoef(np.log(x[pos]), np.log(y[pos]))[0, 1]
    ax2.text(0.03, 0.03, f"$r_{{\\log}}$ = {r_log:+.3f}\nlog-log slope = {slope:.2f}",
             transform=ax2.transAxes, va="bottom", ha="left", fontsize=9, color=INK,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#DDDDDD", linewidth=0.6), zorder=6)

    ax.set_title(title, fontsize=11.5, color=INK, pad=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False,
              loc="upper left", ncol=1, handlelength=1.6)
    return centers, counts, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/fsx/xinnanzh/data/update_consistency_multi_pair")
    ap.add_argument("--pairs", nargs="+", default=["all"])
    ap.add_argument("--steps", nargs="+", type=int, default=None,
                    help="step numbers to include (default: all)")
    ap.add_argument("--nbins", type=int, default=45)
    ap.add_argument("--min-count", type=int, default=30,
                    help="hide mean-line point for bins with fewer tokens")
    ap.add_argument("--linear-counts", action="store_true",
                    help="linear left (count) axis; default is log, which reveals "
                         "the U-shaped distribution hidden by the p~1 spike")
    ap.add_argument("--a-clip", type=float, default=None,
                    help="keep only |A| <= this (mirrors training-time A clamp, e.g. 10)")
    ap.add_argument("--pred-panel", action="store_true",
                    help="add panel (c): predicted |A pi(1-pi)^2| vs realized |Δπ|, "
                         "binned on a log-spaced predictor axis")
    ap.add_argument("--abs", action="store_true",
                    help="plot mean |Δlogπ| / mean |Δπ| (update magnitude) instead of "
                         "the signed mean, which cancels up- and down-weighted tokens")
    ap.add_argument("--out", default="/fsx/xinnanzh/eval_out/prob_vs_update.png")
    args = ap.parse_args()

    df = load(args.data_dir, args.pairs, set(args.steps) if args.steps else None)

    if args.a_clip is not None:
        n0 = len(df)
        df = df[df.A_value.abs() <= args.a_clip]
        print(f"A-clip |A|<={args.a_clip}: kept {len(df):,}/{n0:,} ({100*len(df)/n0:.2f}%)")

    p = df.pi_S_before.to_numpy(dtype=np.float64)
    dlogpi = np.log(df.pi_S_after.to_numpy(np.float64)) - np.log(p)
    dpi = df.delta_pi_actual.to_numpy(np.float64)
    A = df.A_value.to_numpy(np.float64)

    ok = np.isfinite(p) & np.isfinite(dlogpi) & np.isfinite(dpi) & (p >= 0) & (p <= 1)
    p, dlogpi, dpi, A = p[ok], dlogpi[ok], dpi[ok], A[ok]
    print(f"usable tokens: {len(p):,}")

    if args.abs:
        dlogpi, dpi = np.abs(dlogpi), np.abs(dpi)
        tex_lp, tex_p = r"|\Delta\log\pi|", r"|\Delta\pi|"
    else:
        tex_lp, tex_p = r"\Delta\log\pi", r"\Delta\pi"

    edges, centers, counts, mean_dlog = binstats(p, dlogpi, args.nbins)
    _, _, _, mean_dpi = binstats(p, dpi, args.nbins)

    ncols = 3 if args.pred_panel else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.3))
    panel(axes[0], edges, centers, counts, mean_dlog,
          f"Mean actual ${tex_lp}$",
          f"(a) Probability distribution and realized ${tex_lp}$",
          args.min_count, not args.linear_counts, nonneg=args.abs)
    panel(axes[1], edges, centers, counts, mean_dpi,
          f"Mean actual ${tex_p}$",
          f"(b) Probability distribution and realized ${tex_p}$",
          args.min_count, not args.linear_counts, nonneg=args.abs)
    pc = None
    if args.pred_panel:
        pred_c = np.abs(A) * p * (1.0 - p) ** 2
        pc = panel_pred(axes[2], pred_c, np.abs(dpi), args.nbins, args.min_count,
                        not args.linear_counts,
                        r"Predicted $|A\,\pi(1-\pi)^2|$",
                        r"Mean actual $|\Delta\pi|$",
                        r"(c) Predicted $|A\,\pi(1-\pi)^2|$ vs realized $|\Delta\pi|$")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    pdf = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}\nwrote {pdf}")

    # dump binned numbers for further analysis
    csv = os.path.splitext(args.out)[0] + "_bins.csv"
    sfx = "_abs" if args.abs else ""
    pd.DataFrame({
        "bin_left": edges[:-1], "bin_right": edges[1:], "bin_center": centers,
        "token_count": counts.astype(int),
        f"mean{sfx}_dlogpi": mean_dlog, f"mean{sfx}_dpi": mean_dpi,
    }).to_csv(csv, index=False)
    print(f"wrote {csv}")

    # console summary
    print(f"\n bin_center   count   mean{sfx}_dlogpi     mean{sfx}_dpi")
    for c, n, a, b in zip(centers, counts, mean_dlog, mean_dpi):
        if n >= args.min_count:
            print(f"   {c:8.3f} {int(n):8d}   {a:+12.5f}   {b:+12.3e}")

    if pc is not None:
        pcen, pcnt, pmean = pc
        print("\n panel (c):  log10(predicted)     count     mean |dpi|")
        for c, n, v in zip(pcen, pcnt, pmean):
            if n >= args.min_count:
                print(f"        {c:+10.2f} {int(n):12d}   {v:12.3e}")


if __name__ == "__main__":
    main()
