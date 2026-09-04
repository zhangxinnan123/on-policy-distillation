"""Dual-axis binned plots of the advantage A vs realized updates and ΔFKL.

Same visual grammar as plot_prob_vs_update.py, but binned on the raw (signed)
advantage A = log pi_T(u) - log pi_S(u) instead of on the rollout probability.

Default (pooled) figure — three panels over all pairs:
  (a) A-count histogram + mean realized Δlogπ(u)
  (b) A-count histogram + mean realized Δπ(u)
  (c) A-count histogram + mean realized ΔFKL at that position

With --by-pair, writes one figure per quantity instead, each a students × teachers
grid of facets on shared axes so pairs can be compared directly.

One row per (sequence_id, response_pos), so the count histogram is identical in
all panels of the pooled figure.  ΔFKL is truncated to the candidate set saved
per position (the teacher top-k, which covers ~100% of teacher mass):

    ΔFKL = -sum_c pi_T(c) * [log pi_S_after(c) - log pi_S_before(c)]

Negative ΔFKL means the forward KL to the teacher improved.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = ["sequence_id", "response_pos", "is_sampled_u", "pi_S_before", "pi_S_after",
        "pi_T", "delta_pi_actual", "A_value"]

BAR_COLOR  = "#4C78A8"   # blue  — position counts
LINE_COLOR = "#F58518"   # orange — mean realized quantity
INK        = "#1A1A1A"
MUTED      = "#6B6B6B"

SHORT = {
    "Qwen_Qwen3-1.7B-Base": "Qwen3-1.7B-Base",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B": "R1-Distill-1.5B",
    "lllyx_Qwen3-1.7B-SFT": "Qwen3-1.7B-SFT",
    "Qwen_Qwen3-4B": "Qwen3-4B",
    "hbx_JustRL-DeepSeek-1.5B": "JustRL-DS-1.5B",
    "lllyx_Qwen3-4B-Base-GRPO": "Qwen3-4B-GRPO",
}
XLABEL = r"Advantage $A=\log\pi_T(u)-\log\pi_S(u)$"

# quantity key -> (dataframe column, right-axis label, panel-title fragment)
QUANTS = {
    "dlogpi": ("dlogpi", r"Mean actual $\Delta\log\pi(u)$", r"$\Delta\log\pi(u)$"),
    "dpi":    ("delta_pi_actual", r"Mean actual $\Delta\pi(u)$", r"$\Delta\pi(u)$"),
    "dfkl":   ("dFKL", r"Mean actual $\Delta\mathrm{FKL}$", r"$\Delta\mathrm{FKL}$"),
}


def short(name):
    return SHORT.get(name, name.split("_", 1)[-1])


def load_positions(data_dir, pairs, steps):
    """Collapse candidate rows to one row per position, carrying A and ΔFKL."""
    frames = []
    all_pairs = sorted(d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d != "figures")
    use_pairs = all_pairs if pairs == ["all"] else pairs
    for pr in use_pairs:
        for f in sorted(glob.glob(os.path.join(data_dir, pr, "step_*.parquet"))):
            st = int(os.path.basename(f).split("_")[1].split(".")[0])
            if steps is not None and st not in steps:
                continue
            df = pd.read_parquet(f, columns=COLS)
            pb = df.pi_S_before.to_numpy(np.float64)
            pa = df.pi_S_after.to_numpy(np.float64)
            pT = df.pi_T.to_numpy(np.float64)
            df["_w"] = pT * (np.log(pa) - np.log(pb))
            df["_tp"] = pT
            agg = (df.groupby(["sequence_id", "response_pos"], sort=False)
                     .agg(sum_w=("_w", "sum"), m_T=("_tp", "sum")))
            sam = df[df.is_sampled_u].set_index(["sequence_id", "response_pos"])
            sam = sam.assign(dlogpi=np.log(sam.pi_S_after.to_numpy(np.float64))
                                    - np.log(sam.pi_S_before.to_numpy(np.float64)))
            sam = sam[["A_value", "delta_pi_actual", "dlogpi"]]
            j = agg.join(sam, how="inner").reset_index(drop=True)
            j["dFKL"] = -j.sum_w
            j["pair"] = pr
            frames.append(j[["pair", "A_value", "dlogpi", "delta_pi_actual", "dFKL", "m_T"]])
    out = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(out):,} positions from {len(use_pairs)} pair(s)")
    print(f"teacher mass covered m_T: median={out.m_T.median():.4f} "
          f"p10={out.m_T.quantile(.10):.4f}")
    return out


def binstats(a, y, edges):
    nb = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = (a >= edges[0]) & (a <= edges[-1])
    idx = np.clip(np.digitize(a[keep], edges) - 1, 0, nb - 1)
    counts = np.bincount(idx, minlength=nb).astype(float)
    sums = np.bincount(idx, weights=y[keep], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return centers, counts, means


def panel(ax, edges, centers, counts, means, right_label, title, min_count, log_counts,
          ylim=None, count_lim=None, xlabel=True, left_label=True, right_label_on=True,
          legend=True, title_size=11.5):
    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width * 0.92, color=BAR_COLOR,
           edgecolor="white", linewidth=0.4, label="Position count", zorder=2)
    ax.set_xlim(edges[0], edges[-1])
    if xlabel:
        ax.set_xlabel(XLABEL, fontsize=10.5, color=INK)
    if left_label:
        ax.set_ylabel("Position count", fontsize=10.5, color=INK)
    if log_counts:
        ax.set_yscale("log")
    if count_lim:
        ax.set_ylim(*count_lim)
    ax.tick_params(axis="both", labelsize=9.5, colors=MUTED)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CCCCCC")
    ax.axvline(0.0, color="#BBBBBB", linewidth=0.9, zorder=1)

    ax2 = ax.twinx()
    m = np.where(counts >= min_count, means, np.nan)
    ax2.axhline(0.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=2)
    ax2.plot(centers, m, color=LINE_COLOR, linewidth=2.0, marker="o",
             markersize=4.5, markeredgecolor="white", markeredgewidth=0.7,
             label=right_label, zorder=3)
    if ylim:
        ax2.set_ylim(*ylim)
    else:
        lo, hi = np.nanmin(np.append(m, 0.0)), np.nanmax(np.append(m, 0.0))
        span = (hi - lo) or 1.0
        ax2.set_ylim(lo - 0.06 * span, hi + 0.26 * span)
    if right_label_on:
        ax2.set_ylabel(right_label, fontsize=10.5, color=LINE_COLOR)
    ax2.tick_params(axis="y", labelsize=9.5, colors=LINE_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(LINE_COLOR)

    ax.set_title(title, fontsize=title_size, color=INK, pad=8)
    if legend:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False,
                  loc="upper center", ncol=2, handlelength=1.6)
    return ax2


def save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\nwrote {pdf}")


def by_pair(d, edges, args):
    """One figure per quantity; students as rows, teachers as columns."""
    d = d.copy()
    d[["_stu", "_tea"]] = d.pair.str.split("___", expand=True)
    students = sorted(d._stu.unique())
    teachers = sorted(d._tea.unique())
    print(f"\nfaceting {len(students)} students x {len(teachers)} teachers")

    # precompute every facet so axes can be shared
    cell = {}
    for st in students:
        for te in teachers:
            g = d[(d._stu == st) & (d._tea == te)]
            if not len(g):
                continue
            A = g.A_value.to_numpy(np.float64)
            cell[(st, te)] = {
                k: binstats(A, g[col].to_numpy(np.float64), edges)
                for k, (col, _, _) in QUANTS.items()
            }

    cmax = max(c[1].max() for v in cell.values() for c in [v["dlogpi"]])
    count_lim = (0.7, cmax * 12.0) if not args.linear_counts else None

    base, ext = os.path.splitext(args.out)
    for key, (col, rlabel, tfrag) in QUANTS.items():
        vals = [np.where(c[1] >= args.min_count, c[2], np.nan)
                for v in cell.values() for c in [v[key]]]
        lo = min(np.nanmin(np.append(v, 0.0)) for v in vals)
        hi = max(np.nanmax(np.append(v, 0.0)) for v in vals)
        span = (hi - lo) or 1.0
        ylim = (lo - 0.06 * span, hi + 0.26 * span)

        nr, nc = len(students), len(teachers)
        fig, axes = plt.subplots(nr, nc, figsize=(5.9 * nc, 3.9 * nr), squeeze=False)
        for i, st in enumerate(students):
            for j, te in enumerate(teachers):
                ax = axes[i][j]
                if (st, te) not in cell:
                    ax.axis("off")
                    continue
                centers, counts, means = cell[(st, te)][key]
                panel(ax, edges, centers, counts, means, rlabel,
                      f"{short(st)}  $\\rightarrow$  {short(te)}",
                      args.min_count, not args.linear_counts,
                      ylim=ylim, count_lim=count_lim,
                      xlabel=(i == nr - 1), left_label=(j == 0),
                      right_label_on=(j == nc - 1),
                      legend=(i == 0 and j == 0), title_size=10.5)
        fig.suptitle(f"Advantage $A$ vs realized {tfrag}, by student $\\rightarrow$ teacher "
                     f"(shared axes)", fontsize=13, color=INK, y=1.002)
        fig.tight_layout()
        save(fig, f"{base}_{key}_bypair{ext}")

    # per-pair numeric summary
    print("\n=== per-pair summary (positions in range) ===")
    print(f"{'student -> teacher':<38} {'n':>9} {'A<0 %':>7} {'dFKL<0 %':>9} "
          f"{'mean dFKL':>11} {'argmax dpi':>11} {'max dpi':>10}")
    for st in students:
        for te in teachers:
            if (st, te) not in cell:
                continue
            g = d[(d._stu == st) & (d._tea == te)]
            centers, counts, mdpi = cell[(st, te)]["dpi"]
            m = np.where(counts >= args.min_count, mdpi, np.nan)
            pos = centers > 0
            k = np.nanargmax(np.where(pos, m, np.nan))
            print(f"{short(st)+' -> '+short(te):<38} {len(g):>9,} "
                  f"{100*(g.A_value<0).mean():>7.2f} {100*(g.dFKL<0).mean():>9.2f} "
                  f"{g.dFKL.mean():>+11.5f} {centers[k]:>+11.2f} {m[k]:>10.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/fsx/xinnanzh/data/update_consistency_multi_pair")
    ap.add_argument("--pairs", nargs="+", default=["all"])
    ap.add_argument("--steps", nargs="+", type=int, default=None,
                    help="step numbers to include (default: all)")
    ap.add_argument("--nbins", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--a-clip", type=float, default=10.0,
                    help="keep only |A| <= this (mirrors training-time A clamp)")
    ap.add_argument("--a-range", type=float, nargs="+", default=None, metavar="LO [HI]",
                    help="bin A over [LO, HI]; one value R means [-R, R]. "
                         "Default is the symmetric 0.5/99.5 percentile span.")
    ap.add_argument("--by-pair", action="store_true",
                    help="write one faceted figure per quantity instead of the "
                         "pooled three-panel figure")
    ap.add_argument("--linear-counts", action="store_true")
    ap.add_argument("--out", default="/fsx/xinnanzh/eval_out/A_vs_update.png")
    args = ap.parse_args()

    d = load_positions(args.data_dir, args.pairs,
                       set(args.steps) if args.steps else None)
    if args.a_clip is not None:
        n0 = len(d)
        d = d[d.A_value.abs() <= args.a_clip]
        print(f"A-clip |A|<={args.a_clip}: kept {len(d):,}/{n0:,} ({100*len(d)/n0:.2f}%)")

    d = d[np.isfinite(d.A_value) & np.isfinite(d.dlogpi)
          & np.isfinite(d.delta_pi_actual) & np.isfinite(d.dFKL)]
    A = d.A_value.to_numpy(np.float64)

    q = np.percentile(A, [0.5, 1, 25, 50, 75, 99, 99.5])
    print(f"A percentiles  0.5%={q[0]:+.3f}  1%={q[1]:+.3f}  25%={q[2]:+.3f} "
          f"50%={q[3]:+.3f}  75%={q[4]:+.3f}  99%={q[5]:+.3f}  99.5%={q[6]:+.3f}")
    if args.a_range is None:
        R = max(abs(q[0]), abs(q[6])); lo, hi = -R, R
    elif len(args.a_range) == 1:
        lo, hi = -args.a_range[0], args.a_range[0]
    else:
        lo, hi = args.a_range[0], args.a_range[1]
    edges = np.linspace(lo, hi, args.nbins + 1)
    inr = (A >= lo) & (A <= hi)
    print(f"binning A over [{lo:.3f}, {hi:.3f}] in {args.nbins} bins "
          f"(width {edges[1]-edges[0]:.4f}, {100*inr.mean():.2f}% in range; "
          f"{100*(A<lo).mean():.2f}% below, {100*(A>hi).mean():.2f}% above)")

    if args.by_pair:
        by_pair(d, edges, args)
        return

    centers, counts, m_dlp = binstats(A, d.dlogpi.to_numpy(np.float64), edges)
    _, _, m_dpi = binstats(A, d.delta_pi_actual.to_numpy(np.float64), edges)
    _, _, m_dfk = binstats(A, d.dFKL.to_numpy(np.float64), edges)

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 4.3))
    for ax, means, key in zip(axes, (m_dlp, m_dpi, m_dfk), QUANTS):
        _, rlabel, tfrag = QUANTS[key]
        tag = "abc"[list(QUANTS).index(key)]
        panel(ax, edges, centers, counts, means, rlabel,
              f"({tag}) Advantage distribution and realized {tfrag}",
              args.min_count, not args.linear_counts)
    fig.tight_layout()
    save(fig, args.out)

    csv = os.path.splitext(args.out)[0] + "_bins.csv"
    pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "bin_center": centers,
                  "position_count": counts.astype(int), "mean_dlogpi": m_dlp,
                  "mean_dpi": m_dpi, "mean_dFKL": m_dfk}).to_csv(csv, index=False)
    print(f"wrote {csv}")

    print("\n   A_center      count    mean_dlogpi       mean_dpi      mean_dFKL")
    for c, n, a, b, e in zip(centers, counts, m_dlp, m_dpi, m_dfk):
        if n >= args.min_count:
            print(f"   {c:+9.3f} {int(n):10d}   {a:+12.5f}   {b:+12.3e}   {e:+12.3e}")


if __name__ == "__main__":
    main()
