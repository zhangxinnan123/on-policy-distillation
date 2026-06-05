# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Combined 2x2 figure (sign on row 1, magnitude on row 2).

Works for both single-token mode (uses `is_target` rows) and batch mode (uses
all rows). Pass --suffix _b1 / _b8 / etc. to distinguish runs.

  python -m recipe.update_consistency.analyze_combined \\
      --measurements .../measurements.parquet \\
      --output-dir   recipe/update_consistency/figures \\
      --suffix _b1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _sign_agreement(pred: np.ndarray, actual: np.ndarray) -> float:
    m = (pred != 0) & (actual != 0)
    if m.sum() == 0:
        return float("nan")
    return float((pred[m] == actual[m]).mean())


def _binned_metric(x, y, bins, op=np.nanmean, linear=False):
    idx = np.digitize(x, bins) - 1
    centers, vals, counts = [], [], []
    for i in range(len(bins) - 1):
        mask = idx == i
        n = int(mask.sum())
        counts.append(n)
        center = 0.5 * (bins[i] + bins[i + 1]) if linear else np.sqrt(bins[i] * bins[i + 1])
        centers.append(center)
        vals.append(op(y[mask]) if n > 0 else np.nan)
    return np.array(centers), np.array(vals), np.array(counts)


def _sign_binned_panel(ax, x_all, sign_all, *, sub_u, sub_c, bins,
                       xlabel, title):
    """Style mirrors update_consistency.pdf: line+circle for metric (left y),
    lightpink bars for token count (right y, log)."""
    # token count from the union (left bars) — use the overall histogram
    centers_all, _, n_all = _binned_metric(x_all, sign_all, bins,
                                            op=np.nanmean)
    ax2 = ax.twinx()
    ax2.bar(centers_all, np.where(n_all > 0, n_all, np.nan),
            width=np.diff(np.r_[bins[0]*0.9, centers_all]) * 0.7,
            color="lightpink", alpha=0.55, log=True)
    ax2.set_ylabel("token count", color="lightpink")
    ax2.tick_params(axis="y", colors="lightpink")

    # two curves: u and c
    for label, sub, color in [("non-sampled c", sub_c, "steelblue"),
                              ("sampled u",     sub_u, "darkorange")]:
        xv = np.abs(sub[xlabel]).to_numpy() if isinstance(xlabel, str) \
             else sub[xlabel[0]].to_numpy()  # fallback (unused)
        # accept a callable to compute x
        raise_ = False  # placeholder
    # The above is not used; we instead let the caller pass x arrays directly.
    ax.set_xscale("log")
    ax.set_xlabel(xlabel if isinstance(xlabel, str) else "")
    ax.set_ylabel("sign agreement")
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_title(title)


def _binned_curve(ax, x, agree, bins, *, color, label, linear=False):
    centers, vals, n = _binned_metric(x, agree, bins, op=np.nanmean, linear=linear)
    mask = n > 0
    ax.plot(centers[mask], vals[mask], "o-", color=color, label=label,
            markersize=5, lw=1.5)
    return centers, vals, n


def _sign_overall(ax, t, u, c, bins, linear=False):
    """(a) Sign agreement vs |Δπ_actual|, two curves (u, c) + count bars."""
    sign_all = (t["predicted_sign"].to_numpy() == t["actual_sign"].to_numpy()).astype(float)
    x_all = np.abs(t["delta_pi_actual"].to_numpy())
    centers, _, n_all = _binned_metric(x_all, sign_all, bins, op=np.nanmean, linear=linear)

    ax2 = ax.twinx()
    widths = np.diff(np.r_[bins[0] if linear else bins[0]*0.9, centers]) * 0.7
    ax2.bar(centers, np.where(n_all > 0, n_all, np.nan), width=widths,
            color="lightpink", alpha=0.55, log=True)
    ax2.set_ylabel("token count", color="lightpink")
    ax2.tick_params(axis="y", colors="lightpink")

    # c curve
    sign_c = (c["predicted_sign"].to_numpy() == c["actual_sign"].to_numpy()).astype(float)
    x_c = np.abs(c["delta_pi_actual"].to_numpy())
    _binned_curve(ax, x_c, sign_c, bins, color="steelblue", label="non-sampled c", linear=linear)

    # u curve
    sign_u = (u["predicted_sign"].to_numpy() == u["actual_sign"].to_numpy()).astype(float)
    x_u = np.abs(u["delta_pi_actual"].to_numpy())
    _binned_curve(ax, x_u, sign_u, bins, color="darkorange", label="sampled u", linear=linear)

    # merged (u + c)
    _binned_curve(ax, x_all, sign_all, bins, color="black", label="merged (u + c)", linear=linear)

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", label="chance = 0.5")
    ax.set_xscale("linear" if linear else "log")
    ax.set_xlabel("|Δπ_actual|  (signal magnitude)")
    ax.set_ylabel("sign agreement", color="black")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("(a) Sign agreement vs |Δπ_actual|")
    ax.legend(loc="lower right", fontsize=8)


def _sign_vs_A(ax, t, linear=False):
    """(b) Sign agreement vs |A|, binned-curve style (matches Row-2 plots)."""
    u = t[t["is_sampled_u"]]
    c = t[~t["is_sampled_u"]]
    A_abs_max = float(np.abs(t["A_value"].to_numpy()).max())
    bins = np.linspace(0, max(A_abs_max, 1e-6), 30) if linear else np.logspace(-4, 2, 30)

    sign_all = (t["predicted_sign"].to_numpy() == t["actual_sign"].to_numpy()).astype(float)
    x_all = np.abs(t["A_value"].to_numpy())
    centers, _, n_all = _binned_metric(x_all, sign_all, bins, op=np.nanmean, linear=linear)
    ax2 = ax.twinx()
    widths = np.diff(np.r_[bins[0] if linear else bins[0]*0.9, centers]) * 0.7
    ax2.bar(centers, np.where(n_all > 0, n_all, np.nan), width=widths,
            color="lightpink", alpha=0.55, log=True)
    ax2.set_ylabel("token count", color="lightpink")
    ax2.tick_params(axis="y", colors="lightpink")

    sign_c = (c["predicted_sign"].to_numpy() == c["actual_sign"].to_numpy()).astype(float)
    x_c = np.abs(c["A_value"].to_numpy())
    _binned_curve(ax, x_c, sign_c, bins, color="steelblue", label="non-sampled c", linear=linear)
    sign_u = (u["predicted_sign"].to_numpy() == u["actual_sign"].to_numpy()).astype(float)
    x_u = np.abs(u["A_value"].to_numpy())
    _binned_curve(ax, x_u, sign_u, bins, color="darkorange", label="sampled u", linear=linear)
    _binned_curve(ax, x_all, sign_all, bins, color="black", label="merged (u + c)", linear=linear)

    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xscale("linear" if linear else "log")
    ax.set_xlabel("|A| = |log(π_T(u)/π_S(u))|  (signal strength)")
    ax.set_ylabel("sign agreement")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("(b) Sign agreement vs |A|")
    ax.legend(loc="lower right", fontsize=8)
    return pd.DataFrame()  # kept for return-value compatibility


def _loglog_panel(ax, x, y, *, color, xlabel, ylabel, title, linear=False):
    if linear:
        m = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        x, y = x[m], y[m]
        if len(x) < 5:
            ax.text(0.5, 0.5, "(too few)", ha="center", transform=ax.transAxes)
            ax.set_title(title); return
        ax.scatter(x, y, s=18, alpha=0.55, color=color, edgecolor="none")
        hi = float(max(x.max(), y.max()))
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="y = x (theory exact)")
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(0, hi, 50)
        ax.plot(xs, slope*xs + intercept, color="crimson", lw=1.5,
                label=f"fit  slope={slope:.2g}")
        spear = spearmanr(x, y)[0]
        pos = (x > 0)
        ratio = float(np.median(y[pos] / x[pos])) if pos.sum() else float("nan")
        ax.text(0.04, 0.96,
                f"n = {len(x)}\nSpearman = {spear:.3f}\n"
                f"slope = {slope:.2g}\nmedian(act/pred) = {ratio:.1f}×",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        return
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if len(x) < 5:
        ax.text(0.5, 0.5, "(too few)", ha="center", transform=ax.transAxes)
        ax.set_title(title); return
    lx, ly = np.log10(x), np.log10(y)
    ax.scatter(lx, ly, s=18, alpha=0.55, color=color, edgecolor="none")
    lo = float(min(lx.min(), ly.min()))
    hi = float(max(lx.max(), ly.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (theory exact)")
    slope, intercept = np.polyfit(lx, ly, 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope*xs + intercept, color="crimson", lw=1.5,
            label=f"fit  slope={slope:.2f}")
    spear = spearmanr(x, y)[0]
    pear = pearsonr(lx, ly)[0]
    bias = float(np.mean(ly - lx))
    ax.text(0.04, 0.96,
            f"n = {len(x)}\nSpearman = {spear:.3f}\nPearson(log) = {pear:.3f}\n"
            f"slope = {slope:.2f}\nbias = 10^{bias:+.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.set_xlabel(f"log10  {xlabel}")
    ax.set_ylabel(f"log10  {ylabel}")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)


def _signed_scatter_panel(ax, sub, *, color, xlabel, ylabel, title):
    """Signed scatter on symlog axes: x = predicted_sign · predicted_relative,
    y = actual Δπ.  Same quadrant = sign agrees.
    """
    pred_signed = (sub["predicted_sign"].to_numpy()
                   * sub["predicted_relative_magnitude"].to_numpy())
    actual_signed = sub["delta_pi_actual"].to_numpy()
    m = np.isfinite(pred_signed) & np.isfinite(actual_signed)
    pred_signed, actual_signed = pred_signed[m], actual_signed[m]
    if len(pred_signed) < 5:
        ax.text(0.5, 0.5, "(too few)", ha="center", transform=ax.transAxes)
        ax.set_title(title); return

    same_q = (np.sign(pred_signed) == np.sign(actual_signed)) \
             & (pred_signed != 0) & (actual_signed != 0)
    frac_same = float(same_q.mean())

    # Color by sign agreement: same quadrant = base color, different = red.
    colors = np.where(same_q, color, "crimson")
    ax.scatter(pred_signed, actual_signed, s=18, alpha=0.6,
               c=colors, edgecolor="none")
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)

    lim = float(np.abs(np.r_[pred_signed, actual_signed]).max())
    lim = max(lim, 1e-6)
    # y = x reference (theory exact: signed magnitude matches)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x (theory exact)")

    linthresh = max(1e-7,
                    np.percentile(np.abs(np.r_[pred_signed, actual_signed]), 5))
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlim(-lim * 1.3, lim * 1.3)
    ax.set_ylim(-lim * 1.3, lim * 1.3)

    # Stats on |·| log-log (so we can still report the magnitude relationship)
    ax_abs_x = np.abs(pred_signed); ax_abs_y = np.abs(actual_signed)
    mag_m = (ax_abs_x > 0) & (ax_abs_y > 0)
    spear = spearmanr(ax_abs_x[mag_m], ax_abs_y[mag_m])[0] \
            if mag_m.sum() >= 5 else float("nan")

    ax.text(0.04, 0.96,
            f"n = {len(pred_signed)}\nsame-quadrant = {frac_same:.2%}\n"
            f"|·| Spearman = {spear:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)


def make_combined(t: pd.DataFrame, out: Path, linear=False):
    u = t[t["is_sampled_u"]]
    c = t[~t["is_sampled_u"]]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # Row 1 — SIGN (binned-curve style mirroring update_consistency.pdf)
    if linear:
        dmax = float(np.abs(t["delta_pi_actual"].to_numpy()).max())
        delta_bins = np.linspace(0, max(dmax, 1e-9), 40)
    else:
        delta_bins = np.logspace(-10, -1, 40)
    _sign_overall(axes[0, 0], t, u, c, delta_bins, linear=linear)
    sign_tbl = _sign_vs_A(axes[0, 1], t, linear=linear)

    # (col 3) Overall sign agreement bar (chance baseline + merged + u + c)
    ax = axes[0, 2]
    cats = ["merged\n(u + c)", "sampled u", "non-sampled c"]
    sub_sets = [t, u, c]
    vals, ns = [], []
    for sub in sub_sets:
        m = (sub["predicted_sign"].to_numpy() != 0) & (sub["actual_sign"].to_numpy() != 0)
        if m.sum():
            vals.append(float((sub["predicted_sign"].to_numpy()[m]
                               == sub["actual_sign"].to_numpy()[m]).mean()))
        else:
            vals.append(float("nan"))
        ns.append(len(sub))
    bars = ax.bar(cats, vals, color=["black", "darkorange", "steelblue"])
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", label="chance = 0.5")
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                f"{v:.2%}\n(n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("sign agreement")
    ax.set_title("(c) Overall sign agreement")
    ax.legend(loc="lower right", fontsize=8)

    # Row 2 — MAGNITUDE (ABSOLUTE: lr·|A|·shape; y=x is "theory exact")
    _loglog_panel(axes[1, 0],
                  c["predicted_absolute_magnitude"].to_numpy(),
                  np.abs(c["delta_pi_actual"].to_numpy()),
                  color="steelblue",
                  xlabel="predicted_absolute  η·|A|·π_c·|τ−π_c|",
                  ylabel="actual |Δπ(c)|",
                  title="(d) Magnitude — non-sampled c", linear=linear)

    _loglog_panel(axes[1, 1],
                  u["predicted_absolute_magnitude"].to_numpy(),
                  np.abs(u["delta_pi_actual"].to_numpy()),
                  color="darkorange",
                  xlabel="predicted_absolute  η·|A|·π_u·(1+τ−π_u)",
                  ylabel="actual |Δπ(u)|",
                  title="(e) Magnitude — sampled u", linear=linear)

    # (col 3) Merged magnitude scatter — u + c overlaid, colored by group
    ax = axes[1, 2]
    x_c_ = c["predicted_absolute_magnitude"].to_numpy()
    y_c_ = np.abs(c["delta_pi_actual"].to_numpy())
    x_u_ = u["predicted_absolute_magnitude"].to_numpy()
    y_u_ = np.abs(u["delta_pi_actual"].to_numpy())
    if linear:
        mask_c = np.isfinite(x_c_) & np.isfinite(y_c_) & (x_c_ >= 0) & (y_c_ >= 0)
        mask_u = np.isfinite(x_u_) & np.isfinite(y_u_) & (x_u_ >= 0) & (y_u_ >= 0)
        tx = lambda v: v
    else:
        mask_c = np.isfinite(x_c_) & np.isfinite(y_c_) & (x_c_ > 0) & (y_c_ > 0)
        mask_u = np.isfinite(x_u_) & np.isfinite(y_u_) & (x_u_ > 0) & (y_u_ > 0)
        tx = np.log10
    if mask_c.sum() + mask_u.sum() >= 5:
        ax.scatter(tx(x_c_[mask_c]), tx(y_c_[mask_c]),
                   s=14, alpha=0.45, color="steelblue", edgecolor="none",
                   label=f"non-sampled c (n={mask_c.sum()})")
        ax.scatter(tx(x_u_[mask_u]), tx(y_u_[mask_u]),
                   s=18, alpha=0.7, color="darkorange", edgecolor="none",
                   label=f"sampled u (n={mask_u.sum()})")
        gx = np.concatenate([tx(x_c_[mask_c]), tx(x_u_[mask_u])])
        gy = np.concatenate([tx(y_c_[mask_c]), tx(y_u_[mask_u])])
        if linear:
            lo, hi = 0.0, float(max(gx.max(), gy.max()))
        else:
            lo, hi = float(min(gx.min(), gy.min())), float(max(gx.max(), gy.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        slope, intercept = np.polyfit(gx, gy, 1)
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, slope*xs + intercept, color="black", lw=1.5,
                label=f"merged fit slope={slope:.2g}")
        if linear:
            xr = x_c_[mask_c]; yr = y_c_[mask_c]
            allx = np.concatenate([x_c_[mask_c], x_u_[mask_u]])
            ally = np.concatenate([y_c_[mask_c], y_u_[mask_u]])
            pos = allx > 0
            ratio = float(np.median(ally[pos] / allx[pos])) if pos.sum() else float("nan")
            spear = spearmanr(allx, ally)[0]
            ax.text(0.04, 0.96,
                    f"merged n = {len(gx)}\nSpearman = {spear:.3f}\n"
                    f"slope = {slope:.2g}\nmedian(act/pred) = {ratio:.1f}×",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"))
        else:
            spear = spearmanr(10**gx, 10**gy)[0]
            pear = pearsonr(gx, gy)[0]
            bias = float(np.mean(gy - gx))
            ax.text(0.04, 0.96,
                    f"merged n = {len(gx)}\nSpearman = {spear:.3f}\nPearson(log) = {pear:.3f}\n"
                    f"slope = {slope:.2f}\nbias = 10^{bias:+.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"))
    pre = "" if linear else "log10  "
    ax.set_xlabel(f"{pre}predicted_absolute  η·|A|·π·shape(τ, π)")
    ax.set_ylabel(f"{pre}actual |Δπ|")
    ax.set_title("(f) Magnitude — merged (u + c)")
    ax.legend(loc="lower right", fontsize=7)

    scale_lbl = "linear" if linear else "log-log"
    plt.suptitle(
        f"n={len(t)} (u={len(u)}, c={len(c)})  •  "
        f"Row 1: SIGN (binned curves + bar)     Row 2: MAGNITUDE ({scale_lbl}, abs)",
        fontsize=12, y=1.00)
    fig.tight_layout()
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return sign_tbl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--delta-pi-floor", type=float, default=0.0,
                   help="Drop rows with |Δπ_actual| < floor (denoise bf16/NTK).")
    p.add_argument("--suffix", type=str, default="",
                   help="Suffix appended to output filenames.")
    p.add_argument("--name", type=str, default="",
                   help="Subfolder name under output-dir for this run's figures. "
                        "Defaults to the parquet's parent dir name.")
    p.add_argument("--linear", action="store_true",
                   help="Use linear axes for the magnitude/sign panels (no log scale).")
    args = p.parse_args()
    df = pd.read_parquet(args.measurements)
    # If there are no single-token-marked target rows, use the whole batch
    # (batch-update mode: every (i,j) contributes to the gradient).
    n_target = int(df["is_target"].sum())
    if n_target > 0:
        t = df[df["is_target"]].copy()
        print(f"Using is_target filter (single-token mode): {n_target} rows")
    else:
        t = df.copy()
        print("No is_target rows — using all batch rows (multi-token batch mode)")
    n_pre = len(t)
    if args.delta_pi_floor > 0:
        t = t[np.abs(t["delta_pi_actual"]) >= args.delta_pi_floor].copy()
    print(f"Total: {len(df)}, target: {n_pre}, "
          f"after |Δπ|≥{args.delta_pi_floor}: {len(t)} "
          f"(u={int((t['is_sampled_u']).sum())}, c={int((~t['is_sampled_u']).sum())})")

    # Each run's figures go in their own subfolder.
    run_name = args.name or args.measurements.parent.name
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fname = f"combined{'_linear' if args.linear else ''}{args.suffix}"
    sign_tbl = make_combined(t, run_dir / fname, linear=args.linear)
    sign_tbl.to_csv(run_dir / f"sign_vs_A_threshold{args.suffix}.csv", index=False)
    print(f"Wrote: {run_dir/fname}.png (+ .pdf)")


if __name__ == "__main__":
    main()
