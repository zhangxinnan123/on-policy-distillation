# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""B=1 single-token — two separate figures, target-only.

Figure 1: sign agreement.
Figure 2: amplitude (log-log scatter).

  python -m recipe.update_consistency.analyze_b1_two_figs \\
      --measurements .../measurements.parquet \\
      --output-dir   recipe/update_consistency/figures
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


# ---------------------------------------------------------------------- SIGN
def fig_sign(t: pd.DataFrame, out: Path):
    u = t[t["is_sampled_u"]]
    c = t[~t["is_sampled_u"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) overall sign agreement bar
    ax = axes[0]
    cats = ["all target", "sampled u", "non-sampled c"]
    vals = [_sign_agreement(t["predicted_sign"].to_numpy(),
                            t["actual_sign"].to_numpy()),
            _sign_agreement(u["predicted_sign"].to_numpy(),
                            u["actual_sign"].to_numpy()),
            _sign_agreement(c["predicted_sign"].to_numpy(),
                            c["actual_sign"].to_numpy())]
    ns = [len(t), len(u), len(c)]
    bars = ax.bar(cats, vals,
                  color=["dimgray", "darkorange", "steelblue"])
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", label="chance = 0.5")
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                f"{v:.2%}\n(n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("sign agreement")
    ax.set_title("(a) Overall sign agreement")
    ax.legend(loc="lower right", fontsize=8)

    # (b) sign agreement vs |A| threshold
    ax = axes[1]
    thresholds = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    rows = []
    for thr in thresholds:
        sub = t[t["A_value"].abs() >= thr]
        su = sub[sub["is_sampled_u"]]
        sc = sub[~sub["is_sampled_u"]]
        rows.append({
            "thr": thr,
            "n_u": len(su), "n_c": len(sc),
            "agree_u": _sign_agreement(su["predicted_sign"].to_numpy(),
                                       su["actual_sign"].to_numpy()),
            "agree_c": _sign_agreement(sc["predicted_sign"].to_numpy(),
                                       sc["actual_sign"].to_numpy()),
        })
    tbl = pd.DataFrame(rows)
    x = np.arange(len(thresholds)); w = 0.4
    ax.bar(x - w/2, tbl["agree_c"], w, color="steelblue", label="non-sampled c")
    ax.bar(x + w/2, tbl["agree_u"], w, color="darkorange", label="sampled u")
    for i, r in tbl.iterrows():
        if r["n_c"] > 0:
            ax.text(i - w/2, r["agree_c"] + 0.012, f"{int(r['n_c'])}",
                    ha="center", fontsize=7, color="steelblue")
        if r["n_u"] > 0:
            ax.text(i + w/2, r["agree_u"] + 0.012, f"{int(r['n_u'])}",
                    ha="center", fontsize=7, color="darkorange")
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{t_}" for t_ in thresholds])
    ax.set_xlabel("|A| threshold (filter out weak-signal rows)")
    ax.set_ylabel("sign agreement")
    ax.set_ylim(0, 1.1)
    ax.set_title("(b) Sign agreement vs |A| strength")
    ax.legend(loc="lower right", fontsize=8)

    plt.suptitle(
        f"B=1 single-token  •  SIGN  •  target rows (n={len(t)}: u={len(u)}, c={len(c)})",
        fontsize=12, y=1.00)
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return tbl


# ---------------------------------------------------------------------- AMPLITUDE
def _loglog_panel(ax, x, y, *, color, xlabel, ylabel, title):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if len(x) < 5:
        ax.text(0.5, 0.5, "(too few)", ha="center", transform=ax.transAxes)
        ax.set_title(title)
        return
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


def fig_magnitude(t: pd.DataFrame, out: Path):
    u = t[t["is_sampled_u"]]
    c = t[~t["is_sampled_u"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    _loglog_panel(axes[0],
                  c["predicted_relative_magnitude"].to_numpy(),
                  np.abs(c["delta_pi_actual"].to_numpy()),
                  color="steelblue",
                  xlabel="predicted_relative  π_c · |τ − π_c|",
                  ylabel="actual |Δπ(c)|",
                  title="(a) Non-sampled c")

    _loglog_panel(axes[1],
                  u["predicted_relative_magnitude"].to_numpy(),
                  np.abs(u["delta_pi_actual"].to_numpy()),
                  color="darkorange",
                  xlabel="predicted_relative  π_u · (1 + τ − π_u)",
                  ylabel="actual |Δπ(u)|",
                  title="(b) Sampled u")

    plt.suptitle(
        f"B=1 single-token  •  MAGNITUDE  •  target rows (n={len(t)}: u={len(u)}, c={len(c)})",
        fontsize=12, y=1.00)
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.measurements)
    t = df[df["is_target"]].copy()
    print(f"Total: {len(df)}, target: {len(t)}, "
          f"sampled u: {(t['is_sampled_u']).sum()}, "
          f"non-sampled c: {(~t['is_sampled_u']).sum()}")
    sign_tbl = fig_sign(t, args.output_dir / "b1_sign")
    fig_magnitude(t, args.output_dir / "b1_magnitude")
    sign_tbl.to_csv(args.output_dir / "b1_sign_vs_A_threshold.csv", index=False)
    print("Wrote:")
    print(f"  {args.output_dir/'b1_sign.png'} (+ .pdf)")
    print(f"  {args.output_dir/'b1_magnitude.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
