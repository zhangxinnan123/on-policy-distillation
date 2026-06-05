# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""B=1 single-token full analysis — sign + amplitude in one figure.

  python -m recipe.update_consistency.analyze_b1_full \\
      --measurements .../run_update_consistency_singletoken_b1_lr1e3_50steps_1.7b_8b/measurements.parquet \\
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


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-30, None))


def _sign_agreement(pred: np.ndarray, actual: np.ndarray) -> float:
    m = (pred != 0) & (actual != 0)
    if m.sum() == 0:
        return float("nan")
    return float((pred[m] == actual[m]).mean())


def _hexbin_loglog(ax, x, y, *, xlabel, ylabel, title, color="steelblue"):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", transform=ax.transAxes)
        ax.set_title(title)
        return None
    lx, ly = np.log10(x), np.log10(y)
    ax.scatter(lx, ly, s=14, alpha=0.55, color=color, edgecolor="none")

    # y = x reference
    lo = float(min(lx.min(), ly.min()))
    hi = float(max(lx.max(), ly.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (theory exact)")

    # OLS linear fit on log-log
    slope, intercept = np.polyfit(lx, ly, 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.5,
            label=f"fit slope={slope:.2f}")

    spear = spearmanr(x, y)[0]
    pear = pearsonr(lx, ly)[0]
    bias = float(np.mean(ly - lx))  # mean log10(actual/predicted)
    ax.text(0.04, 0.96,
            f"n = {len(x)}\nSpearman = {spear:.2f}\nPearson(log) = {pear:.2f}\n"
            f"mean log10(act/pred) = {bias:+.2f}  (×10^{bias:.1f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7)
    return slope, intercept, spear, pear, bias


def _sign_vs_threshold(ax, df: pd.DataFrame):
    """Bar plot: sign agreement vs |A| threshold, split by sampled / non-sampled."""
    thresholds = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    rows = []
    for thr in thresholds:
        sub = df[df["A_value"].abs() >= thr]
        u = sub[sub["is_sampled_u"]]
        c = sub[~sub["is_sampled_u"]]
        rows.append({
            "thr": thr,
            "n_u": len(u),
            "n_c": len(c),
            "agree_u": _sign_agreement(u["predicted_sign"].to_numpy(),
                                       u["actual_sign"].to_numpy()),
            "agree_c": _sign_agreement(c["predicted_sign"].to_numpy(),
                                       c["actual_sign"].to_numpy()),
        })
    tbl = pd.DataFrame(rows)
    x = np.arange(len(thresholds))
    w = 0.4
    ax.bar(x - w/2, tbl["agree_c"], w, color="steelblue",
           label=f"non-sampled c (n@0={tbl.n_c.iloc[0]})")
    ax.bar(x + w/2, tbl["agree_u"], w, color="darkorange",
           label=f"sampled u (n@0={tbl.n_u.iloc[0]})")
    for i, r in tbl.iterrows():
        ax.text(i - w/2, r["agree_c"] + 0.01, f"{int(r['n_c'])}",
                ha="center", fontsize=7, color="steelblue")
        ax.text(i + w/2, r["agree_u"] + 0.01, f"{int(r['n_u'])}",
                ha="center", fontsize=7, color="darkorange")
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{t}" for t in thresholds])
    ax.set_xlabel("|A| threshold")
    ax.set_ylabel("sign agreement (π-level)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sign agreement vs |A| strength")
    ax.legend(loc="lower right", fontsize=8)
    return tbl


def _signed_scatter(ax, df_sub: pd.DataFrame, title: str):
    """Signed Δπ scatter: predicted_sign · |predicted_relative| vs actual Δπ."""
    pred_signed = df_sub["predicted_sign"].to_numpy() * \
                  df_sub["predicted_relative_magnitude"].to_numpy()
    actual_signed = df_sub["delta_pi_actual"].to_numpy()
    m = np.isfinite(pred_signed) & np.isfinite(actual_signed)
    pred_signed, actual_signed = pred_signed[m], actual_signed[m]
    if len(pred_signed) == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    # symlog scatter (signed)
    ax.scatter(pred_signed, actual_signed, s=14, alpha=0.55,
               color="purple", edgecolor="none")
    ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
    lim = max(np.abs(np.r_[pred_signed, actual_signed]).max(), 1e-6)
    # quadrant agreement
    same = (np.sign(pred_signed) == np.sign(actual_signed)) & (pred_signed != 0) & (actual_signed != 0)
    frac = same.mean() if len(same) else float("nan")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlim(-lim*1.2, lim*1.2)
    ax.set_ylim(-lim*1.2, lim*1.2)
    ax.set_xlabel("predicted_sign · predicted_relative_magnitude")
    ax.set_ylabel("actual Δπ (signed)")
    ax.set_title(f"{title}  (same-quadrant frac: {frac:.2%})")


def make_figure(df: pd.DataFrame, out_path: Path):
    # target rows only
    t = df[df["is_target"]].copy()
    u = t[t["is_sampled_u"]]
    c = t[~t["is_sampled_u"]]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # ------- ROW 1: SIGN -------
    # (0,0) Sign agreement bar — overall + sampled / non-sampled, no threshold
    ax = axes[0, 0]
    cats = ["all\ntarget", "sampled u", "non-sampled c"]
    vals = [_sign_agreement(t["predicted_sign"].to_numpy(),
                            t["actual_sign"].to_numpy()),
            _sign_agreement(u["predicted_sign"].to_numpy(),
                            u["actual_sign"].to_numpy()),
            _sign_agreement(c["predicted_sign"].to_numpy(),
                            c["actual_sign"].to_numpy())]
    ns = [len(t), len(u), len(c)]
    bars = ax.bar(cats, vals,
                  color=["dimgray", "darkorange", "steelblue"])
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", label="chance")
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                f"{v:.2%}\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("sign agreement (π-level)")
    ax.set_title("(a) Overall sign agreement at target tokens")
    ax.legend(loc="lower right", fontsize=8)

    # (0,1) Sign agreement vs |A| threshold
    tbl = _sign_vs_threshold(axes[0, 1], t)

    # (0,2) Signed scatter — sampled u
    _signed_scatter(axes[0, 2], u, "(c) Signed Δπ — sampled u (target)")

    # ------- ROW 2: AMPLITUDE -------
    # (1,0) |Δπ| actual vs predicted_relative — non-sampled c
    _hexbin_loglog(axes[1, 0],
                   c["predicted_relative_magnitude"].to_numpy(),
                   np.abs(c["delta_pi_actual"].to_numpy()),
                   xlabel="predicted_relative_magnitude  π_c·|τ − π_c|",
                   ylabel="actual |Δπ(c)|",
                   title="(d) Magnitude — non-sampled c",
                   color="steelblue")

    # (1,1) |Δπ| actual vs predicted_relative — sampled u
    _hexbin_loglog(axes[1, 1],
                   u["predicted_relative_magnitude"].to_numpy(),
                   np.abs(u["delta_pi_actual"].to_numpy()),
                   xlabel="predicted_relative_magnitude  π_u·(1+τ−π_u)",
                   ylabel="actual |Δπ(u)|",
                   title="(e) Magnitude — sampled u",
                   color="darkorange")

    # (1,2) |Δπ| actual vs predicted ABSOLUTE (lr · |A| · shape)
    # combine sampled u & non-sampled c for one panel showing the bias
    pred_abs = t["predicted_absolute_magnitude"].to_numpy()
    actual_abs = np.abs(t["delta_pi_actual"].to_numpy())
    _hexbin_loglog(axes[1, 2], pred_abs, actual_abs,
                   xlabel="predicted_absolute_magnitude  lr·|A|·shape",
                   ylabel="actual |Δπ|",
                   title="(f) Absolute magnitude (all target)",
                   color="purple")

    plt.suptitle(
        "B=1 single-token full analysis (50 steps, lr=1e-3, SGD)  —  "
        f"target rows: {len(t)} (u: {len(u)}, c: {len(c)})",
        fontsize=12, y=1.00)
    plt.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return tbl


def print_summary(df: pd.DataFrame):
    t = df[df["is_target"]].copy()
    u = t[t["is_sampled_u"]]
    c = t[~t["is_sampled_u"]]

    print(f"\n=== B=1 target rows: {len(t)} (sampled u={len(u)}, non-sampled c={len(c)})\n")

    print("--- SIGN AGREEMENT (no threshold) ---")
    for name, sub in [("all target", t), ("sampled u", u), ("non-sampled c", c)]:
        v = _sign_agreement(sub["predicted_sign"].to_numpy(),
                            sub["actual_sign"].to_numpy())
        print(f"  {name:<18s}: {v:.4f}  (n={len(sub)})")

    print("\n--- SIGN AGREEMENT vs |A| (stronger signal -> tabular theory cleaner) ---")
    for thr in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
        sub = t[t["A_value"].abs() >= thr]
        su = sub[sub["is_sampled_u"]]
        sc = sub[~sub["is_sampled_u"]]
        au = _sign_agreement(su["predicted_sign"].to_numpy(),
                             su["actual_sign"].to_numpy())
        ac = _sign_agreement(sc["predicted_sign"].to_numpy(),
                             sc["actual_sign"].to_numpy())
        print(f"  |A|>={thr:<4}  n_u={len(su):<4d}  n_c={len(sc):<4d}  agree_u={au:.3f}  agree_c={ac:.3f}")

    print("\n--- AMPLITUDE (log-log Pearson / Spearman + multiplicative bias) ---")
    for name, sub in [("non-sampled c", c), ("sampled u", u)]:
        pred = sub["predicted_relative_magnitude"].to_numpy()
        actual = np.abs(sub["delta_pi_actual"].to_numpy())
        m = np.isfinite(pred) & np.isfinite(actual) & (pred > 0) & (actual > 0)
        if m.sum() < 5:
            print(f"  {name:<18s}: too few"); continue
        spear = spearmanr(pred[m], actual[m])[0]
        pear = pearsonr(np.log10(pred[m]), np.log10(actual[m]))[0]
        slope, intercept = np.polyfit(np.log10(pred[m]), np.log10(actual[m]), 1)
        bias = float(np.mean(np.log10(actual[m]) - np.log10(pred[m])))
        print(f"  {name:<18s}: Spearman={spear:.3f}  Pearson(log)={pear:.3f}  "
              f"slope={slope:.2f}  multiplicative bias = 10^{bias:+.2f}")

    print("\n--- ABSOLUTE MAGNITUDE (predicted = lr·|A|·shape) ---")
    pred = t["predicted_absolute_magnitude"].to_numpy()
    actual = np.abs(t["delta_pi_actual"].to_numpy())
    m = np.isfinite(pred) & np.isfinite(actual) & (pred > 0) & (actual > 0)
    if m.sum() >= 5:
        spear = spearmanr(pred[m], actual[m])[0]
        pear = pearsonr(np.log10(pred[m]), np.log10(actual[m]))[0]
        slope, intercept = np.polyfit(np.log10(pred[m]), np.log10(actual[m]), 1)
        bias = float(np.mean(np.log10(actual[m]) - np.log10(pred[m])))
        print(f"  all target        : Spearman={spear:.3f}  Pearson(log)={pear:.3f}  "
              f"slope={slope:.2f}  multiplicative bias = 10^{bias:+.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.measurements)
    print(f"Loaded {len(df)} rows")
    print_summary(df)
    out = args.output_dir / "b1_full"
    tbl = make_figure(df, out)
    tbl.to_csv(args.output_dir / "b1_sign_vs_A_threshold.csv", index=False)
    print(f"\nFigure: {out}.png / {out}.pdf")
    print(f"CSV:    {args.output_dir / 'b1_sign_vs_A_threshold.csv'}")


if __name__ == "__main__":
    main()
