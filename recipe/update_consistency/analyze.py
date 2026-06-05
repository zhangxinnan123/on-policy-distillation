# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Stage 4 — offline analysis. Reads measurements.parquet and produces the
2x3 paper figure plus a CSV summary.

  python -m recipe.update_consistency.analyze \\
      --measurements data/measurements/measurements.parquet \\
      --output-dir figures/update_consistency
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sign agreement + magnitude correlation, stratified by all dimensions."""
    rows = []

    def _agg(sub, label):
        if len(sub) == 0:
            return
        agree = (sub["predicted_sign"] == sub["actual_sign"]).mean()
        # Magnitude correlation: predicted_relative_magnitude vs |delta_pi_actual|
        pred = sub["predicted_relative_magnitude"].to_numpy()
        actual = np.abs(sub["delta_pi_actual"].to_numpy())
        # filter out exact-zero predictions and NaNs for correlation
        mask = np.isfinite(pred) & np.isfinite(actual) & (pred > 0)
        if mask.sum() >= 10:
            pear = pearsonr(np.log(pred[mask] + 1e-30), np.log(actual[mask] + 1e-30))[0]
            spear = spearmanr(pred[mask], actual[mask])[0]
        else:
            pear, spear = np.nan, np.nan
        rows.append(
            {"stratum": label, "n": len(sub), "sign_agreement": agree,
             "pearson_log_magnitude": pear, "spearman_magnitude": spear}
        )

    _agg(df, "all")
    _agg(df[df["is_sampled_u"]], "sampled_u")
    _agg(df[~df["is_sampled_u"]], "non_sampled")
    _agg(df[df["tau"] > 0], "tau_positive (productive)")
    _agg(df[df["tau"] <= 0], "tau_nonpositive (mode_sampling)")
    nonu = df[~df["is_sampled_u"]]
    _agg(nonu[nonu["is_above_watershed"]], "non_sampled & above_watershed")
    _agg(nonu[~nonu["is_above_watershed"]], "non_sampled & below_watershed")
    return pd.DataFrame(rows)


def _per_step(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for step, sub in df.groupby("step"):
        agree = (sub["predicted_sign"] == sub["actual_sign"]).mean()
        pred = sub["predicted_relative_magnitude"].to_numpy()
        actual = np.abs(sub["delta_pi_actual"].to_numpy())
        mask = np.isfinite(pred) & np.isfinite(actual) & (pred > 0)
        spear = spearmanr(pred[mask], actual[mask])[0] if mask.sum() >= 10 else np.nan
        out.append({"step": step, "sign_agreement": agree, "spearman_magnitude": spear, "n": len(sub)})
    return pd.DataFrame(out)


def _figure(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    log_bins = np.logspace(-9, -2, 40)

    def _binned_metric(x, y, bins, op):
        idx = np.digitize(x, bins) - 1
        centers, vals, counts = [], [], []
        for i in range(len(bins) - 1):
            mask = idx == i
            n = int(mask.sum())
            counts.append(n)
            centers.append(np.sqrt(bins[i] * bins[i + 1]))
            vals.append(op(y[mask]) if n > 0 else np.nan)
        return np.array(centers), np.array(vals), np.array(counts)

    def _draw(ax, x, y, bins, y_label, title, op=np.nanmean):
        c, v, n = _binned_metric(x, y, bins, op)
        ax2 = ax.twinx()
        ax2.bar(c, n, width=np.diff(np.r_[bins[0]*0.9, c]) * 0.7, color="lightpink", alpha=0.5, log=True)
        ax2.set_ylabel("token count", color="lightpink")
        ax.plot(c, v, "o-", color="steelblue")
        ax.set_xscale("log")
        ax.set_xlabel("")
        ax.set_ylabel(y_label, color="steelblue")
        ax.set_title(title)

    sign_agree = (df["predicted_sign"] == df["actual_sign"]).astype(float).to_numpy()
    pred_mag = df["predicted_relative_magnitude"].to_numpy()
    pi_S = df["pi_S_before"].to_numpy()
    tau = df["tau"].to_numpy()
    actual_abs = np.abs(df["delta_pi_actual"].to_numpy())
    pi_T = df["pi_T"].to_numpy()

    # Row 1: direction consistency
    _draw(axes[0, 0], pred_mag, sign_agree, log_bins,
          "sign agreement", "vs predicted relative magnitude")
    _draw(axes[0, 1], pi_S, sign_agree, log_bins,
          "sign agreement", "vs π_S(c)")
    # Stratify by tau regime — bins on tau magnitude
    tau_bins = np.logspace(-9, -1, 30)
    _draw(axes[0, 2], np.abs(tau), sign_agree, tau_bins,
          "sign agreement", "vs |τ|")

    # Row 2: magnitude consistency
    _draw(axes[1, 0], pred_mag, actual_abs, log_bins,
          "mean |Δπ|", "vs predicted relative magnitude")
    _draw(axes[1, 1], pi_S, actual_abs, log_bins,
          "mean |Δπ|", "vs π_S(c)")
    _draw(axes[1, 2], np.abs(pi_T - pi_S), actual_abs, log_bins,
          "mean |Δπ|", "vs |π_T − π_S| (naive baseline)")

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.measurements)
    print(f"Loaded {len(df)} measurement rows from {args.measurements}")
    print(f"Steps: {sorted(df['step'].unique())[:10]} ...")

    # Overall numbers
    print("\n=== Stratified summary ===")
    summary = _summary_table(df)
    print(summary.to_string(index=False))
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    print("\n=== Per-step summary ===")
    per_step = _per_step(df)
    print(per_step.to_string(index=False))
    per_step.to_csv(args.output_dir / "per_step.csv", index=False)

    # Outliers (non-sampled candidates where predicted_sign disagrees with actual_sign)
    nonu = df[~df["is_sampled_u"]].copy()
    bad = nonu[nonu["predicted_sign"] != nonu["actual_sign"]].copy()
    bad["abs_delta"] = np.abs(bad["delta_pi_actual"])
    top10 = bad.nlargest(10, "abs_delta")
    print("\n=== Top-10 sign-disagreement outliers (non-sampled candidates) ===")
    print(top10[["step", "sequence_id", "response_pos", "candidate_token_id",
                 "pi_S_before", "pi_S_after",
                 "tau", "A_value", "delta_pi_actual",
                 "predicted_sign", "actual_sign"]].to_string(index=False))

    out_pdf = args.output_dir / "update_consistency.pdf"
    out_png = args.output_dir / "update_consistency.png"
    _figure(df, out_pdf, out_png)
    print(f"\nFigure: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
