"""Statistical analysis: within high init_FKL bins, which feature values are
significantly associated with WORSE learning (ΔFKL > 0)?

Approach:
  1. Restrict to init_FKL >= FKL_THRESHOLD.
  2. For each of 6 features, bin into 8 quantile buckets, compute:
     - P(WORSE | bucket)       — rate of ΔFKL >= 0.05
     - E[ΔFKL | bucket]        — mean signed ΔFKL
     - odds ratio vs baseline  — enrichment
     - z-score / p-value       — statistical significance
  3. Plot per-feature curves with error bars.
  4. Also: 2-feature interaction — for each pair of features (f_i, f_j) find
     the joint bucket with highest P(WORSE), report as rule.
"""
import os, glob, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

EPS = 1e-12
BASE = os.environ.get("BASE", "/fsx/xinnanzh/data/update_consistency_multi_pair")
N_STEPS = int(os.environ.get("N_STEPS", "10"))
FKL_THRESHOLD = float(os.environ.get("FKL_THRESHOLD", "3.0"))
WORSE_THRESHOLD = float(os.environ.get("WORSE_THRESHOLD", "0.05"))
N_QBINS = int(os.environ.get("N_QBINS", "8"))

RAW_COLS = [
    "sequence_id", "response_pos",
    "pi_S_before", "pi_S_after", "pi_T",
    "teacher_entropy", "overlap_ratio_k8",
    "A_value",
]

FEATURES = [
    ("S_ent_topk", "S_ent"),
    ("T_ent", "T_ent"),
    ("overlap_ratio_k8", "overlap"),
    ("S_mass_T_p95", "S_mass_T"),
    ("max_ps", "max_ps"),
    ("A_value", "A"),
]


def build(raw):
    ps = np.maximum(raw["pi_S_before"].values, EPS)
    pt = np.maximum(raw["pi_T"].values, EPS)
    psa = np.maximum(raw["pi_S_after"].values, EPS)
    r = raw.copy()
    r["_fkl_b"] = pt * np.log(pt / ps)
    r["_fkl_a"] = pt * np.log(pt / psa)
    r["_neg_ps_logps"] = -ps * np.log(ps)
    r["_ps"] = ps
    r["_pt"] = pt

    agg = r.groupby(["sequence_id", "response_pos"], as_index=False, sort=False).agg(
        init_fkl=("_fkl_b", "sum"),
        fkl_a=("_fkl_a", "sum"),
        S_ent_topk=("_neg_ps_logps", "sum"),
        max_ps=("_ps", "max"),
        T_ent=("teacher_entropy", "first"),
        overlap_ratio_k8=("overlap_ratio_k8", "first"),
        A_value=("A_value", "first"),
    )
    agg["delta_fkl"] = agg["fkl_a"] - agg["init_fkl"]

    r_sorted = r.sort_values(["sequence_id", "response_pos", "_pt"],
                             ascending=[True, True, False]).copy()
    r_sorted["_cumsum_pt"] = r_sorted.groupby(["sequence_id", "response_pos"])["_pt"].cumsum()
    r_sorted["_prev"] = r_sorted["_cumsum_pt"] - r_sorted["_pt"]
    r_sorted["_in_p95"] = (r_sorted["_prev"] < 0.95).astype(float)
    r_sorted["_ps_if"] = r_sorted["_ps"] * r_sorted["_in_p95"]
    smt = r_sorted.groupby(["sequence_id", "response_pos"], as_index=False, sort=False).agg(
        S_mass_T_p95=("_ps_if", "sum"))
    return agg.merge(smt, on=["sequence_id", "response_pos"], how="left")


def qcut_bins(x, n):
    e = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)))
    if len(e) < 3:
        e = np.linspace(x.min(), x.max() + EPS, n + 1)
    return e


def main():
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    pair_dirs = [d for d in pair_dirs if os.path.basename(d) != "figures"]
    parts = []
    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
        if not step_files: continue
        for f in step_files:
            parts.append(build(pd.read_parquet(f, columns=RAW_COLS)))
    all_df = pd.concat(parts, ignore_index=True)

    sub = all_df[all_df["init_fkl"] >= FKL_THRESHOLD].copy()
    print(f"Total pooled N = {len(all_df)}", flush=True)
    print(f"High-FKL subset (init_FKL >= {FKL_THRESHOLD}): N = {len(sub)}", flush=True)

    sub["is_worse"] = (sub["delta_fkl"] >= WORSE_THRESHOLD).astype(int)
    baseline_worse_rate = float(sub["is_worse"].mean())
    print(f"Baseline P(WORSE) = {baseline_worse_rate:.3%}\n")

    # =============== 1D analysis per feature ===============
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    feature_stats = {}

    for ax_i, (feat, flabel) in enumerate(FEATURES):
        ax = axes_flat[ax_i]
        bins = qcut_bins(sub[feat].values, N_QBINS)
        idx = np.clip(np.searchsorted(bins, sub[feat].values, side="right") - 1,
                       0, len(bins) - 2)
        centers = 0.5 * (bins[:-1] + bins[1:])

        p_worse = []
        n_bucket = []
        mean_dfkl = []
        z_scores = []
        odds_ratios = []
        for b in range(len(bins) - 1):
            m = idx == b
            n = int(m.sum())
            if n < 30:
                p_worse.append(np.nan); n_bucket.append(n)
                mean_dfkl.append(np.nan); z_scores.append(np.nan); odds_ratios.append(np.nan); continue
            k = int(sub["is_worse"].values[m].sum())
            p = k / n
            p_worse.append(p)
            n_bucket.append(n)
            mean_dfkl.append(float(sub["delta_fkl"].values[m].mean()))
            # Z-test on proportion vs baseline
            se = np.sqrt(baseline_worse_rate * (1 - baseline_worse_rate) / n)
            z = (p - baseline_worse_rate) / max(se, 1e-9)
            z_scores.append(z)
            # Odds ratio
            p0 = baseline_worse_rate
            odds_ratios.append((p / (1 - p)) / (p0 / (1 - p0)))

        p_worse = np.array(p_worse)
        z_scores = np.array(z_scores)
        odds_ratios = np.array(odds_ratios)

        # Two-axis plot: bars = P(WORSE), line = mean ΔFKL
        valid = ~np.isnan(p_worse)
        ax2 = ax.twinx()
        bar_colors = ["#DC2626" if z_scores[i] > 2 else
                      ("#f59e0b" if z_scores[i] > 1 else
                       ("#93c5fd" if z_scores[i] > -1 else
                        ("#3b82f6" if z_scores[i] > -2 else "#1e40af")))
                      for i in range(len(z_scores))]
        ax.bar(np.arange(len(bins) - 1), np.where(valid, p_worse, np.nan),
               color=bar_colors, alpha=0.85, edgecolor="black", lw=0.4)
        ax.axhline(baseline_worse_rate, color="black", ls="--", lw=0.8,
                   label=f"baseline={baseline_worse_rate:.2%}")

        # Line for mean ΔFKL on secondary axis
        ax2.plot(np.arange(len(bins) - 1), mean_dfkl, "o-",
                 color="#374151", lw=1.5, markersize=5)
        ax2.axhline(0, color="black", ls=":", lw=0.5)
        ax2.set_ylabel("mean ΔFKL", fontsize=9)

        # Annotate n and z per bar
        for i in range(len(bins) - 1):
            if not valid[i]: continue
            ax.text(i, p_worse[i] + 0.005, f"n={n_bucket[i]}\nz={z_scores[i]:+.1f}",
                    ha="center", va="bottom", fontsize=6)

        ax.set_xticks(np.arange(len(bins) - 1))
        ax.set_xticklabels([f"[{bins[i]:.2g},\n{bins[i+1]:.2g}]"
                            for i in range(len(bins) - 1)],
                           fontsize=6, rotation=0)
        ax.set_xlabel(f"{flabel} quantile bucket", fontsize=9)
        ax.set_ylabel("P(ΔFKL > 0.05)", fontsize=9)
        ax.set_title(f"{flabel}: P(WORSE) by bucket\n"
                     f"(bar color: red=z>2 enriched, blue=z<-2 depleted)",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(p_worse[valid].max() * 1.25, 0.3))
        ax.legend(fontsize=7, loc="upper left")

        feature_stats[feat] = {
            "bins": bins, "p_worse": p_worse, "z": z_scores,
            "odds_ratio": odds_ratios, "n": n_bucket, "mean_dfkl": mean_dfkl,
        }

    fig.suptitle(f"WHICH FEATURE VALUES ENRICH WORSE LEARNING? (init_FKL ≥ {FKL_THRESHOLD}, N={len(sub)})\n"
                 f"Bars=P(WORSE), colored by z-score vs baseline. Line=mean ΔFKL.",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out1 = f"{out_dir}/worse_predictors_1d.png"
    plt.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    # =============== 2D interaction: top-k enriched buckets ===============
    feat_pairs = list(itertools.combinations(range(len(FEATURES)), 2))
    rows_all = []

    for i, j in feat_pairs:
        fi, li = FEATURES[i]
        fj, lj = FEATURES[j]
        bi = qcut_bins(sub[fi].values, N_QBINS)
        bj = qcut_bins(sub[fj].values, N_QBINS)
        idi = np.clip(np.searchsorted(bi, sub[fi].values, side="right") - 1, 0, len(bi) - 2)
        idj = np.clip(np.searchsorted(bj, sub[fj].values, side="right") - 1, 0, len(bj) - 2)
        for xa in range(len(bi) - 1):
            for xb in range(len(bj) - 1):
                m = (idi == xa) & (idj == xb)
                n = int(m.sum())
                if n < 50: continue
                k = int(sub["is_worse"].values[m].sum())
                p = k / n
                mean_dfkl = float(sub["delta_fkl"].values[m].mean())
                se = np.sqrt(baseline_worse_rate * (1 - baseline_worse_rate) / n)
                z = (p - baseline_worse_rate) / max(se, 1e-9)
                rows_all.append({
                    "feat_pair": f"{li}×{lj}",
                    "condition": f"{li}∈[{bi[xa]:.2g},{bi[xa+1]:.2g}] & {lj}∈[{bj[xb]:.2g},{bj[xb+1]:.2g}]",
                    "n": n, "p_worse": p, "z": z,
                    "odds_ratio": (p / max(1 - p, 1e-9)) / (baseline_worse_rate / max(1 - baseline_worse_rate, 1e-9)),
                    "mean_dfkl": mean_dfkl,
                })
    rules_df = pd.DataFrame(rows_all)

    # Save top-K enriched (highest z) and top-K depleted (lowest z)
    top_enriched = rules_df.sort_values("z", ascending=False).head(20)
    top_depleted = rules_df.sort_values("z", ascending=True).head(20)

    with open(f"{out_dir}/worse_predictor_rules.txt", "w") as fh:
        fh.write(f"Restricted to init_FKL >= {FKL_THRESHOLD}, N={len(sub)}\n")
        fh.write(f"WORSE = ΔFKL >= {WORSE_THRESHOLD}\n")
        fh.write(f"Baseline P(WORSE) = {baseline_worse_rate:.3%}\n\n")
        fh.write("=" * 120 + "\n")
        fh.write("TOP 20 BUCKETS MOST ENRICHED FOR WORSE (highest z-score)\n")
        fh.write("=" * 120 + "\n")
        for _, r in top_enriched.iterrows():
            fh.write(f"[{r['feat_pair']:15s}] {r['condition']:60s}  "
                     f"n={r['n']:5d}  P(WORSE)={r['p_worse']:.2%}  z={r['z']:+.1f}  "
                     f"OR={r['odds_ratio']:.2f}  ΔFKL={r['mean_dfkl']:+.3f}\n")
        fh.write("\n" + "=" * 120 + "\n")
        fh.write("TOP 20 BUCKETS MOST DEPLETED FOR WORSE (lowest z-score, i.e. safest)\n")
        fh.write("=" * 120 + "\n")
        for _, r in top_depleted.iterrows():
            fh.write(f"[{r['feat_pair']:15s}] {r['condition']:60s}  "
                     f"n={r['n']:5d}  P(WORSE)={r['p_worse']:.2%}  z={r['z']:+.1f}  "
                     f"OR={r['odds_ratio']:.2f}  ΔFKL={r['mean_dfkl']:+.3f}\n")
    print(f"Saved {out_dir}/worse_predictor_rules.txt")

    print("\nTOP 10 ENRICHED FOR WORSE:")
    for _, r in top_enriched.head(10).iterrows():
        print(f"  [{r['feat_pair']:15s}] {r['condition']:60s}  P(WORSE)={r['p_worse']:.2%}  z={r['z']:+.1f}  OR={r['odds_ratio']:.2f}")

    # =============== Figure 2: 15-panel enrichment maps ===============
    fig2, axes2 = plt.subplots(3, 5, figsize=(4 * 5, 3.5 * 3), squeeze=False)
    for fp_i, (i, j) in enumerate(feat_pairs):
        ax = axes2[fp_i // 5, fp_i % 5]
        fi, li = FEATURES[i]; fj, lj = FEATURES[j]
        bi = qcut_bins(sub[fi].values, N_QBINS)
        bj = qcut_bins(sub[fj].values, N_QBINS)
        idi = np.clip(np.searchsorted(bi, sub[fi].values, side="right") - 1, 0, len(bi) - 2)
        idj = np.clip(np.searchsorted(bj, sub[fj].values, side="right") - 1, 0, len(bj) - 2)
        Z = np.full((len(bj) - 1, len(bi) - 1), np.nan)
        for xa in range(len(bi) - 1):
            for xb in range(len(bj) - 1):
                m = (idi == xa) & (idj == xb)
                n = int(m.sum())
                if n < 50: continue
                p = float(sub["is_worse"].values[m].mean())
                se = np.sqrt(baseline_worse_rate * (1 - baseline_worse_rate) / n)
                Z[xb, xa] = (p - baseline_worse_rate) / max(se, 1e-9)
        vmax = max(4, float(np.nanpercentile(np.abs(Z), 95)))
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        ax.set_xlabel(li, fontsize=8)
        ax.set_ylabel(lj, fontsize=8)
        ax.set_title(f"{li}×{lj}", fontsize=9, fontweight="bold")
        ax.set_xticks([0, len(bi) - 2])
        ax.set_xticklabels([f"{bi[0]:.2g}", f"{bi[-1]:.2g}"], fontsize=6)
        ax.set_yticks([0, len(bj) - 2])
        ax.set_yticklabels([f"{bj[0]:.2g}", f"{bj[-1]:.2g}"], fontsize=6)

    fig2.suptitle(f"z-score enrichment for WORSE across joint feature buckets "
                  f"(init_FKL ≥ {FKL_THRESHOLD}, baseline P(WORSE)={baseline_worse_rate:.2%})\n"
                  "red=significantly enriched, blue=depleted",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out2 = f"{out_dir}/worse_predictors_2d_zscore.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
