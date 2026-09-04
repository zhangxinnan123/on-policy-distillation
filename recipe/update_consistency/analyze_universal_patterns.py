"""Cross-pair analysis: which feature combinations universally predict STUCK/WORSE?

For each of 9 pairs, compute the STUCK-rate and WORSE-rate in each 3D bin of
(feature_1, feature_2, feature_3). Then look for bins where:
  - All pairs agree on high STUCK/WORSE rate → "universal stuck" pattern
  - Only 1-2 pairs affected → "pair-specific" pattern

Simpler visualization: pick top-3 features (max_ps, S_mass_T_p95, |A|), bin each
into 3 levels (low/mid/high), and for each of the 27 combinations plot mean
STUCK-rate and WORSE-rate across pairs (as a heatmap grid).
"""
import os, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
BASE = os.environ.get("BASE", "/fsx/xinnanzh/data/update_consistency_multi_pair")
N_STEPS = int(os.environ.get("N_STEPS", "10"))
FKL_THRESHOLD = float(os.environ.get("FKL_THRESHOLD", "2.0"))

RAW_COLS = [
    "sequence_id", "response_pos",
    "pi_S_before", "pi_S_after", "pi_T",
    "teacher_entropy", "overlap_ratio_k8",
    "teacher_mass_on_student_topp95",
    "A_value",
]


def build(raw):
    ps = np.maximum(raw["pi_S_before"].values, EPS)
    pt = np.maximum(raw["pi_T"].values, EPS)
    psa = np.maximum(raw["pi_S_after"].values, EPS)
    r = raw.copy()
    r["_fkl_b"] = pt * np.log(pt / ps)
    r["_fkl_a"] = pt * np.log(pt / psa)
    r["_ps"] = ps
    r["_pt"] = pt

    agg = r.groupby(["sequence_id", "response_pos"], as_index=False, sort=False).agg(
        init_fkl=("_fkl_b", "sum"),
        fkl_a=("_fkl_a", "sum"),
        max_ps=("_ps", "max"),
        teacher_entropy=("teacher_entropy", "first"),
        A_value=("A_value", "first"),
    )
    agg["delta_fkl"] = agg["fkl_a"] - agg["init_fkl"]
    agg["abs_A"] = agg["A_value"].abs()

    r_sorted = r.sort_values(["sequence_id", "response_pos", "_pt"],
                             ascending=[True, True, False]).copy()
    r_sorted["_cumsum_pt"] = r_sorted.groupby(["sequence_id", "response_pos"])["_pt"].cumsum()
    r_sorted["_prev"] = r_sorted["_cumsum_pt"] - r_sorted["_pt"]
    r_sorted["_in_p95"] = (r_sorted["_prev"] < 0.95).astype(float)
    r_sorted["_ps_if"] = r_sorted["_ps"] * r_sorted["_in_p95"]
    smt = r_sorted.groupby(["sequence_id", "response_pos"], as_index=False, sort=False).agg(
        S_mass_T_p95=("_ps_if", "sum"))
    return agg.merge(smt, on=["sequence_id", "response_pos"], how="left")


# Bin definitions for 3 axes
MAX_PS_BINS = [0, 0.5, 0.95, 1.01]
MAX_PS_LABELS = ["low\n(<0.5)", "mid\n(0.5-0.95)", "high (frozen)\n(>0.95)"]

S_MASS_BINS = [-0.01, 0.01, 0.5, 1.01]
S_MASS_LABELS = ["disjoint\n(=0)", "partial\n(0-0.5)", "aligned\n(>0.5)"]

ABS_A_BINS = [0, 2, 6, 30]
ABS_A_LABELS = ["weak\n(<2)", "moderate\n(2-6)", "strong\n(>6)"]


def main():
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    pair_dirs = [d for d in pair_dirs if os.path.basename(d) != "figures"]

    all_records = []
    pair_names = []

    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
        if not step_files: continue
        parts = [build(pd.read_parquet(f, columns=RAW_COLS)) for f in step_files]
        df = pd.concat(parts, ignore_index=True)
        sub = df[df["init_fkl"] >= FKL_THRESHOLD].copy()
        if len(sub) < 100:
            continue

        sub["max_ps_bin"] = pd.cut(sub["max_ps"], MAX_PS_BINS, labels=MAX_PS_LABELS)
        sub["s_mass_bin"] = pd.cut(sub["S_mass_T_p95"], S_MASS_BINS, labels=S_MASS_LABELS)
        sub["abs_a_bin"] = pd.cut(sub["abs_A"], ABS_A_BINS, labels=ABS_A_LABELS)
        sub["is_stuck"] = (sub["delta_fkl"] > -0.05) & (sub["delta_fkl"] < 0.05)
        sub["is_worse"] = sub["delta_fkl"] >= 0.05
        sub["is_stuck_or_worse"] = sub["is_stuck"] | sub["is_worse"]
        sub["pair"] = pair

        grp = sub.groupby(["max_ps_bin", "s_mass_bin", "abs_a_bin"], observed=False).agg(
            n=("delta_fkl", "size"),
            stuck_rate=("is_stuck", "mean"),
            worse_rate=("is_worse", "mean"),
            bad_rate=("is_stuck_or_worse", "mean"),
            mean_dfkl=("delta_fkl", "mean"),
        ).reset_index()
        grp["pair"] = pair
        all_records.append(grp)
        pair_names.append(pair)

    agg_df = pd.concat(all_records, ignore_index=True)

    # For each 3D bin, average across pairs (weight by n)
    def weighted_mean(g, val_col):
        return (g[val_col] * g["n"]).sum() / max(g["n"].sum(), 1)

    consensus = agg_df.groupby(["max_ps_bin", "s_mass_bin", "abs_a_bin"], observed=False).apply(
        lambda g: pd.Series({
            "n_total": g["n"].sum(),
            "n_pairs_with_data": (g["n"] >= 20).sum(),
            "avg_stuck_rate": weighted_mean(g[g["n"] >= 20], "stuck_rate") if (g["n"] >= 20).any() else np.nan,
            "avg_worse_rate": weighted_mean(g[g["n"] >= 20], "worse_rate") if (g["n"] >= 20).any() else np.nan,
            "avg_bad_rate": weighted_mean(g[g["n"] >= 20], "bad_rate") if (g["n"] >= 20).any() else np.nan,
            "std_bad_rate": g[g["n"] >= 20]["bad_rate"].std() if (g["n"] >= 20).sum() >= 2 else np.nan,
        })
    ).reset_index()

    print("=== Consensus across pairs — top-10 worst bins by avg_bad_rate ===")
    top = consensus[consensus["n_pairs_with_data"] >= 3].sort_values("avg_bad_rate", ascending=False).head(15)
    print(top[["max_ps_bin", "s_mass_bin", "abs_a_bin",
               "n_total", "n_pairs_with_data",
               "avg_stuck_rate", "avg_worse_rate", "avg_bad_rate", "std_bad_rate"]].to_string(index=False))

    print("\n=== Consensus — top-10 best (learn-most) bins by lowest avg_bad_rate ===")
    bot = consensus[consensus["n_pairs_with_data"] >= 3].sort_values("avg_bad_rate", ascending=True).head(15)
    print(bot[["max_ps_bin", "s_mass_bin", "abs_a_bin",
               "n_total", "n_pairs_with_data",
               "avg_stuck_rate", "avg_worse_rate", "avg_bad_rate", "std_bad_rate"]].to_string(index=False))

    # === Figure: 3x3 grid of heatmaps ===
    # 3 columns of max_ps_bin; each subplot is s_mass_bin (y) x abs_a_bin (x)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    metrics = [
        ("avg_stuck_rate", "avg STUCK rate", "Greys"),
        ("avg_worse_rate", "avg WORSE rate", "Reds"),
        ("avg_bad_rate", "avg STUCK+WORSE rate", "OrRd"),
    ]

    for row_i, (metric, mlabel, cmap) in enumerate(metrics):
        for col_i, mps_lbl in enumerate(MAX_PS_LABELS):
            ax = axes[row_i, col_i]
            mat = np.full((3, 3), np.nan)
            annot = np.empty((3, 3), dtype=object)
            for si, s_lbl in enumerate(S_MASS_LABELS):
                for ai, a_lbl in enumerate(ABS_A_LABELS):
                    row = consensus[(consensus["max_ps_bin"] == mps_lbl)
                                    & (consensus["s_mass_bin"] == s_lbl)
                                    & (consensus["abs_a_bin"] == a_lbl)]
                    if len(row) == 0:
                        annot[si, ai] = ""
                        continue
                    row = row.iloc[0]
                    if row["n_pairs_with_data"] < 3:
                        annot[si, ai] = f"insuf\n({int(row['n_pairs_with_data'])}p)"
                        continue
                    mat[si, ai] = row[metric]
                    annot[si, ai] = f"{row[metric]:.2f}\nn={int(row['n_total'])}\n{int(row['n_pairs_with_data'])}p"

            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="upper")
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(ABS_A_LABELS, fontsize=8)
            ax.set_yticklabels(S_MASS_LABELS, fontsize=8)
            for si in range(3):
                for ai in range(3):
                    color = "white" if (not np.isnan(mat[si, ai]) and mat[si, ai] > 0.5) else "black"
                    ax.text(ai, si, annot[si, ai], ha="center", va="center",
                            fontsize=7, color=color)
            if row_i == 0:
                ax.set_title(f"max π_S: {mps_lbl}", fontsize=10, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(f"{mlabel}\n\nπ_S mass on T-nucleus", fontsize=9, fontweight="bold")
            if row_i == 2:
                ax.set_xlabel("|A|", fontsize=9)

    fig.suptitle(f"Universal patterns: fraction STUCK / WORSE by (max π_S, S_mass_T, |A|)\n"
                 f"init_FKL ≥ {FKL_THRESHOLD}, averaged across pairs (weighted by n)\n"
                 f"Annotation: rate / total n / #pairs contributing",
                 fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/universal_patterns_3d.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")

    # === Second figure: per-pair variability heatmap ===
    # For each 3D bin, show std of bad_rate across pairs → tells us where pairs disagree
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
    for col_i, mps_lbl in enumerate(MAX_PS_LABELS):
        ax = axes2[col_i]
        mat = np.full((3, 3), np.nan)
        annot = np.empty((3, 3), dtype=object)
        for si, s_lbl in enumerate(S_MASS_LABELS):
            for ai, a_lbl in enumerate(ABS_A_LABELS):
                row = consensus[(consensus["max_ps_bin"] == mps_lbl)
                                & (consensus["s_mass_bin"] == s_lbl)
                                & (consensus["abs_a_bin"] == a_lbl)]
                if len(row) == 0: annot[si, ai] = ""; continue
                row = row.iloc[0]
                if row["n_pairs_with_data"] < 3:
                    annot[si, ai] = ""; continue
                mat[si, ai] = row["std_bad_rate"]
                annot[si, ai] = f"{row['std_bad_rate']:.2f}"
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=0.4, aspect="auto")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(ABS_A_LABELS, fontsize=8)
        ax.set_yticklabels(S_MASS_LABELS, fontsize=8)
        for si in range(3):
            for ai in range(3):
                ax.text(ai, si, annot[si, ai], ha="center", va="center",
                        fontsize=8, color="white")
        ax.set_title(f"max π_S: {mps_lbl}", fontsize=10, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel("std of bad_rate across pairs", fontsize=9)
        ax.set_xlabel("|A|", fontsize=9)

    fig2.suptitle("Cross-pair disagreement: std of STUCK+WORSE rate across pairs\n"
                  "(low = universal pattern; high = pair-specific)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out2 = f"{out_dir}/universal_patterns_variability.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
