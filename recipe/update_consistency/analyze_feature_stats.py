"""Feature statistics: correlation heatmap + linear regression coefficients,
both stratified by init_FKL bin.

A. Correlation (Pearson & Spearman) of each feature with ΔFKL per FKL bin.
B. Standardized linear regression coefficient of each feature per FKL bin.
"""
import os, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
BASE = os.environ.get("BASE", "/fsx/xinnanzh/data/update_consistency_multi_pair")
N_STEPS = int(os.environ.get("N_STEPS", "10"))
FKL_EDGES = [0.0, 0.05, 0.2, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0]

RAW_COLS = [
    "sequence_id", "response_pos",
    "pi_S_before", "pi_S_after", "pi_T",
    "teacher_entropy", "overlap_ratio_k8",
    "A_value",
]

FEATURES = [
    ("S_ent_topk", "S_ent"),
    ("T_ent", "T_ent"),
    ("overlap_ratio_k8", "overlap k8"),
    ("S_mass_T_p95", "π_S mass on T-nucleus"),
    ("max_ps", "max π_S"),
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


def main():
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    pair_dirs = [d for d in pair_dirs if os.path.basename(d) != "figures"]

    all_dfs = []
    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
        if not step_files: continue
        parts = [build(pd.read_parquet(f, columns=RAW_COLS)) for f in step_files]
        df = pd.concat(parts, ignore_index=True)
        df["pair"] = pair
        all_dfs.append(df)
        print(f"  Loaded {pair}: N={len(df)}", flush=True)
    all_df = pd.concat(all_dfs, ignore_index=True)

    fkl_edges = [e for e in FKL_EDGES if e <= all_df["init_fkl"].max() + EPS]
    data_max = float(all_df["init_fkl"].max())
    if fkl_edges[-1] < data_max:
        fkl_edges.append(data_max + EPS)
    fkl_labels = [f"[{fkl_edges[i]:.2g},{fkl_edges[i+1]:.2g})"
                  for i in range(len(fkl_edges) - 1)]
    all_df["_fkl_bin"] = np.clip(
        np.searchsorted(fkl_edges, all_df["init_fkl"].values, side="right") - 1,
        0, len(fkl_edges) - 2,
    )

    n_bins = len(fkl_labels)
    n_feat = len(FEATURES)

    pearson_mat = np.full((n_feat, n_bins), np.nan)
    spearman_mat = np.full((n_feat, n_bins), np.nan)
    lin_coef_mat = np.full((n_feat, n_bins), np.nan)
    bin_stats = []

    for c in range(n_bins):
        sub = all_df[all_df["_fkl_bin"] == c]
        n = len(sub)
        bin_stats.append({"bin": fkl_labels[c], "n": n})
        if n < 500:
            continue
        y = sub["delta_fkl"].values
        # Subsample if too many for speed
        if n > 200_000:
            idx = np.random.default_rng(0).choice(n, 200_000, replace=False)
            sub = sub.iloc[idx]
            y = sub["delta_fkl"].values

        # A. Correlations
        for r, (feat, _) in enumerate(FEATURES):
            x = sub[feat].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 100: continue
            pr, _ = pearsonr(x[mask], y[mask])
            sr, _ = spearmanr(x[mask], y[mask])
            pearson_mat[r, c] = pr
            spearman_mat[r, c] = sr

        # B. Linear regression with standardized inputs
        feat_cols = [f for f, _ in FEATURES]
        X = sub[feat_cols].values
        mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        Xm, ym = X[mask], y[mask]
        if len(ym) < 100: continue
        Xs = StandardScaler().fit_transform(Xm)
        # Also standardize y for interpretable coefficients
        y_std = (ym - ym.mean()) / (ym.std() + 1e-12)
        lr = LinearRegression().fit(Xs, y_std)
        for r, coef in enumerate(lr.coef_):
            lin_coef_mat[r, c] = coef

    # ============ Plot ============
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    for ax, mat, title in [
        (axes[0], pearson_mat, "A1. Pearson correlation (feature, ΔFKL)"),
        (axes[1], spearman_mat, "A2. Spearman correlation (feature, ΔFKL)"),
        (axes[2], lin_coef_mat, "B. Std. linear regression coefficient\n(standardized X, standardized y)"),
    ]:
        vmax = float(np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([f"{fkl_labels[c]}\nN={bin_stats[c]['n']}" for c in range(n_bins)],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels([label for _, label in FEATURES], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("init_FKL bin", fontsize=9)
        for i in range(n_feat):
            for j in range(n_bins):
                if np.isfinite(mat[i, j]):
                    txt_color = "white" if abs(mat[i, j]) > 0.5 * vmax else "black"
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                            fontsize=6, color=txt_color)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle("Feature statistics vs ΔFKL, stratified by init_FKL bin  [pooled across 9 pairs]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/feature_stats_corr_lin.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
