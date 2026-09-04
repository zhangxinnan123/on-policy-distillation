"""Per-pair 2D regularity analysis.

For each of 9 pairs, compute dynamic range of the 2D ΔFKL heatmap for each
(feature_pair × FKL_bin). Then plot 9 subplots (one per pair) as feature-pair
× FKL_bin heatmaps of dynamic range.

Also compute cross-pair correlation of these regularity maps to see which
pairs have similar regularity patterns.
"""
import os, glob, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
BASE = os.environ.get("BASE", "/fsx/xinnanzh/data/update_consistency_multi_pair")
N_STEPS = int(os.environ.get("N_STEPS", "10"))
N_INNER = int(os.environ.get("N_INNER", "10"))
FKL_EDGES = [0.0, 0.05, 0.2, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0]

RAW_COLS = [
    "sequence_id", "response_pos",
    "pi_S_before", "pi_S_after", "pi_T",
    "teacher_entropy", "overlap_ratio_k8",
    "A_value",
]

FEATURES = [
    ("S_ent_topk", "S_ent", None),
    ("T_ent", "T_ent", None),
    ("overlap_ratio_k8", "overlap", (0, 1)),
    ("S_mass_T_p95", "S_mass_T", (0, 1)),
    ("max_ps", "max_ps", (0, 1)),
    ("A_value", "A", None),
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


def bin_edges(x, n, fixed=None):
    if fixed is not None:
        return np.linspace(fixed[0], fixed[1], n + 1)
    e = np.quantile(x, np.linspace(0, 1, n + 1))
    e = np.unique(e)
    if len(e) < 3:
        e = np.linspace(x.min(), x.max() + EPS, n + 1)
    return e


def compute_range(df, x_col, y_col, x_range=None, y_range=None):
    """Return dynamic range max-min of the 2D binned heatmap."""
    xe = bin_edges(df[x_col].values, N_INNER, x_range)
    ye = bin_edges(df[y_col].values, N_INNER, y_range)
    xi = np.clip(np.searchsorted(xe, df[x_col].values, side="right") - 1, 0, len(xe) - 2)
    yi = np.clip(np.searchsorted(ye, df[y_col].values, side="right") - 1, 0, len(ye) - 2)
    nx, ny = len(xe) - 1, len(ye) - 1
    mat = np.full((ny, nx), np.nan)
    v = df["delta_fkl"].values
    for iy in range(ny):
        for ix in range(nx):
            m = (xi == ix) & (yi == iy)
            if m.sum() >= 5:
                mat[iy, ix] = np.mean(v[m])
    if np.isfinite(mat).sum() < 4:
        return np.nan
    return float(np.nanmax(mat) - np.nanmin(mat))


def main():
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    pair_dirs = [d for d in pair_dirs if os.path.basename(d) != "figures"]

    pair_ranges = {}   # pair -> (n_fp, n_fkl) matrix

    feat_pairs = list(itertools.combinations(range(len(FEATURES)), 2))
    n_fp = len(feat_pairs)
    pair_labels = []
    for i, j in feat_pairs:
        _, lx, _ = FEATURES[i]; _, ly, _ = FEATURES[j]
        pair_labels.append(f"{lx}×{ly}")

    fkl_labels = None
    n_fkl = None

    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
        if not step_files: continue
        parts = [build(pd.read_parquet(f, columns=RAW_COLS)) for f in step_files]
        df = pd.concat(parts, ignore_index=True)
        print(f"  {pair}: N={len(df)}", flush=True)

        if fkl_labels is None:
            fkl_edges = [e for e in FKL_EDGES if e <= df["init_fkl"].max() + EPS]
            dm = float(df["init_fkl"].max())
            if fkl_edges[-1] < dm: fkl_edges.append(dm + EPS)
            fkl_labels = [f"[{fkl_edges[i]:.2g},{fkl_edges[i+1]:.2g})"
                          for i in range(len(fkl_edges) - 1)]
            n_fkl = len(fkl_labels)

        fkl_edges_pair = [e for e in FKL_EDGES if e <= df["init_fkl"].max() + EPS]
        dm = float(df["init_fkl"].max())
        if fkl_edges_pair[-1] < dm: fkl_edges_pair.append(dm + EPS)
        df["_fkl_bin"] = np.clip(
            np.searchsorted(fkl_edges_pair, df["init_fkl"].values, side="right") - 1,
            0, len(fkl_edges_pair) - 2,
        )

        rmat = np.full((n_fp, n_fkl), np.nan)
        for fp_i, (i, j) in enumerate(feat_pairs):
            fx, _, rx = FEATURES[i]; fy, _, ry = FEATURES[j]
            for c in range(min(len(fkl_edges_pair) - 1, n_fkl)):
                sub = df[df["_fkl_bin"] == c]
                if len(sub) < 300: continue
                rmat[fp_i, c] = compute_range(sub, fx, fy, rx, ry)
        pair_ranges[pair] = rmat

    # ============ Figure 1: 9 pairs × (feature_pair × FKL_bin) heatmaps ============
    pairs = list(pair_ranges.keys())
    n_p = len(pairs)
    n_cols = 3
    n_rows = (n_p + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows), squeeze=False)

    # Global vmax to compare across pairs
    all_finite = np.concatenate([m.ravel() for m in pair_ranges.values()])
    all_finite = all_finite[np.isfinite(all_finite)]
    vmax = float(np.percentile(all_finite, 98))

    for idx, pair in enumerate(pairs):
        ax = axes[idx // n_cols, idx % n_cols]
        mat = pair_ranges[pair]
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_fkl))
        ax.set_xticklabels(fkl_labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(n_fp))
        ax.set_yticklabels(pair_labels, fontsize=6)
        for i in range(n_fp):
            for j in range(n_fkl):
                if np.isfinite(mat[i, j]):
                    color = "white" if mat[i, j] < 0.4 * vmax else "black"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=5, color=color)
        short = pair.replace("___", "\n→ ")
        ax.set_title(f"{short}", fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    for idx in range(n_p, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(f"Dynamic range of 2D ΔFKL heatmaps: per pair (rows: feature-pair, cols: init_FKL bin)\n"
                 f"Common colorscale: 0 to {vmax:.2f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out1 = f"{out_dir}/regularity_per_pair.png"
    plt.savefig(out1, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out1}")

    # ============ Figure 2: cross-pair similarity of regularity patterns ============
    # For each pair, flatten (feature_pair × FKL_bin) into a vector; compute
    # Pearson correlation between pairs' vectors.
    vecs = []
    for pair in pairs:
        v = pair_ranges[pair].flatten()
        vecs.append(v)
    V = np.array(vecs)  # (n_pairs, n_fp*n_fkl)

    # Compute pairwise correlations using only positions where both are finite
    n_p = len(pairs)
    corr = np.full((n_p, n_p), np.nan)
    for i in range(n_p):
        for j in range(n_p):
            m = np.isfinite(V[i]) & np.isfinite(V[j])
            if m.sum() < 10: continue
            corr[i, j] = np.corrcoef(V[i, m], V[j, m])[0, 1]

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 9))
    im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(n_p))
    ax2.set_xticklabels([p.replace("___", "→\n") for p in pairs], rotation=45,
                        ha="right", fontsize=7)
    ax2.set_yticks(range(n_p))
    ax2.set_yticklabels([p.replace("___", "→ ") for p in pairs], fontsize=7)
    for i in range(n_p):
        for j in range(n_p):
            if np.isfinite(corr[i, j]):
                color = "white" if abs(corr[i, j]) > 0.6 else "black"
                ax2.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color=color)
    plt.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
    ax2.set_title("Cross-pair similarity of regularity patterns\n"
                  "(Pearson corr of the (feature-pair × FKL_bin) range matrix)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    out2 = f"{out_dir}/regularity_pair_similarity.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
