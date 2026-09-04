"""Quantify the regularity/structure of 2D ΔFKL heatmaps across (feature_pair, init_FKL_bin).

Method A: Compute a "structure strength" score per cell:
    - gradient magnitude (mean |∇|) — how steep the pattern is
    - dynamic range (max − min)     — how much ΔFKL spans across the plane
    - explained variance R² of a plane fit — how "linear/monotonic" the pattern is
Then plot feature_pair × FKL_bin heatmaps of these scores.

Method B: Select top feature pairs from Method A and plot diagonal slices
of the 2D pattern across init_FKL bins.
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


def heatmap_and_scores(df, x_col, y_col, x_range=None, y_range=None):
    """Return (mat, gradient_score, range_score, plane_r2)."""
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
    finite = np.isfinite(mat)
    if finite.sum() < 4:
        return mat, np.nan, np.nan, np.nan, xe, ye

    valid_vals = mat[finite]
    # 1. Dynamic range: max - min
    range_score = float(np.nanmax(mat) - np.nanmin(mat))

    # 2. Gradient magnitude: mean |∇| on filled cells
    gy, gx = np.gradient(np.where(finite, mat, np.nan))
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    grad_score = float(np.nanmean(gmag))

    # 3. Plane fit R²: fit z = a*ix + b*iy + c to filled cells, report R²
    iy_grid, ix_grid = np.mgrid[0:ny, 0:nx]
    X = np.column_stack([ix_grid[finite], iy_grid[finite], np.ones(finite.sum())])
    y = mat[finite]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    plane_r2 = 1 - ss_res / max(ss_tot, 1e-12) if ss_tot > 1e-12 else np.nan

    return mat, grad_score, range_score, plane_r2, xe, ye


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
        all_dfs.append(df)
        print(f"  Loaded {pair}: N={len(df)}", flush=True)
    all_df = pd.concat(all_dfs, ignore_index=True)

    fkl_edges = [e for e in FKL_EDGES if e <= all_df["init_fkl"].max() + EPS]
    dm = float(all_df["init_fkl"].max())
    if fkl_edges[-1] < dm: fkl_edges.append(dm + EPS)
    fkl_labels = [f"[{fkl_edges[i]:.2g},{fkl_edges[i+1]:.2g})"
                  for i in range(len(fkl_edges) - 1)]
    all_df["_fkl_bin"] = np.clip(
        np.searchsorted(fkl_edges, all_df["init_fkl"].values, side="right") - 1,
        0, len(fkl_edges) - 2,
    )

    # Feature pairs C(6,2) = 15
    feat_pairs = list(itertools.combinations(range(len(FEATURES)), 2))
    n_fp = len(feat_pairs)
    n_fkl = len(fkl_labels)

    range_mat = np.full((n_fp, n_fkl), np.nan)
    grad_mat = np.full((n_fp, n_fkl), np.nan)
    r2_mat = np.full((n_fp, n_fkl), np.nan)
    pair_labels = []

    # Precompute all heatmaps for method B
    hm_store = {}   # (fp_idx, fkl_bin) -> (mat, xe, ye)

    for fp_i, (i, j) in enumerate(feat_pairs):
        fx, lx, rx = FEATURES[i]; fy, ly, ry = FEATURES[j]
        pair_labels.append(f"{lx} × {ly}")

        for c in range(n_fkl):
            sub = all_df[all_df["_fkl_bin"] == c]
            if len(sub) < 500:
                continue
            mat, gs, rs, r2, xe, ye = heatmap_and_scores(sub, fx, fy, rx, ry)
            grad_mat[fp_i, c] = gs
            range_mat[fp_i, c] = rs
            r2_mat[fp_i, c] = r2
            hm_store[(fp_i, c)] = (mat, xe, ye)

    # ============ Method A: 3 heatmaps of structure scores ============
    fig, axes = plt.subplots(1, 3, figsize=(21, max(6, 0.4 * n_fp)))
    for ax, mat, title in [
        (axes[0], range_mat, "A1. Dynamic range (max−min of ΔFKL)"),
        (axes[1], grad_mat, "A2. Mean gradient magnitude |∇ΔFKL|"),
        (axes[2], r2_mat, "A3. Plane-fit R²\n(monotonic linear structure)"),
    ]:
        vmax = float(np.nanpercentile(mat, 98))
        vmin = 0 if title.startswith("A1") or title.startswith("A2") else float(np.nanmin(mat))
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_fkl))
        ax.set_xticklabels(fkl_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_fp))
        ax.set_yticklabels(pair_labels, fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("init_FKL bin", fontsize=9)
        for i in range(n_fp):
            for j in range(n_fkl):
                if np.isfinite(mat[i, j]):
                    txt_color = "white" if mat[i, j] < 0.5 * vmax else "black"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=5, color=txt_color)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.suptitle("Structure strength of 2D ΔFKL heatmaps: feature-pair × init_FKL bin  [pooled across 9 pairs]",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out_a = f"{out_dir}/structure_strength_scores.png"
    plt.savefig(out_a, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out_a}")

    # Print top pairs by mean range across FKL bins ≥ 1.5
    high_fkl_cols = [c for c in range(n_fkl) if fkl_edges[c] >= 1.5]
    if high_fkl_cols:
        mean_range_high = np.nanmean(range_mat[:, high_fkl_cols], axis=1)
        sorted_idx = np.argsort(mean_range_high)[::-1]
        print("\nTop feature pairs by mean dynamic range in high-FKL bins (≥1.5):")
        for k, idx in enumerate(sorted_idx[:6]):
            print(f"  {k+1}. {pair_labels[idx]:30s}: range={mean_range_high[idx]:.3f}")
        top_pairs = sorted_idx[:4]
    else:
        top_pairs = np.argsort(np.nanmean(range_mat, axis=1))[::-1][:4]

    # ============ Method B: diagonal slices across FKL bins for top pairs ============
    # For each top feature pair, take the diagonal / anti-diagonal of each bin's heatmap
    # and plot them together to see how the pattern evolves.
    n_top = len(top_pairs)
    fig2, axes2 = plt.subplots(n_top, 2, figsize=(15, 3.5 * n_top), squeeze=False)
    cmap = plt.get_cmap("viridis")

    for row, fp_i in enumerate(top_pairs):
        i, j = feat_pairs[fp_i]
        fx, lx, _ = FEATURES[i]; fy, ly, _ = FEATURES[j]

        for col, direction in enumerate([("diagonal", 1), ("anti-diagonal", -1)]):
            name, sign = direction
            ax = axes2[row, col]

            for c in range(n_fkl):
                if (fp_i, c) not in hm_store: continue
                mat, xe, ye = hm_store[(fp_i, c)]
                if sign == 1:
                    diag = np.array([mat[k, k] for k in range(min(mat.shape))])
                else:
                    diag = np.array([mat[k, mat.shape[1] - 1 - k]
                                     for k in range(min(mat.shape))])
                if not np.any(np.isfinite(diag)): continue
                xs = np.arange(len(diag)) / (len(diag) - 1)
                color = cmap(c / max(n_fkl - 1, 1))
                ax.plot(xs, diag, marker="o", ms=4, lw=1.4, color=color,
                        alpha=0.85, label=fkl_labels[c])

            ax.axhline(0, color="red", lw=0.6, ls="-", alpha=0.5)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(f"position along {name}\n(0 = {lx if sign==1 else lx} low, 1 = {ly if sign==1 else ly} low)"
                          .replace(name, name), fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{lx} × {ly}\nmean ΔFKL", fontsize=9, fontweight="bold")
            ax.set_title(f"{name} of 2D heatmap ({lx} × {ly})", fontsize=9)
            if row == 0 and col == 1:
                ax.legend(fontsize=6, loc="best", ncol=2, framealpha=0.8, title="init_FKL bin")

    fig2.suptitle("Diagonal slices of top feature-pair 2D heatmaps, colored by init_FKL bin",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_b = f"{out_dir}/diagonal_slices_top_pairs.png"
    plt.savefig(out_b, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_b}")


if __name__ == "__main__":
    main()
