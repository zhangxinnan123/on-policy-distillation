"""C: Cross-pair pooled 2D grid (10 feature pairs × N FKL bins) — mean ΔFKL.
E: Extract top-N most-red / most-blue cells across all combinations as rules.
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
MIN_CELL_N = 30    # only annotate cells with >= this many samples
TOP_K_RULES = 25   # top rules for E

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


def compute_heatmap(sub, x_col, y_col, xe, ye):
    xi = np.clip(np.searchsorted(xe, sub[x_col].values, side="right") - 1, 0, len(xe) - 2)
    yi = np.clip(np.searchsorted(ye, sub[y_col].values, side="right") - 1, 0, len(ye) - 2)
    nx, ny = len(xe) - 1, len(ye) - 1
    v = sub["delta_fkl"].values
    mat = np.full((ny, nx), np.nan)
    cnt = np.zeros((ny, nx), dtype=np.int64)
    for iy in range(ny):
        for ix in range(nx):
            m = (xi == ix) & (yi == iy)
            n = m.sum()
            if n >= MIN_CELL_N:
                mat[iy, ix] = np.mean(v[m])
                cnt[iy, ix] = n
    return mat, cnt


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
    print(f"Pooled N = {len(all_df)}", flush=True)

    # Global bin edges per feature (across all pairs)
    global_edges = {}
    for fname, _, frange in FEATURES:
        global_edges[fname] = bin_edges(all_df[fname].values, N_INNER, frange)

    # FKL bins
    fkl_edges = [e for e in FKL_EDGES if e <= all_df["init_fkl"].max() + EPS]
    dm = float(all_df["init_fkl"].max())
    if fkl_edges[-1] < dm: fkl_edges.append(dm + EPS)
    fkl_labels = [f"[{fkl_edges[i]:.2g},{fkl_edges[i+1]:.2g})"
                  for i in range(len(fkl_edges) - 1)]
    n_fkl = len(fkl_labels)
    all_df["_fkl_bin"] = np.clip(
        np.searchsorted(fkl_edges, all_df["init_fkl"].values, side="right") - 1,
        0, n_fkl - 1,
    )
    fkl_counts = [int((all_df["_fkl_bin"] == c).sum()) for c in range(n_fkl)]

    feat_pairs = list(itertools.combinations(range(len(FEATURES)), 2))
    n_fp = len(feat_pairs)

    # ============ C: full grid n_fp × n_fkl of heatmaps ============
    fig, axes = plt.subplots(n_fp, n_fkl, figsize=(2.7 * n_fkl, 2.5 * n_fp), squeeze=False)

    # First pass: compute all heatmaps + collect global vmax per feature-pair row
    heatmaps = {}   # (fp_i, c) -> (mat, cnt, xe, ye)
    for fp_i, (i, j) in enumerate(feat_pairs):
        fx, _, _ = FEATURES[i]; fy, _, _ = FEATURES[j]
        for c in range(n_fkl):
            sub = all_df[all_df["_fkl_bin"] == c]
            if len(sub) < 500: continue
            mat, cnt = compute_heatmap(sub, fx, fy, global_edges[fx], global_edges[fy])
            heatmaps[(fp_i, c)] = (mat, cnt, global_edges[fx], global_edges[fy])

    for fp_i, (i, j) in enumerate(feat_pairs):
        fx, lx, _ = FEATURES[i]; fy, ly, _ = FEATURES[j]
        # Row-specific vmax to make small-FKL rows still readable
        row_vals = []
        for c in range(n_fkl):
            if (fp_i, c) in heatmaps:
                mat = heatmaps[(fp_i, c)][0]
                row_vals.append(mat[np.isfinite(mat)])
        row_vals = np.concatenate(row_vals) if row_vals else np.array([0.0])
        vmax = float(np.percentile(np.abs(row_vals), 98)) if len(row_vals) else 0.1
        vmax = max(vmax, 0.01)

        for c in range(n_fkl):
            ax = axes[fp_i, c]
            if (fp_i, c) not in heatmaps:
                ax.text(0.5, 0.5, "N insuf", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            mat, cnt, xe, ye = heatmaps[(fp_i, c)]
            im = ax.imshow(mat, origin="lower", aspect="auto", cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax,
                            extent=[xe[0], xe[-1], ye[0], ye[-1]])
            if fp_i == 0:
                ax.set_title(f"init_FKL {fkl_labels[c]}\nN={fkl_counts[c]}",
                             fontsize=7, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{lx}×{ly}\n{ly}", fontsize=6, fontweight="bold")
            if fp_i == n_fp - 1:
                ax.set_xlabel(lx, fontsize=7)
            ax.tick_params(labelsize=5)

    fig.suptitle("Pooled cross-pair mean ΔFKL, 15 feature pairs × 8 init_FKL bins\n"
                 "(row-normalized colorscale: red=learn worse, blue=learn better)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out_c = f"{out_dir}/pooled_grid_15feat.png"
    plt.savefig(out_c, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_c}")

    # ============ E: extract top-K red and top-K blue cells as rules ============
    rows = []
    for (fp_i, c), (mat, cnt, xe, ye) in heatmaps.items():
        i, j = feat_pairs[fp_i]
        fx, lx, _ = FEATURES[i]; fy, ly, _ = FEATURES[j]
        ny, nx = mat.shape
        for iy in range(ny):
            for ix in range(nx):
                if not np.isfinite(mat[iy, ix]): continue
                if cnt[iy, ix] < MIN_CELL_N: continue
                rows.append({
                    "fp": f"{lx}×{ly}",
                    "fkl_bin": fkl_labels[c],
                    "x_feat": lx, "x_lo": xe[ix], "x_hi": xe[ix + 1],
                    "y_feat": ly, "y_lo": ye[iy], "y_hi": ye[iy + 1],
                    "n": int(cnt[iy, ix]),
                    "mean_dfkl": float(mat[iy, ix]),
                })
    rules = pd.DataFrame(rows)
    rules_by_worst = rules.sort_values("mean_dfkl", ascending=False).head(TOP_K_RULES)
    rules_by_best = rules.sort_values("mean_dfkl", ascending=True).head(TOP_K_RULES)

    # Save readable rule tables
    def fmt_rule(r):
        return (f"init_FKL {r['fkl_bin']:16s}  {r['x_feat']}∈[{r['x_lo']:.2g},{r['x_hi']:.2g}] "
                f"& {r['y_feat']}∈[{r['y_lo']:.2g},{r['y_hi']:.2g}]  "
                f"n={r['n']:5d}  ΔFKL={r['mean_dfkl']:+.3f}")

    with open(f"{out_dir}/top_rules.txt", "w") as fh:
        fh.write(f"Pooled data: N={len(all_df)}\n")
        fh.write(f"Min cell samples: {MIN_CELL_N}\n\n")
        fh.write("=" * 100 + "\n")
        fh.write(f"TOP {TOP_K_RULES} WORST cells (largest positive ΔFKL — 'learns worse')\n")
        fh.write("=" * 100 + "\n")
        for _, r in rules_by_worst.iterrows():
            fh.write(fmt_rule(r) + "\n")
        fh.write("\n" + "=" * 100 + "\n")
        fh.write(f"TOP {TOP_K_RULES} BEST cells (most negative ΔFKL — 'learns most')\n")
        fh.write("=" * 100 + "\n")
        for _, r in rules_by_best.iterrows():
            fh.write(fmt_rule(r) + "\n")
    print(f"Saved {out_dir}/top_rules.txt")

    print("\nTOP 10 WORST cells:")
    for _, r in rules_by_worst.head(10).iterrows():
        print(f"  {fmt_rule(r)}")
    print("\nTOP 10 BEST cells:")
    for _, r in rules_by_best.head(10).iterrows():
        print(f"  {fmt_rule(r)}")

    # Also save rules table as CSV
    rules_sorted = rules.sort_values("mean_dfkl", key=lambda s: s.abs(), ascending=False)
    rules_sorted.to_csv(f"{out_dir}/all_rules.csv", index=False)
    print(f"Saved {out_dir}/all_rules.csv ({len(rules)} rules)")


if __name__ == "__main__":
    main()
