"""Per-feature ΔFKL curves stratified by init_FKL bin.

Layout: 6 rows (one per feature) × N columns (init_FKL bins).
Each cell: mean ΔFKL as function of the feature, computed across all 9 pairs.
"""
import os, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
BASE = os.environ.get("BASE", "/fsx/xinnanzh/data/update_consistency_multi_pair")
N_STEPS = int(os.environ.get("N_STEPS", "10"))
N_FEAT_BINS = int(os.environ.get("N_FEAT_BINS", "12"))

# Same bins as fkl_2d_grid
FKL_EDGES = [0.0, 0.05, 0.2, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0]

RAW_COLS = [
    "sequence_id", "response_pos",
    "pi_S_before", "pi_S_after", "pi_T",
    "teacher_entropy", "overlap_ratio_k8",
    "A_value",
]

FEATURES = [
    ("S_ent_topk", "S_ent (top-K)", None),
    ("T_ent", "T_ent", None),
    ("overlap_ratio_k8", "overlap k8", (0, 1)),
    ("S_mass_T_p95", "π_S mass on T-nucleus", (0, 1)),
    ("max_ps", "max π_S", (0, 1)),
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
    print(f"Total N = {len(all_df)}", flush=True)

    # Determine actual FKL bins based on data range
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

    n_cols = len(fkl_labels)
    n_rows = len(FEATURES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.5 * n_rows),
                              squeeze=False, sharex=False, sharey=False)

    # Per-feature global range (from pooled data) for consistent x-axis across cols
    feat_ranges = {}
    for fname, _, frange in FEATURES:
        if frange is None:
            vals = all_df[fname].values
            feat_ranges[fname] = (np.percentile(vals, 1), np.percentile(vals, 99))
        else:
            feat_ranges[fname] = frange

    for r, (fname, flabel, _) in enumerate(FEATURES):
        lo, hi = feat_ranges[fname]
        # Equal-width bins over the global range
        f_edges = np.linspace(lo, hi, N_FEAT_BINS + 1)
        f_centers = 0.5 * (f_edges[:-1] + f_edges[1:])

        for c in range(n_cols):
            ax = axes[r, c]
            sub = all_df[all_df["_fkl_bin"] == c]
            if len(sub) < 200:
                ax.text(0.5, 0.5, f"N={len(sub)}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8)
                ax.axis("off"); continue

            # Bin the feature
            fi = np.clip(np.searchsorted(f_edges, sub[fname].values, side="right") - 1,
                          0, N_FEAT_BINS - 1)
            means, sems, counts = [], [], []
            for i in range(N_FEAT_BINS):
                m = fi == i
                if m.sum() < 20:
                    means.append(np.nan); sems.append(np.nan); counts.append(0)
                    continue
                d = sub["delta_fkl"].values[m]
                means.append(d.mean())
                sems.append(d.std() / np.sqrt(m.sum()))
                counts.append(m.sum())
            means = np.array(means); sems = np.array(sems)

            valid = ~np.isnan(means)
            if valid.sum() < 2:
                ax.text(0.5, 0.5, "insuf", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8)
                ax.axis("off"); continue

            ax.errorbar(f_centers[valid], means[valid], yerr=sems[valid],
                        color="#374151", lw=1.2, capsize=2, marker="o", markersize=4,
                        markerfacecolor="#2563EB", markeredgecolor="black", mew=0.4)
            ax.axhline(0, color="red", lw=0.6, alpha=0.5)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(lo, hi)

            if r == 0:
                ax.set_title(f"init_FKL {fkl_labels[c]}\nN={len(sub)}",
                             fontsize=8, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel(flabel, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{flabel}\nmean ΔFKL", fontsize=8, fontweight="bold")

    fig.suptitle("Per-feature mean ΔFKL vs feature value, stratified by init_FKL bin  "
                 "[pooled across 9 pairs]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/feature_vs_dfkl_stratified.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
