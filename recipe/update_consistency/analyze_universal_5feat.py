"""Universal cross-pair 2D patterns using 5 features from fkl_2d_grid.

Features (matches fkl_2d_grid figure):
  S_ent_topk, T_ent, overlap_ratio_k8, S_mass_T_p95, max_ps

For each of C(5,2)=10 feature pairs, compute STUCK+WORSE rate averaged across
all 9 model pairs (weighted by n). Plot 10 heatmaps showing universal patterns.
"""
import os, glob, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
BASE = os.environ.get("BASE", "/fsx/xinnanzh/data/update_consistency_multi_pair")
N_STEPS = int(os.environ.get("N_STEPS", "10"))
FKL_THRESHOLD = float(os.environ.get("FKL_THRESHOLD", "2.0"))
N_BINS = int(os.environ.get("N_BINS", "8"))  # bins per axis

RAW_COLS = [
    "sequence_id", "response_pos",
    "pi_S_before", "pi_S_after", "pi_T",
    "teacher_entropy", "overlap_ratio_k8",
    "A_value",
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


FEATURES = [
    ("S_ent_topk", "S_ent", None),           # entropy — no fixed range
    ("T_ent", "T_ent", None),
    ("overlap_ratio_k8", "overlap", (0, 1)),
    ("S_mass_T_p95", "S_mass_T", (0, 1)),
    ("max_ps", "max_ps", (0, 1)),
    ("A_value", "A", None),                  # raw A value (signed)
]


def make_bin_edges(x, n, fixed_range=None):
    """Quantile bins if no range, else equal-width."""
    if fixed_range is not None:
        return np.linspace(fixed_range[0], fixed_range[1], n + 1)
    e = np.quantile(x, np.linspace(0, 1, n + 1))
    e = np.unique(e)
    if len(e) < 3:
        e = np.linspace(x.min(), x.max() + EPS, n + 1)
    return e


def main():
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    pair_dirs = [d for d in pair_dirs if os.path.basename(d) != "figures"]

    # Load all pair data
    pair_data = {}
    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
        if not step_files: continue
        parts = [build(pd.read_parquet(f, columns=RAW_COLS)) for f in step_files]
        df = pd.concat(parts, ignore_index=True)
        sub = df[df["init_fkl"] >= FKL_THRESHOLD].copy()
        if len(sub) < 100: continue
        sub["is_stuck"] = (sub["delta_fkl"] > -0.05) & (sub["delta_fkl"] < 0.05)
        sub["is_worse"] = sub["delta_fkl"] >= 0.05
        sub["is_bad"] = sub["is_stuck"] | sub["is_worse"]
        pair_data[pair] = sub
        print(f"  Loaded {pair}: N={len(sub)}", flush=True)

    # Compute global bin edges (based on all pairs pooled)
    all_pooled = pd.concat(pair_data.values(), ignore_index=True)
    bin_edges = {}
    for feat, _, fixed_range in FEATURES:
        bin_edges[feat] = make_bin_edges(all_pooled[feat].values, N_BINS, fixed_range)

    # For each pair, for each feature, get bin index
    for pair, sub in pair_data.items():
        for feat, _, _ in FEATURES:
            e = bin_edges[feat]
            sub[f"{feat}_bin"] = np.clip(np.searchsorted(e, sub[feat].values, side="right") - 1,
                                          0, len(e) - 2)

    # === Compute per-pair-avg matrix for each feature pair ===
    axis_pairs = list(itertools.combinations(range(len(FEATURES)), 2))
    n_pair_feat = len(axis_pairs)  # 10

    fig, axes = plt.subplots(3, 5, figsize=(4 * 5, 3.7 * 3), squeeze=False)

    for idx, (i, j) in enumerate(axis_pairs):
        ax = axes[idx // 5, idx % 5]
        fx, lx, _ = FEATURES[i]
        fy, ly, _ = FEATURES[j]

        # Per-pair matrix: for each pair, compute bad_rate in each 2D bin
        # Then weighted-mean across pairs
        sum_n = np.zeros((N_BINS, N_BINS))
        sum_bad = np.zeros((N_BINS, N_BINS))
        for pair, sub in pair_data.items():
            xi = sub[f"{fx}_bin"].values
            yi = sub[f"{fy}_bin"].values
            for iy in range(N_BINS):
                for ix in range(N_BINS):
                    m = (xi == ix) & (yi == iy)
                    n = m.sum()
                    if n < 20: continue
                    sum_n[iy, ix] += n
                    sum_bad[iy, ix] += sub[m]["is_bad"].sum()
        mat = np.divide(sum_bad, sum_n, out=np.full_like(sum_bad, np.nan), where=sum_n > 0)

        xe = bin_edges[fx]; ye = bin_edges[fy]
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="OrRd",
                        vmin=0, vmax=0.6,
                        extent=[xe[0], xe[-1], ye[0], ye[-1]])
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        ax.set_xlabel(lx, fontsize=9)
        ax.set_ylabel(ly, fontsize=9)
        ax.set_title(f"{lx} × {ly}", fontsize=10, fontweight="bold")

    fig.suptitle(f"Universal STUCK+WORSE rate across 9 pairs, init_FKL ≥ {FKL_THRESHOLD}\n"
                 f"(all C(5,2)=10 feature combinations; averaged/weighted across pairs)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/universal_2d_5feat.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")

    # === Second figure: mean ΔFKL heatmap for each 2D combo ===
    fig2, axes2 = plt.subplots(3, 5, figsize=(4 * 5, 3.7 * 3), squeeze=False)

    for idx, (i, j) in enumerate(axis_pairs):
        ax = axes2[idx // 5, idx % 5]
        fx, lx, _ = FEATURES[i]
        fy, ly, _ = FEATURES[j]

        sum_n = np.zeros((N_BINS, N_BINS))
        sum_dfkl = np.zeros((N_BINS, N_BINS))
        for pair, sub in pair_data.items():
            xi = sub[f"{fx}_bin"].values
            yi = sub[f"{fy}_bin"].values
            for iy in range(N_BINS):
                for ix in range(N_BINS):
                    m = (xi == ix) & (yi == iy)
                    n = m.sum()
                    if n < 20: continue
                    sum_n[iy, ix] += n
                    sum_dfkl[iy, ix] += sub[m]["delta_fkl"].sum()
        mat = np.divide(sum_dfkl, sum_n, out=np.full_like(sum_dfkl, np.nan), where=sum_n > 0)
        finite = mat[np.isfinite(mat)]
        vmax = float(np.nanmax(np.abs(finite))) if len(finite) else 0.3

        xe = bin_edges[fx]; ye = bin_edges[fy]
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax,
                        extent=[xe[0], xe[-1], ye[0], ye[-1]])
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        ax.set_xlabel(lx, fontsize=9)
        ax.set_ylabel(ly, fontsize=9)
        ax.set_title(f"{lx} × {ly}", fontsize=10, fontweight="bold")

    fig2.suptitle(f"Universal mean ΔFKL across 9 pairs, init_FKL ≥ {FKL_THRESHOLD}\n"
                  f"(red = learning worse; blue = learning better)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = f"{out_dir}/universal_2d_5feat_dfkl.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
