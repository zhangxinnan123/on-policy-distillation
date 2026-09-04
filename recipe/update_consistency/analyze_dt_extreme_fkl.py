"""Decision tree analysis: find feature combinations that predict ΔFKL sign/magnitude
in the extreme init_FKL bin [12, 25] for a specific pair.

Fits max_depth=4 regression tree on 6 features → mean ΔFKL, and visualizes:
  1. The tree itself (rules)
  2. Feature importance
  3. Predicted vs actual ΔFKL per leaf
"""
import os, glob, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text

EPS = 1e-12
BASE = os.environ.get("BASE", "/fsx/xinnanzh/data/update_consistency_multi_pair")
PAIR = os.environ.get("PAIR", "lllyx_Qwen3-1.7B-SFT___lllyx_Qwen3-4B-Base-GRPO")
FKL_LO = float(os.environ.get("FKL_LO", "12"))
FKL_HI = float(os.environ.get("FKL_HI", "25"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "4"))
N_STEPS = int(os.environ.get("N_STEPS", "10"))

RAW_COLS = [
    "sequence_id", "response_pos",
    "pi_S_before", "pi_S_after", "pi_T",
    "teacher_entropy",
    "overlap_ratio_k8",
    "teacher_mass_on_student_topp95",
    "is_sampled_u", "A_value",
]

FEATURES = [
    "student_entropy_topk",
    "teacher_entropy",
    "overlap_ratio_k8",
    "student_mass_on_teacher_topp95",
    "max_ps",
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
        student_entropy_topk=("_neg_ps_logps", "sum"),
        max_ps=("_ps", "max"),
        teacher_entropy=("teacher_entropy", "first"),
        teacher_mass_on_student_topp95=("teacher_mass_on_student_topp95", "first"),
        overlap_ratio_k8=("overlap_ratio_k8", "first"),
        A_value=("A_value", "first"),
    )
    agg["delta_fkl"] = agg["fkl_a"] - agg["init_fkl"]

    # student_mass_on_teacher_topp95
    r_sorted = r.sort_values(["sequence_id", "response_pos", "_pt"],
                             ascending=[True, True, False]).copy()
    r_sorted["_cumsum_pt"] = r_sorted.groupby(["sequence_id", "response_pos"])["_pt"].cumsum()
    r_sorted["_prev"] = r_sorted["_cumsum_pt"] - r_sorted["_pt"]
    r_sorted["_in_p95"] = (r_sorted["_prev"] < 0.95).astype(float)
    r_sorted["_ps_if"] = r_sorted["_ps"] * r_sorted["_in_p95"]
    smt = r_sorted.groupby(["sequence_id", "response_pos"], as_index=False, sort=False).agg(
        student_mass_on_teacher_topp95=("_ps_if", "sum"))
    return agg.merge(smt, on=["sequence_id", "response_pos"], how="left")


def main():
    pair_dir = f"{BASE}/{PAIR}"
    step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
    parts = [build(pd.read_parquet(f, columns=RAW_COLS)) for f in step_files]
    df = pd.concat(parts, ignore_index=True)
    print(f"Total N = {len(df)}", flush=True)

    # Filter to [FKL_LO, FKL_HI]
    sub = df[(df["init_fkl"] >= FKL_LO) & (df["init_fkl"] < FKL_HI)].copy()
    sub = sub.dropna(subset=FEATURES + ["delta_fkl"])
    print(f"[init_FKL in [{FKL_LO}, {FKL_HI})] N = {len(sub)}", flush=True)
    if len(sub) < 100:
        print("Too few samples; abort")
        return

    X = sub[FEATURES].values
    y = sub["delta_fkl"].values

    # === Fit regression tree ===
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_leaf=max(30, len(sub) // 50),
                                 random_state=42)
    tree.fit(X, y)
    r2 = tree.score(X, y)
    y_pred = tree.predict(X)
    print(f"\nTree R² = {r2:.3f}", flush=True)
    print(f"Feature importances:")
    for f, imp in sorted(zip(FEATURES, tree.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f:40s}: {imp:.4f}")

    # === Rules as text ===
    rules_text = export_text(tree, feature_names=FEATURES, decimals=3, max_depth=MAX_DEPTH)
    print(f"\nRules:\n{rules_text}", flush=True)

    # Save rules to file
    out_dir = f"{pair_dir}/dt_extreme_fkl_{int(FKL_LO)}_{int(FKL_HI)}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/tree_rules.txt", "w") as fh:
        fh.write(f"Pair: {PAIR}\n")
        fh.write(f"init_FKL bin: [{FKL_LO}, {FKL_HI})\n")
        fh.write(f"N samples: {len(sub)}\n")
        fh.write(f"Tree R²: {r2:.4f}\n")
        fh.write(f"Max depth: {MAX_DEPTH}\n\n")
        fh.write("Feature importances:\n")
        for f, imp in sorted(zip(FEATURES, tree.feature_importances_), key=lambda x: -x[1]):
            fh.write(f"  {f:40s}: {imp:.4f}\n")
        fh.write(f"\nTree rules:\n{rules_text}\n")

        # Per-leaf stats
        leaf_ids = tree.apply(X)
        fh.write(f"\nPer-leaf stats (leaf_id: n, mean_dfkl, std, min, max):\n")
        for lid in np.unique(leaf_ids):
            m = leaf_ids == lid
            fh.write(f"  leaf {lid:3d}: n={m.sum():5d}, "
                     f"mean={y[m].mean():+.4f}, std={y[m].std():.4f}, "
                     f"min={y[m].min():+.4f}, max={y[m].max():+.4f}\n")

    print(f"Saved rules to {out_dir}/tree_rules.txt")

    # === Plot tree ===
    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    # Compute symmetric vmax for color
    node_vals = tree.tree_.value.flatten()
    vabs = float(np.max(np.abs(node_vals)))
    plot_tree(tree, feature_names=FEATURES, filled=True,
              precision=3, fontsize=8, rounded=True, ax=ax,
              impurity=False, proportion=False)
    fig.suptitle(f"{PAIR}  |  init_FKL ∈ [{FKL_LO}, {FKL_HI})  |  N={len(sub)}  |  R²={r2:.3f}\n"
                 f"Regression tree of ΔFKL (leaf color: red=positive/learn-worse, blue=negative/learn-better)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_tree = f"{out_dir}/decision_tree.png"
    plt.savefig(out_tree, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_tree}")

    # === Bar plot: mean ΔFKL per leaf ===
    leaf_ids = tree.apply(X)
    leaf_stats = []
    for lid in np.unique(leaf_ids):
        m = leaf_ids == lid
        leaf_stats.append({
            "leaf": lid,
            "n": m.sum(),
            "mean_dfkl": y[m].mean(),
            "sem": y[m].std() / np.sqrt(m.sum()),
        })
    leaf_df = pd.DataFrame(leaf_stats).sort_values("mean_dfkl")

    fig2, ax2 = plt.subplots(1, 1, figsize=(max(8, len(leaf_df) * 0.5), 6))
    colors = ["#DC2626" if m > 0 else "#2563EB" for m in leaf_df["mean_dfkl"]]
    ax2.bar(range(len(leaf_df)), leaf_df["mean_dfkl"], yerr=leaf_df["sem"],
            color=colors, alpha=0.8, capsize=4)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xticks(range(len(leaf_df)))
    ax2.set_xticklabels([f"L{r['leaf']}\nn={r['n']}" for _, r in leaf_df.iterrows()],
                        rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Leaf (sorted by mean ΔFKL)", fontsize=9)
    ax2.set_ylabel("mean ΔFKL", fontsize=9)
    ax2.set_title(f"{PAIR}: per-leaf mean ΔFKL, init_FKL ∈ [{FKL_LO}, {FKL_HI})",
                  fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_bar = f"{out_dir}/per_leaf_dfkl.png"
    plt.savefig(out_bar, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_bar}")


def main_all():
    """Run decision tree analysis on ALL pair dirs under BASE."""
    global PAIR
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        if pair == "figures":
            continue
        if not glob.glob(f"{pair_dir}/step_*.parquet"):
            continue
        PAIR = pair
        print(f"\n{'='*70}\n=== {pair} ===\n{'='*70}", flush=True)
        try:
            main()
        except Exception as e:
            import traceback
            print(f"[error] {pair}: {e}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    if os.environ.get("MULTI_PAIR"):
        main_all()
    else:
        main()
