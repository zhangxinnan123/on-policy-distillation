"""For each pair, analyze the distribution of 'stuck' high-FKL positions.

Categorize positions with init_FKL > threshold into learnability buckets:
  - LEARNED_A_LOT   : ΔFKL <= -0.5   (large drop)
  - LEARNED_SOME    : -0.5 < ΔFKL <= -0.05
  - STUCK           : |ΔFKL| < 0.05  (no change)
  - GETTING_WORSE   : ΔFKL >= 0.05   (learning worse)

Also stratify STUCK positions by mechanism:
  - FROZEN          : max_ps > 0.95  (nucleus collapsed)
  - DISJOINT        : S_mass_T_p95 = 0  (student nucleus & teacher nucleus disjoint)
  - MISALIGNED_A    : |A| > 5 AND not frozen (large signal but batch update dominates)
  - OTHER

Outputs a stacked bar chart and per-pair heatmap.
"""
import os, glob, itertools
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
        S_mass_T_p95=("_ps_if", "sum"))
    return agg.merge(smt, on=["sequence_id", "response_pos"], how="left")


def categorize_learning(df):
    """Returns categorical column for learnability."""
    cats = np.full(len(df), "OTHER", dtype=object)
    dfkl = df["delta_fkl"].values
    cats[dfkl >= 0.05] = "GETTING_WORSE"
    cats[(dfkl < 0.05) & (dfkl > -0.05)] = "STUCK"
    cats[(dfkl <= -0.05) & (dfkl > -0.5)] = "LEARNED_SOME"
    cats[dfkl <= -0.5] = "LEARNED_A_LOT"
    return cats


def categorize_stuck_mechanism(df):
    """For each row, identify WHY it might be stuck."""
    mech = np.full(len(df), "OTHER", dtype=object)
    max_ps = df["max_ps"].values
    s_mass = df["S_mass_T_p95"].values
    A = np.abs(df["A_value"].values)

    mech[(s_mass < 0.01)] = "DISJOINT"          # priority: no overlap in nuclei
    mech[max_ps > 0.95] = "FROZEN"              # highest priority: nucleus collapse
    # Positions that are both frozen AND disjoint stay as FROZEN (overwrite happens second)
    # Wait, we want frozen to be checked first (most specific)
    mech = np.full(len(df), "OTHER", dtype=object)
    is_frozen = max_ps > 0.95
    is_disjoint = s_mass < 0.01
    is_large_A = A > 5

    # Priority: frozen → disjoint → misaligned A → other
    mech[is_large_A & ~is_frozen & ~is_disjoint] = "LARGE_A_ONLY"
    mech[is_disjoint & ~is_frozen] = "DISJOINT"
    mech[is_frozen & is_disjoint] = "FROZEN+DISJOINT"
    mech[is_frozen & ~is_disjoint] = "FROZEN"
    return mech


def main():
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    pair_dirs = [d for d in pair_dirs if os.path.basename(d) != "figures"]

    LEARN_CATS = ["LEARNED_A_LOT", "LEARNED_SOME", "STUCK", "GETTING_WORSE"]
    LEARN_COLORS = {"LEARNED_A_LOT": "#1e40af", "LEARNED_SOME": "#93c5fd",
                    "STUCK": "#d1d5db", "GETTING_WORSE": "#dc2626"}

    MECH_CATS = ["FROZEN", "FROZEN+DISJOINT", "DISJOINT", "LARGE_A_ONLY", "OTHER"]
    MECH_COLORS = {"FROZEN": "#6b7280", "FROZEN+DISJOINT": "#111827",
                   "DISJOINT": "#dc2626", "LARGE_A_ONLY": "#f59e0b", "OTHER": "#93c5fd"}

    all_stats = []
    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
        if not step_files: continue
        parts = [build(pd.read_parquet(f, columns=RAW_COLS)) for f in step_files]
        df = pd.concat(parts, ignore_index=True)

        # Filter to high-FKL positions
        sub = df[df["init_fkl"] >= FKL_THRESHOLD].copy()
        if len(sub) < 50:
            print(f"[skip] {pair}: only {len(sub)} high-FKL positions")
            continue

        sub["learn_cat"] = categorize_learning(sub)
        sub["mech_cat"] = categorize_stuck_mechanism(sub)

        # Per-pair fraction
        learn_frac = sub["learn_cat"].value_counts(normalize=True).reindex(LEARN_CATS).fillna(0)
        stuck_sub = sub[sub["learn_cat"].isin(["STUCK", "GETTING_WORSE"])]
        mech_frac = stuck_sub["mech_cat"].value_counts(normalize=True).reindex(MECH_CATS).fillna(0)

        print(f"\n{pair}  (N={len(sub)}, {100*len(sub)/len(df):.1f}% of {len(df)})")
        print(f"  Learn distribution: LEARNED_A_LOT={learn_frac['LEARNED_A_LOT']:.1%}, "
              f"LEARNED_SOME={learn_frac['LEARNED_SOME']:.1%}, "
              f"STUCK={learn_frac['STUCK']:.1%}, "
              f"WORSE={learn_frac['GETTING_WORSE']:.1%}")
        print(f"  Stuck mech: {mech_frac.to_dict()}")

        all_stats.append({
            "pair": pair,
            "n_total": len(df),
            "n_high_fkl": len(sub),
            "n_stuck": len(stuck_sub),
            **{f"learn_{k}": learn_frac[k] for k in LEARN_CATS},
            **{f"mech_{k}": mech_frac[k] for k in MECH_CATS},
        })

    stats_df = pd.DataFrame(all_stats)

    # === Figure 1: Learning-outcome stacked bar per pair ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    x = np.arange(len(stats_df))
    bottom = np.zeros(len(stats_df))
    for cat in LEARN_CATS:
        vals = stats_df[f"learn_{cat}"].values
        ax1.bar(x, vals, bottom=bottom, color=LEARN_COLORS[cat],
                label=cat, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace("___", "\n→ ").replace("_", "/", 2)
                         for p in stats_df["pair"]],
                        rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel(f"Fraction of positions with init_FKL ≥ {FKL_THRESHOLD}", fontsize=10)
    ax1.set_title(f"Learning outcome for high-FKL positions (init_FKL ≥ {FKL_THRESHOLD})",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Annotate N
    for i, r in stats_df.iterrows():
        pct = 100 * r["n_high_fkl"] / r["n_total"]
        ax1.text(i, 1.02, f"N={r['n_high_fkl']}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=6)

    # === Figure 2: Mechanism breakdown for STUCK+WORSE positions ===
    bottom = np.zeros(len(stats_df))
    for cat in MECH_CATS:
        vals = stats_df[f"mech_{cat}"].values
        ax2.bar(x, vals, bottom=bottom, color=MECH_COLORS[cat],
                label=cat, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace("___", "\n→ ").replace("_", "/", 2)
                         for p in stats_df["pair"]],
                        rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Fraction of STUCK+WORSE positions", fontsize=10)
    ax2.set_title("Mechanism of being stuck (among STUCK+WORSE only)",
                  fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")

    for i, r in stats_df.iterrows():
        ax2.text(i, 1.02, f"n_stuck={r['n_stuck']}",
                 ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/stuck_high_fkl_distribution.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")

    # === Figure 3: |ΔFKL| distribution per pair, split by learnability ===
    # For each pair, show histogram of ΔFKL for high-FKL positions
    fig2, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    for i, row in stats_df.iterrows():
        pair = row["pair"]
        pair_dir = f"{BASE}/{pair}"
        step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
        if not step_files: continue
        parts = [build(pd.read_parquet(f, columns=RAW_COLS)) for f in step_files]
        df = pd.concat(parts, ignore_index=True)
        sub = df[df["init_fkl"] >= FKL_THRESHOLD].copy()

        ax = axes[i // 3, i % 3]
        ax.hist(sub["delta_fkl"], bins=50, color="#93c5fd", edgecolor="white")
        ax.axvline(0, color="red", lw=1)
        ax.axvline(sub["delta_fkl"].median(), color="black", lw=1, ls="--",
                   label=f"median={sub['delta_fkl'].median():.2f}")
        ax.set_xlabel("ΔFKL", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        short = pair.replace("___", "\n→ ")
        ax.set_title(f"{short}\nN={len(sub)}, mean={sub['delta_fkl'].mean():.2f}",
                     fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for i in range(len(stats_df), 9):
        axes[i // 3, i % 3].set_visible(False)

    fig2.suptitle(f"ΔFKL distribution for high-FKL positions (init_FKL ≥ {FKL_THRESHOLD}) per pair",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = f"{out_dir}/dfkl_hist_high_fkl.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
