"""Characterize what makes high-FKL positions STUCK vs LEARNED.

For each pair, compare feature distributions between three groups:
  - LEARNED   : ΔFKL <= -0.05  (FKL decreased meaningfully)
  - STUCK     : -0.05 < ΔFKL < 0.05
  - WORSE     : ΔFKL >= 0.05

For high-FKL positions (init_FKL >= threshold), plot per-feature violin
or overlaid density, plus a "delta" summary heatmap showing which features
separate STUCK/WORSE from LEARNED.
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
    r["_neg_ps_logps"] = -ps * np.log(ps)
    r["_ps"] = ps
    r["_pt"] = pt

    agg = r.groupby(["sequence_id", "response_pos"], as_index=False, sort=False).agg(
        init_fkl=("_fkl_b", "sum"),
        fkl_a=("_fkl_a", "sum"),
        student_entropy_topk=("_neg_ps_logps", "sum"),
        max_ps=("_ps", "max"),
        teacher_entropy=("teacher_entropy", "first"),
        overlap_ratio_k8=("overlap_ratio_k8", "first"),
        A_value=("A_value", "first"),
    )
    agg["delta_fkl"] = agg["fkl_a"] - agg["init_fkl"]
    agg["abs_A"] = agg["A_value"].abs()

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


FEATURES = [
    ("max_ps", "max π_S", (0, 1)),
    ("S_mass_T_p95", "π_S mass on T-nucleus", (0, 1)),
    ("teacher_entropy", "Teacher entropy", None),
    ("student_entropy_topk", "Student entropy (top-K)", None),
    ("overlap_ratio_k8", "Overlap ratio k8", (0, 1)),
    ("abs_A", "|A|", (0, 15)),
    ("init_fkl", "init FKL", None),
]


def classify(dfkl):
    cats = np.full(len(dfkl), "STUCK", dtype=object)
    cats[dfkl <= -0.05] = "LEARNED"
    cats[dfkl >= 0.05] = "WORSE"
    return cats


def main():
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    pair_dirs = [d for d in pair_dirs if os.path.basename(d) != "figures"]

    # Collect per-pair per-group summaries
    summary_rows = []
    per_pair_data = {}

    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
        if not step_files: continue
        parts = [build(pd.read_parquet(f, columns=RAW_COLS)) for f in step_files]
        df = pd.concat(parts, ignore_index=True)
        sub = df[df["init_fkl"] >= FKL_THRESHOLD].copy()
        if len(sub) < 100:
            print(f"[skip] {pair}: only {len(sub)} high-FKL positions")
            continue
        sub["group"] = classify(sub["delta_fkl"].values)
        per_pair_data[pair] = sub

        for group in ["LEARNED", "STUCK", "WORSE"]:
            g = sub[sub["group"] == group]
            if len(g) < 10: continue
            row = {"pair": pair, "group": group, "n": len(g)}
            for feat, _, _ in FEATURES:
                row[f"{feat}_median"] = float(g[feat].median())
                row[f"{feat}_q25"] = float(g[feat].quantile(0.25))
                row[f"{feat}_q75"] = float(g[feat].quantile(0.75))
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print(summary[["pair", "group", "n"] + [f"{f}_median" for f, _, _ in FEATURES]].to_string(index=False))

    # === Figure 1: Per-pair per-feature violin, 3 groups side by side ===
    pairs = list(per_pair_data.keys())
    n_pairs = len(pairs)
    n_feat = len(FEATURES)

    fig, axes = plt.subplots(n_pairs, n_feat, figsize=(2.8 * n_feat, 2.5 * n_pairs), squeeze=False)
    GROUP_COLORS = {"LEARNED": "#2563EB", "STUCK": "#9CA3AF", "WORSE": "#DC2626"}

    for r, pair in enumerate(pairs):
        sub = per_pair_data[pair]
        for c, (feat, feat_label, ylim) in enumerate(FEATURES):
            ax = axes[r, c]
            data = []
            labels = []
            colors = []
            for group in ["LEARNED", "STUCK", "WORSE"]:
                g_vals = sub[sub["group"] == group][feat].dropna().values
                if len(g_vals) < 5: continue
                # Clip extreme outliers for readability
                if ylim is None:
                    lo, hi = np.percentile(g_vals, [1, 99])
                    g_vals = g_vals[(g_vals >= lo) & (g_vals <= hi)]
                data.append(g_vals)
                labels.append(f"{group}\nn={len(g_vals)}")
                colors.append(GROUP_COLORS[group])

            if not data: continue
            parts = ax.violinplot(data, positions=range(len(data)),
                                  showmedians=True, widths=0.75)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color); pc.set_alpha(0.6)
                pc.set_edgecolor(color)
            for k in ("cbars", "cmins", "cmaxes", "cmedians"):
                if k in parts:
                    parts[k].set_color("black"); parts[k].set_linewidth(0.8)

            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(labels, fontsize=6)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=0.2, axis="y")
            if r == 0:
                ax.set_title(feat_label, fontsize=9, fontweight="bold")
            if c == 0:
                short = pair.replace("___", "\n→ ")
                ax.set_ylabel(f"{short}", fontsize=6, fontweight="bold")

    fig.suptitle(f"Feature distributions per learning-outcome group (init_FKL ≥ {FKL_THRESHOLD})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{BASE}/figures/learnable_rkl"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/stuck_feature_violins.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")

    # === Figure 2: Heatmap — median feature difference (STUCK/WORSE − LEARNED) per pair ===
    # For each pair, compute effect size: (median_STUCK - median_LEARNED) / IQR_all
    # Also (median_WORSE - median_LEARNED) / IQR_all
    fig2, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(14, max(4, 0.5 * n_pairs)))

    def diff_matrix(target_group):
        rows = []
        for pair in pairs:
            sub = per_pair_data[pair]
            learned = sub[sub["group"] == "LEARNED"]
            target = sub[sub["group"] == target_group]
            if len(learned) < 10 or len(target) < 10:
                rows.append([np.nan] * n_feat); continue
            row = []
            for feat, _, _ in FEATURES:
                med_l = learned[feat].median()
                med_t = target[feat].median()
                iqr_all = sub[feat].quantile(0.75) - sub[feat].quantile(0.25)
                if iqr_all < 1e-9: iqr_all = 1e-9
                row.append((med_t - med_l) / iqr_all)
            rows.append(row)
        return np.array(rows)

    M_s = diff_matrix("STUCK")
    M_w = diff_matrix("WORSE")
    vmax = float(np.nanmax(np.abs(np.concatenate([M_s.ravel(), M_w.ravel()]))))
    vmin = -vmax

    for ax, M, title in [(ax_s, M_s, "STUCK vs LEARNED"), (ax_w, M_w, "WORSE vs LEARNED")]:
        im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels([f_lbl for _, f_lbl, _ in FEATURES], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_pairs))
        ax.set_yticklabels([p.replace("___", " → ") for p in pairs], fontsize=7)
        ax.set_title(f"{title}\n(median diff / IQR — red=target higher, blue=lower)",
                     fontsize=10, fontweight="bold")
        # Annotate each cell
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                            fontsize=6, color="black")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig2.suptitle(f"Feature median shift: STUCK/WORSE vs LEARNED (init_FKL ≥ {FKL_THRESHOLD})",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = f"{out_dir}/stuck_feature_shift_heatmap.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
