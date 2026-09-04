"""15-pair 2D grid (all C(6,2) combinations of features) stratified by init_FKL.

Adapted from user's script to our parquet schema:
  batch_idx     → sequence_id
  position      → response_pos
  pi_student_*  → pi_S_before / pi_S_after
  pi_teacher    → pi_T
  overlap_ratio → overlap_ratio_k8

Features (C(5,2) = 10 pairs, since init_fkl is the stratifier):
  student_entropy_topk, teacher_entropy, overlap_ratio_k8,
  student_mass_on_teacher_topp95, max_ps

Layout: 10 rows × 8 init_FKL bins per pair directory.
"""
import os, glob, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
BASE = os.environ.get("BASE", "/fsx/xinnanzh/data/update_consistency_multi_pair")
N_STEPS = int(os.environ.get("N_STEPS", "10"))
N_INNER = int(os.environ.get("N_INNER_BINS", "10"))
CUSTOM_INIT_FKL_EDGES = [0.0, 0.05, 0.2, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0]

RAW_COLS = [
    "sequence_id", "response_pos",
    "pi_S_before", "pi_S_after", "pi_T",
    "teacher_entropy",
    "overlap_k8", "overlap_ratio_k8",
    "teacher_mass_on_student_top8",
    "teacher_mass_on_student_topp95",
    "in_teacher_top8",
    "is_sampled_u", "A_value",
]

FEATURES = [
    ("student_entropy_topk",           "S_ent_topk"),
    ("teacher_entropy",                "T_ent"),
    ("overlap_ratio_k8",               "overlap_ratio"),
    ("student_mass_on_teacher_topp95", "S_mass_T_p95"),
    ("max_ps",                         "max_ps"),
    ("A_value",                        "A"),
]

AXIS_PAIRS = []
for i, j in itertools.combinations(range(len(FEATURES)), 2):
    fa, la = FEATURES[i]
    fb, lb = FEATURES[j]
    AXIS_PAIRS.append((fa, fb, la, lb))


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

    # A_value is per-position (constant across candidates). Take sampled-u row's A if present.
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

    # student_mass_on_teacher_topp95: sort by teacher desc, cumsum-based nucleus
    r_sorted = r.sort_values(["sequence_id", "response_pos", "_pt"],
                             ascending=[True, True, False]).copy()
    r_sorted["_cumsum_pt"] = r_sorted.groupby(["sequence_id", "response_pos"])["_pt"].cumsum()
    r_sorted["_prev"] = r_sorted["_cumsum_pt"] - r_sorted["_pt"]
    r_sorted["_in_p95"] = (r_sorted["_prev"] < 0.95).astype(float)
    r_sorted["_ps_if"] = r_sorted["_ps"] * r_sorted["_in_p95"]
    smt = r_sorted.groupby(["sequence_id", "response_pos"], as_index=False, sort=False).agg(
        student_mass_on_teacher_topp95=("_ps_if", "sum"))
    return agg.merge(smt, on=["sequence_id", "response_pos"], how="left")


def bin_edges(x, n):
    e = np.quantile(x, np.linspace(0, 1, n + 1))
    e = np.unique(e)
    if len(e) < 3:
        e = np.linspace(x.min(), x.max() + EPS, n + 1)
    return e


def make_heatmap(df, x_col, y_col):
    xe = bin_edges(df[x_col].values, N_INNER)
    ye = bin_edges(df[y_col].values, N_INNER)
    xi = np.clip(np.searchsorted(xe, df[x_col].values, side="right") - 1, 0, len(xe) - 2)
    yi = np.clip(np.searchsorted(ye, df[y_col].values, side="right") - 1, 0, len(ye) - 2)
    nx, ny = len(xe) - 1, len(ye) - 1
    mat = np.full((ny, nx), np.nan)
    v = df["delta_fkl"].values
    for iy in range(ny):
        for ix in range(nx):
            m = (xi == ix) & (yi == iy)
            if m.any():
                mat[iy, ix] = np.mean(v[m])
    return xe, ye, mat


def run_A_analysis(df, pair_dir, pair):
    """Two analyses of A_value's effect on ΔFKL:

    Figure A1: ΔFKL binned by A magnitude & sign, stratified by init_FKL bin.
      Line plot: x = |A| bin, y = mean ΔFKL, one line per (init_FKL bin, sign(A)).
      Shows whether |A|'s magnitude drives ΔFKL (not just sign).

    Figure A2: 2D heatmap of A vs each other feature, stratified by init_FKL.
      Rows = features, cols = init_FKL bins, cells colored by mean ΔFKL.
    """
    fkl_edges = np.array(CUSTOM_INIT_FKL_EDGES, dtype=np.float64)
    data_max = float(df["init_fkl"].max())
    fkl_edges = fkl_edges[fkl_edges <= data_max + EPS]
    if len(fkl_edges) < 2 or fkl_edges[-1] < data_max:
        fkl_edges = np.append(fkl_edges, data_max + EPS)
    fkl_labels = [f"[{fkl_edges[i]:.2g},{fkl_edges[i+1]:.2g}]"
                   for i in range(len(fkl_edges) - 1)]
    fkl_idx = np.clip(np.searchsorted(fkl_edges, df["init_fkl"].values, side="right") - 1,
                       0, len(fkl_edges) - 2)
    df = df.copy()
    df["_fkl_bin"] = fkl_idx

    # ============ Figure A1: line plot of ΔFKL vs raw A value, stratified by init_FKL ============
    # Use raw A bins so we can see the sign transition; asymmetric range because A is
    # typically much more negative than positive (student is overconfident on sampled u).
    a_edges = np.array([-np.inf, -15, -8, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf])
    a_labels = []
    for i in range(len(a_edges) - 1):
        lo, hi = a_edges[i], a_edges[i+1]
        if lo == -np.inf: a_labels.append(f"<{hi:g}")
        elif hi == np.inf: a_labels.append(f">{lo:g}")
        else: a_labels.append(f"[{lo:g},{hi:g})")
    df["_A_bin"] = pd.cut(df["A_value"], bins=a_edges, labels=a_labels)
    # Also record a numeric bin center for the x-axis position — use the midpoint of
    # each finite bin; for the two open-ended bins, use the edge itself as position.
    bin_centers = []
    for i in range(len(a_edges) - 1):
        lo, hi = a_edges[i], a_edges[i+1]
        if lo == -np.inf: bin_centers.append(hi - 3)  # -18
        elif hi == np.inf: bin_centers.append(lo + 1)  # 3
        else: bin_centers.append(0.5 * (lo + hi))

    n_cols = len(fkl_edges) - 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.5), squeeze=False)

    for c in range(n_cols):
        ax = axes[0, c]
        sub = df[df["_fkl_bin"] == c]
        if len(sub) < 100:
            ax.text(0.5, 0.5, f"N={len(sub)}", transform=ax.transAxes,
                    ha="center", va="center")
            ax.axis("off"); continue

        grp = sub.groupby("_A_bin", observed=False)["delta_fkl"].agg(["mean", "std", "count"])
        grp = grp[grp["count"] >= 10]
        if grp.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue

        # Position each point at the bin center, color by A sign at that center
        idx_present = [a_labels.index(l) for l in grp.index]
        xs = [bin_centers[i] for i in idx_present]
        colors = ["#DC2626" if bin_centers[i] > 0 else "#2563EB" for i in idx_present]
        means = grp["mean"].values
        sems = (grp["std"] / np.sqrt(grp["count"])).values

        ax.errorbar(xs, means, yerr=sems, color="#374151",
                    marker="none", lw=1.2, capsize=3, zorder=1)
        ax.scatter(xs, means, c=colors, s=45, zorder=2, edgecolor="black", lw=0.5)

        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="gray", lw=0.7, ls="--")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("A = log π_T(u) − log π_S(u)", fontsize=8)
        if c == 0:
            ax.set_ylabel("mean ΔFKL", fontsize=9)
        ax.set_title(f"init_FKL {fkl_labels[c]}\nN={len(sub)}", fontsize=8)

    fig.suptitle(f"{pair}: ΔFKL vs A (raw value), stratified by init_FKL  "
                 "[red dot: A>0 (student under-confident); blue dot: A<0 (student over-confident)]",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{pair_dir}/fkl_stratified"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/dfkl_vs_A_raw.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}", flush=True)

    # ============ Figure A2: A vs other features, 2D heatmap per init_FKL bin ============
    OTHER_FEATURES = [
        ("teacher_entropy", "T_ent"),
        ("student_entropy_topk", "S_ent"),
        ("overlap_ratio_k8", "overlap"),
        ("student_mass_on_teacher_topp95", "S_mass_T_p95"),
        ("max_ps", "max_ps"),
    ]
    fig2, axes2 = plt.subplots(len(OTHER_FEATURES), n_cols,
                                figsize=(3.5 * n_cols, 3.0 * len(OTHER_FEATURES)),
                                squeeze=False)

    for f_idx, (feat, feat_label) in enumerate(OTHER_FEATURES):
        mats = []
        for c in range(n_cols):
            sub = df[df["_fkl_bin"] == c]
            if len(sub) < 100: mats.append(None); continue
            _, _, mat = make_heatmap(sub, "A_value", feat)
            mats.append(mat)
        finite_parts = [m[np.isfinite(m)].ravel() for m in mats if m is not None]
        if not finite_parts: continue
        finite = np.concatenate(finite_parts)
        if len(finite) == 0: continue
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = -vmax

        for c in range(n_cols):
            ax = axes2[f_idx, c]
            sub = df[df["_fkl_bin"] == c]
            if len(sub) < 100 or mats[c] is None:
                ax.text(0.5, 0.5, f"N={len(sub)}", transform=ax.transAxes,
                        ha="center", va="center")
                ax.axis("off"); continue
            xe = bin_edges(sub["A_value"].values, N_INNER)
            ye = bin_edges(sub[feat].values, N_INNER)
            im = ax.imshow(mats[c], origin="lower", aspect="auto", cmap="RdBu_r",
                            vmin=vmin, vmax=vmax,
                            extent=[xe[0], xe[-1], ye[0], ye[-1]])
            ax.axvline(0, color="black", lw=0.5, alpha=0.6)  # A=0 line
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.set_xlabel("A_value", fontsize=7)
            ax.set_ylabel(feat_label, fontsize=7)
            if f_idx == 0:
                pct = 100.0 * len(sub) / len(df)
                ax.set_title(f"init_FKL {fkl_labels[c]}\nN={len(sub)} ({pct:.1f}%)",
                             fontsize=8, fontweight="bold")
            else:
                ax.set_title(f"{fkl_labels[c]}", fontsize=7)

    fig2.suptitle(f"{pair}: mean ΔFKL — A_value × feature, stratified by init_FKL",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out2 = f"{out_dir}/dfkl_A_vs_features.png"
    plt.savefig(out2, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}", flush=True)


def run_pair(pair_dir, pair):
    step_files = sorted(glob.glob(f"{pair_dir}/step_*.parquet"))[:N_STEPS]
    if not step_files:
        print(f"[skip] {pair}: no step files")
        return
    parts = []
    for f in step_files:
        raw = pd.read_parquet(f, columns=RAW_COLS)
        parts.append(build(raw))
    df = pd.concat(parts, ignore_index=True)
    print(f"[{pair}] N = {len(df)}", flush=True)

    fkl_edges = np.array(CUSTOM_INIT_FKL_EDGES, dtype=np.float64)
    data_max = float(df["init_fkl"].max())
    fkl_edges = fkl_edges[fkl_edges <= data_max + EPS]
    if len(fkl_edges) < 2 or fkl_edges[-1] < data_max:
        fkl_edges = np.append(fkl_edges, data_max + EPS)
    fkl_labels = [f"[{fkl_edges[i]:.3g}, {fkl_edges[i+1]:.3g}]"
                   for i in range(len(fkl_edges) - 1)]
    fkl_idx = np.clip(np.searchsorted(fkl_edges, df["init_fkl"].values, side="right") - 1,
                       0, len(fkl_edges) - 2)
    df["_bin"] = fkl_idx

    n_rows = len(AXIS_PAIRS)
    n_cols = len(fkl_edges) - 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows), squeeze=False)

    for r_idx, (x_col, y_col, x_lbl, y_lbl) in enumerate(AXIS_PAIRS):
        mats = []
        for c in range(n_cols):
            sub = df[df["_bin"] == c]
            if len(sub) < 100:
                mats.append(None); continue
            _, _, mat = make_heatmap(sub, x_col, y_col)
            mats.append(mat)
        finite_parts = [m[np.isfinite(m)].ravel() for m in mats if m is not None]
        if not finite_parts:
            continue
        finite = np.concatenate(finite_parts)
        if len(finite) == 0:
            continue
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = -vmax

        for c in range(n_cols):
            ax = axes[r_idx, c]
            sub = df[df["_bin"] == c]
            n = len(sub)
            if n < 100 or mats[c] is None:
                ax.text(0.5, 0.5, f"N={n}", transform=ax.transAxes, ha="center", va="center")
                ax.axis("off"); continue
            xe = bin_edges(sub[x_col].values, N_INNER)
            ye = bin_edges(sub[y_col].values, N_INNER)
            im = ax.imshow(mats[c], origin="lower", aspect="auto", cmap="RdBu_r",
                            vmin=vmin, vmax=vmax,
                            extent=[xe[0], xe[-1], ye[0], ye[-1]])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.set_xlabel(x_lbl, fontsize=7)
            ax.set_ylabel(y_lbl, fontsize=7)
            if r_idx == 0:
                pct = 100.0 * n / len(df)
                ax.set_title(f"init_FKL {fkl_labels[c]}\nN={n} ({pct:.1f}%)",
                             fontsize=8, fontweight="bold")
            else:
                ax.set_title(f"{fkl_labels[c]}", fontsize=7)

    fig.suptitle(f"{pair}: mean ΔFKL — {len(AXIS_PAIRS)} feature pairs × {n_cols} init_FKL bins  "
                 f"[first {N_STEPS} steps, N={len(df)}]",
                 y=1.001, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = f"{pair_dir}/fkl_stratified"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/fkl_2d_grid.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}", flush=True)

    # A-value analyses
    run_A_analysis(df, pair_dir, pair)


def main():
    only_pair = os.environ.get("ONLY_PAIR", "")
    pair_dirs = sorted(d for d in glob.glob(f"{BASE}/*") if os.path.isdir(d))
    for pair_dir in pair_dirs:
        pair = os.path.basename(pair_dir)
        if pair == "figures":
            continue
        if only_pair and only_pair not in pair:
            continue
        if not glob.glob(f"{pair_dir}/step_*.parquet"):
            continue
        try:
            run_pair(pair_dir, pair)
        except Exception as e:
            import traceback
            print(f"[error] {pair}: {e}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
