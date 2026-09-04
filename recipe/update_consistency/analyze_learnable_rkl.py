"""Analyze per-position learnable RKL decomposition.

Computes five derived quantities per position from saved parquet:
  μ  — corrected student nucleus mass (max(0.95, max_c π_S(c)))
  β  — teacher mass on student's p=0.95 nucleus (teacher_mass_on_student_topp95)
  kl2(μ, β) — RKL floor (mass-mismatch term)
  L_RKL — learnable RKL (conditional KL within nucleus)
  J_A — signal variance (variance of A_c within nucleus, weighted by π̃)

Key insight: when max π_S > 0.95, nucleus collapses to single token →
L_RKL = 0 (constructive freeze). This phase transition is the main plot.
"""
import warnings
warnings.filterwarnings("ignore")
import os, argparse
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_position_quantities(df):
    """From candidate-level parquet rows, compute per-position derived quantities."""
    groups = df.groupby(["step", "sequence_id", "response_pos"])
    records = []

    for (step, seq_id, pos), g in groups:
        pi_T = g["pi_T"].values
        pi_S_b = g["pi_S_before"].values
        pi_S_a = g["pi_S_after"].values
        is_sampled = g["is_sampled_u"].values

        # Need enough candidates
        if len(g) < 3:
            continue

        # Student max prob (best available from candidates + sampled)
        student_max_pi = pi_S_b.max()

        # μ: corrected nucleus mass
        mu = max(0.95, student_max_pi)

        # β: teacher mass on student nucleus (per-position scalar, same for all rows)
        row0 = g.iloc[0]
        beta = row0["teacher_mass_on_student_topp95"]

        # Phase transition flag
        is_single_token_nucleus = student_max_pi > 0.95

        # kl2(μ, β) — RKL floor
        safe_mu = np.clip(mu, 1e-10, 1 - 1e-10)
        safe_beta = np.clip(beta, 1e-10, 1 - 1e-10)
        kl2_mu_beta = (
            safe_mu * np.log(safe_mu / safe_beta)
            + (1 - safe_mu) * np.log((1 - safe_mu) / (1 - safe_beta))
        )

        # kl2(β, μ) — FKL floor
        kl2_beta_mu = (
            safe_beta * np.log(safe_beta / safe_mu)
            + (1 - safe_beta) * np.log((1 - safe_beta) / (1 - safe_mu))
        )

        # Build nucleus mask (p=0.95) within these candidates
        # Sort by student prob descending, take until cumprob crosses 0.95
        non_sampled = ~is_sampled
        # Use all candidates (including sampled) for nucleus computation
        sorted_idx = np.argsort(-pi_S_b)
        sorted_probs = pi_S_b[sorted_idx]
        cumprob = np.cumsum(sorted_probs)
        # Token in nucleus if cumprob before it (exclusive) < 0.95
        nucleus_sorted = (cumprob - sorted_probs) < 0.95
        nucleus_mask = np.zeros(len(g), dtype=bool)
        nucleus_mask[sorted_idx] = nucleus_sorted

        nucleus_pi_S = pi_S_b[nucleus_mask]
        nucleus_pi_T = pi_T[nucleus_mask]

        if nucleus_mask.sum() == 0 or mu < 1e-10 or beta < 1e-10:
            l_rkl = 0.0
            j_a = 0.0
        else:
            # π̃_c = π_S(c) / μ,  p̃_c = π_T(c) / β
            pi_S_tilde = nucleus_pi_S / mu
            pi_T_tilde = nucleus_pi_T / beta

            # L_RKL = μ * Σ π̃_c (log π̃_c − log p̃_c)
            log_ratio = (np.log(np.maximum(pi_S_tilde, 1e-30))
                         - np.log(np.maximum(pi_T_tilde, 1e-30)))
            l_rkl = mu * np.sum(pi_S_tilde * log_ratio)
            l_rkl = max(l_rkl, 0.0)  # clamp numerical noise

            # J_A = Σ π̃_c (A_c − Ā)²
            A_c = (np.log(np.maximum(nucleus_pi_T, 1e-30))
                   - np.log(np.maximum(nucleus_pi_S, 1e-30)))
            A_bar = np.sum(pi_S_tilde * A_c)
            j_a = np.sum(pi_S_tilde * (A_c - A_bar) ** 2)

        # Student mass on TEACHER's nucleus (p=0.95) — reverse of β
        # Build teacher nucleus (p=0.95): sort π_T desc, cumsum < 0.95 → include
        sorted_idx_T = np.argsort(-pi_T)
        sorted_probs_T = pi_T[sorted_idx_T]
        cumprob_T = np.cumsum(sorted_probs_T)
        teacher_nucleus_sorted = (cumprob_T - sorted_probs_T) < 0.95
        teacher_nucleus_mask = np.zeros(len(g), dtype=bool)
        teacher_nucleus_mask[sorted_idx_T] = teacher_nucleus_sorted
        student_mass_on_teacher_nuc = float(pi_S_b[teacher_nucleus_mask].sum())

        # FKL (actual, for reference)
        valid = (pi_T > 1e-30) & (pi_S_b > 1e-30) & (pi_S_a > 1e-30)
        if valid.sum() >= 3:
            fkl_before = np.sum(pi_T[valid] * np.log(pi_T[valid] / pi_S_b[valid]))
            fkl_after = np.sum(pi_T[valid] * np.log(pi_T[valid] / pi_S_a[valid]))
            delta_fkl = fkl_after - fkl_before
            # RKL(π_S || π_T) over teacher top-K candidates (approximation)
            rkl_before = np.sum(pi_S_b[valid] * np.log(pi_S_b[valid] / pi_T[valid]))
            rkl_after = np.sum(pi_S_a[valid] * np.log(pi_S_a[valid] / pi_T[valid]))
            delta_rkl = rkl_after - rkl_before
        else:
            fkl_before = np.nan
            fkl_after = np.nan
            delta_fkl = np.nan
            rkl_before = np.nan
            rkl_after = np.nan
            delta_rkl = np.nan

        # ε = 1 − max π_S (frozen tail mass)
        epsilon = 1.0 - student_max_pi

        records.append({
            "step": step,
            "sequence_id": seq_id,
            "response_pos": pos,
            "mu": mu,
            "beta": beta,
            "student_max_pi": student_max_pi,
            "is_single_token_nucleus": is_single_token_nucleus,
            "kl2_mu_beta": kl2_mu_beta,
            "kl2_beta_mu": kl2_beta_mu,
            "l_rkl": l_rkl,
            "j_a": j_a,
            "epsilon": epsilon,
            "nucleus_size": int(nucleus_mask.sum()),
            "fkl_before": fkl_before,
            "delta_fkl": delta_fkl,
            "rkl_before": rkl_before,
            "delta_rkl": delta_rkl,
            "student_mass_on_teacher_nuc": student_mass_on_teacher_nuc,
            "teacher_entropy": row0["teacher_entropy"],
            "overlap_ratio_k8": row0["overlap_ratio_k8"],
            "A_value": row0["A_value"],
            "entropy_S": row0["entropy_S"],
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=5000)
    args = parser.parse_args()

    pairs = sorted([d for d in os.listdir(args.data_dir)
                    if os.path.isdir(os.path.join(args.data_dir, d))])
    print(f"Found {len(pairs)} pairs")

    out_dir = os.path.join(args.data_dir, "figures", "learnable_rkl")
    os.makedirs(out_dir, exist_ok=True)

    all_pos_dfs = []

    for pair in pairs:
        pair_dir = os.path.join(args.data_dir, pair)
        step_file = os.path.join(pair_dir, f"step_{args.step:06d}.parquet")
        if not os.path.exists(step_file):
            print(f"  {pair}: no data, skipping")
            continue

        print(f"  {pair}: loading...")
        df = pd.read_parquet(step_file)

        positions = df[["step", "sequence_id", "response_pos"]].drop_duplicates()
        if len(positions) > args.max_positions:
            positions = positions.sample(args.max_positions, random_state=42)
            df = df.merge(positions, on=["step", "sequence_id", "response_pos"])

        pos_df = compute_position_quantities(df)
        if len(pos_df) < 10:
            print(f"  {pair}: too few positions, skipping")
            continue

        pos_df["pair"] = pair
        all_pos_dfs.append(pos_df)
        print(f"  {pair}: {len(pos_df)} positions, "
              f"single-token nucleus: {pos_df.is_single_token_nucleus.mean():.1%}")

    if not all_pos_dfs:
        print("No data found")
        return

    all_df = pd.concat(all_pos_dfs, ignore_index=True)
    pairs_with_data = all_df["pair"].unique()
    n_pairs = len(pairs_with_data)

    # === Figure 1: Phase transition — L_RKL vs student_max_pi ===
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, pair in enumerate(pairs_with_data):
        ax = axes1[idx // n_cols, idx % n_cols]
        sub = all_df[all_df["pair"] == pair]

        sc = ax.scatter(sub["student_max_pi"], sub["l_rkl"],
                        c=sub["beta"], cmap="RdYlGn", vmin=0, vmax=1,
                        s=8, alpha=0.4, edgecolors="none")
        ax.axvline(0.95, color="red", lw=1.5, ls="--", alpha=0.7, label="p=0.95 threshold")
        ax.set_xlabel("max π_S (student top-1 prob)", fontsize=9)
        ax.set_ylabel("L_RKL (learnable RKL)", fontsize=9)
        short = pair.replace("___", "\n→ ")
        ax.set_title(f"{short}\nN={len(sub)}, frozen={sub.is_single_token_nucleus.mean():.0%}",
                     fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)

    for idx in range(n_pairs, n_rows * n_cols):
        axes1[idx // n_cols, idx % n_cols].set_visible(False)

    fig1.suptitle("Phase transition: L_RKL → 0 when max π_S > 0.95\n"
                  "(color = β = teacher mass on student nucleus)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out1 = os.path.join(out_dir, "phase_transition_l_rkl.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved {out1}")

    # === Figure 2: Five quantities summary — violin/box by pair ===
    quantities = ["kl2_mu_beta", "kl2_beta_mu", "l_rkl", "j_a", "epsilon"]
    q_labels = ["kl2(μ,β)\nRKL floor", "kl2(β,μ)\nFKL floor",
                "L_RKL\nlearnable", "J_A\nsignal var", "ε\nfrozen tail"]

    fig2, axes2 = plt.subplots(1, len(quantities), figsize=(4 * len(quantities), 5), squeeze=False)
    for col, (q, qlabel) in enumerate(zip(quantities, q_labels)):
        ax = axes2[0, col]
        data_per_pair = []
        labels = []
        for pair in pairs_with_data:
            sub = all_df[all_df["pair"] == pair]
            data_per_pair.append(sub[q].values)
            labels.append(pair.split("___")[0][:12] + "\n→" + pair.split("___")[1][:12]
                          if "___" in pair else pair[:20])

        bp = ax.boxplot(data_per_pair, labels=labels, vert=True, patch_artist=True,
                        showfliers=False, medianprops=dict(color="red", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#4A90D9")
            patch.set_alpha(0.5)
        ax.set_title(qlabel, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.grid(True, alpha=0.2, axis="y")

    fig2.suptitle(f"Per-position decomposition (step {args.step})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out2 = os.path.join(out_dir, "five_quantities_boxplot.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")

    # === Figure 3: kl2(μ,β) vs β scatter — shows floor growth ===
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, pair in enumerate(pairs_with_data):
        ax = axes3[idx // n_cols, idx % n_cols]
        sub = all_df[all_df["pair"] == pair]

        ax.scatter(sub["beta"], sub["kl2_mu_beta"],
                   c=sub["student_max_pi"], cmap="plasma", vmin=0.5, vmax=1.0,
                   s=8, alpha=0.4, edgecolors="none")
        # Theoretical curve: kl2(0.95, β) for reference
        beta_range = np.linspace(0.01, 0.99, 200)
        kl2_curve = 0.95 * np.log(0.95 / beta_range) + 0.05 * np.log(0.05 / (1 - beta_range))
        ax.plot(beta_range, kl2_curve, "k--", lw=1.2, alpha=0.6, label="kl2(0.95, β)")
        ax.axvline(0.7, color="orange", lw=1, ls=":", alpha=0.7, label="β=0.7 warning")
        ax.set_xlabel("β (teacher mass on student nucleus)", fontsize=9)
        ax.set_ylabel("kl2(μ, β) — RKL floor", fontsize=9)
        short = pair.replace("___", "\n→ ")
        ax.set_title(f"{short}", fontsize=8)
        ax.set_ylim(bottom=-0.05)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)

    for idx in range(n_pairs, n_rows * n_cols):
        axes3[idx // n_cols, idx % n_cols].set_visible(False)

    fig3.suptitle("RKL floor kl2(μ,β) vs β\n(color = max π_S; dashed = μ=0.95 reference)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out3 = os.path.join(out_dir, "rkl_floor_vs_beta.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {out3}")

    # === Figure 4: ΔFKL vs L_RKL + kl2 decomposition ===
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, pair in enumerate(pairs_with_data):
        ax = axes4[idx // n_cols, idx % n_cols]
        sub = all_df[all_df["pair"] == pair].dropna(subset=["delta_fkl"])
        if len(sub) < 10:
            ax.set_visible(False)
            continue

        total_kl0 = sub["l_rkl"] + sub["kl2_mu_beta"]
        sc = ax.scatter(total_kl0, sub["delta_fkl"],
                        c=sub["is_single_token_nucleus"].astype(float),
                        cmap="coolwarm", vmin=0, vmax=1,
                        s=8, alpha=0.4, edgecolors="none")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("L_RKL + kl2(μ,β) ≈ full RKL", fontsize=9)
        ax.set_ylabel("ΔFKL (actual)", fontsize=9)
        short = pair.replace("___", "\n→ ")
        frozen_frac = sub["is_single_token_nucleus"].mean()
        ax.set_title(f"{short}\nfrozen={frozen_frac:.0%}", fontsize=8)
        ax.grid(True, alpha=0.2)

    for idx in range(n_pairs, n_rows * n_cols):
        axes4[idx // n_cols, idx % n_cols].set_visible(False)

    fig4.suptitle("ΔFKL vs RKL decomposition (L_RKL + floor)\n"
                  "(red = single-token nucleus / frozen)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out4 = os.path.join(out_dir, "delta_fkl_vs_decomposition.png")
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved {out4}")

    # === Figure 5: Phase transition detail — ΔFKL vs max π_S ===
    fig5, axes5 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, pair in enumerate(pairs_with_data):
        ax = axes5[idx // n_cols, idx % n_cols]
        sub = all_df[all_df["pair"] == pair].dropna(subset=["delta_fkl"])
        if len(sub) < 10:
            ax.set_visible(False)
            continue

        sc = ax.scatter(sub["student_max_pi"], sub["delta_fkl"],
                        c=sub["l_rkl"], cmap="viridis",
                        vmin=0, vmax=sub["l_rkl"].quantile(0.95),
                        s=8, alpha=0.4, edgecolors="none")
        ax.axvline(0.95, color="red", lw=1.5, ls="--", alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("max π_S", fontsize=9)
        ax.set_ylabel("ΔFKL", fontsize=9)
        short = pair.replace("___", "\n→ ")
        ax.set_title(f"{short}", fontsize=8)
        ax.grid(True, alpha=0.2)
        fig5.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="L_RKL")

    for idx in range(n_pairs, n_rows * n_cols):
        axes5[idx // n_cols, idx % n_cols].set_visible(False)

    fig5.suptitle("ΔFKL vs max π_S — phase transition at p=0.95\n"
                  "(color = L_RKL; red dashed = threshold)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out5 = os.path.join(out_dir, "delta_fkl_vs_max_pi.png")
    plt.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"Saved {out5}")

    # === Figure 6b: Initial FKL vs FKL floor ===
    # FKL is better-approximated by top-K sum than RKL (teacher mass concentrated in top-K).
    # FKL floor = kl2(β, μ) — direction flipped from RKL floor.
    fig6b, axes6b = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, pair in enumerate(pairs_with_data):
        ax = axes6b[idx // n_cols, idx % n_cols]
        sub = all_df[all_df["pair"] == pair].dropna(subset=["fkl_before"])
        if len(sub) < 10:
            ax.set_visible(False)
            continue

        fkl_b = sub["fkl_before"].values
        floor = sub["kl2_beta_mu"].values
        frozen = sub["is_single_token_nucleus"].values

        lim = max(np.percentile(fkl_b, 99), np.percentile(floor, 99), 1e-3)
        ax.scatter(fkl_b[~frozen], floor[~frozen], s=8, alpha=0.4,
                   c="#2563EB", edgecolors="none", label="non-frozen")
        ax.scatter(fkl_b[frozen], floor[frozen], s=8, alpha=0.5,
                   c="#DC2626", edgecolors="none", label="frozen")
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6, label="floor = FKL")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("FKL_before (initial)", fontsize=9)
        ax.set_ylabel("kl2(β, μ) — FKL floor", fontsize=9)
        short = pair.replace("___", "\n→ ")
        floor_dominated = ((floor >= 0.5 * fkl_b) & (fkl_b > 0.01)).mean()
        ax.set_title(f"{short}\nfloor≥50% FKL: {floor_dominated:.0%}", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="upper left")

    for idx in range(n_pairs, n_rows * n_cols):
        axes6b[idx // n_cols, idx % n_cols].set_visible(False)

    fig6b.suptitle("Initial FKL vs FKL floor kl2(β, μ)",
                   fontsize=12, fontweight="bold")
    plt.tight_layout()
    out6b = os.path.join(out_dir, "fkl_before_vs_floor.png")
    plt.savefig(out6b, dpi=150, bbox_inches="tight")
    plt.close(fig6b)
    print(f"Saved {out6b}")

    # === Figure 7b: FKL floor share by FKL bin ===
    fig7b, ax7b = plt.subplots(1, 1, figsize=(10, 6))
    fkl_bins = np.array([0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf])
    bin_centers_f = ["<0.05", "0.05-0.1", "0.1-0.3", "0.3-0.5", "0.5-1", "1-2", "2-5", ">5"]

    for pair in pairs_with_data:
        sub = all_df[all_df["pair"] == pair].dropna(subset=["fkl_before"])
        sub = sub[sub["fkl_before"] > 1e-6]
        floor_share = np.clip(sub["kl2_beta_mu"] / sub["fkl_before"], 0, 1)
        sub = sub.assign(floor_share=floor_share)
        sub["fkl_bin"] = pd.cut(sub["fkl_before"], bins=fkl_bins, labels=bin_centers_f)
        mean_share = sub.groupby("fkl_bin", observed=False)["floor_share"].mean()

        short = pair.replace("___", "→").replace("_", "/", 2)[:35]
        ax7b.plot(range(len(bin_centers_f)), mean_share.values,
                  marker="o", lw=1.5, alpha=0.75, label=short)

    ax7b.set_xticks(range(len(bin_centers_f)))
    ax7b.set_xticklabels(bin_centers_f, rotation=30, ha="right")
    ax7b.set_xlabel("FKL_before bin", fontsize=10)
    ax7b.set_ylabel("mean FKL floor / FKL_before", fontsize=10)
    ax7b.set_ylim(0, 1.05)
    ax7b.axhline(0.5, color="red", ls=":", lw=1, alpha=0.5, label="50% floor")
    ax7b.grid(True, alpha=0.3)
    ax7b.legend(fontsize=7, loc="upper right", ncol=2)
    ax7b.set_title("FKL floor share by FKL magnitude\n"
                   "kl2(β,μ) captures the frozen-tail contribution to teacher-side KL",
                   fontsize=11, fontweight="bold")
    plt.tight_layout()
    out7b = os.path.join(out_dir, "fkl_floor_share_by_bin.png")
    plt.savefig(out7b, dpi=150, bbox_inches="tight")
    plt.close(fig7b)
    print(f"Saved {out7b}")

    # === Figure 6: Initial RKL vs floor ===
    # Two panels per pair: (a) scatter of RKL_before vs floor with y=x diagonal;
    # (b) stacked "floor share" bar showing what fraction of initial RKL is
    # unlearnable across bins of RKL magnitude.
    fig6, axes6 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, pair in enumerate(pairs_with_data):
        ax = axes6[idx // n_cols, idx % n_cols]
        sub = all_df[all_df["pair"] == pair].dropna(subset=["rkl_before"])
        if len(sub) < 10:
            ax.set_visible(False)
            continue

        # RKL_before vs floor kl2(mu, beta)
        # Color by whether frozen; size by learnable magnitude
        rkl_b = sub["rkl_before"].values
        floor = sub["kl2_mu_beta"].values
        frozen = sub["is_single_token_nucleus"].values

        lim = max(np.percentile(rkl_b, 99), np.percentile(floor, 99), 1e-3)
        ax.scatter(rkl_b[~frozen], floor[~frozen], s=8, alpha=0.4,
                   c="#2563EB", edgecolors="none", label="non-frozen")
        ax.scatter(rkl_b[frozen], floor[frozen], s=8, alpha=0.5,
                   c="#DC2626", edgecolors="none", label="frozen")
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6, label="floor = RKL")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("RKL_before (initial)", fontsize=9)
        ax.set_ylabel("kl2(μ, β) — floor", fontsize=9)
        short = pair.replace("___", "\n→ ")
        # Fraction of positions where floor >= 50% of RKL_before
        floor_dominated = ((floor >= 0.5 * rkl_b) & (rkl_b > 0.01)).mean()
        ax.set_title(f"{short}\nfloor≥50% RKL: {floor_dominated:.0%}", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="upper left")

    for idx in range(n_pairs, n_rows * n_cols):
        axes6[idx // n_cols, idx % n_cols].set_visible(False)

    fig6.suptitle("Initial RKL vs floor kl2(μ,β)\n"
                  "(points above y=x line: floor exceeds total RKL — pure mass-mismatch position)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out6 = os.path.join(out_dir, "rkl_before_vs_floor.png")
    plt.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print(f"Saved {out6}")

    # === Figure 7: Floor share of initial RKL, binned by RKL magnitude ===
    # For each pair, group positions into RKL bins and plot mean(floor/rkl) per bin.
    fig7, ax7 = plt.subplots(1, 1, figsize=(10, 6))
    rkl_bins = np.array([0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf])
    bin_centers = ["<0.05", "0.05-0.1", "0.1-0.3", "0.3-0.5", "0.5-1", "1-2", "2-5", ">5"]

    for pair in pairs_with_data:
        sub = all_df[all_df["pair"] == pair].dropna(subset=["rkl_before"])
        sub = sub[sub["rkl_before"] > 1e-6]
        floor_share = np.clip(sub["kl2_mu_beta"] / sub["rkl_before"], 0, 1)
        sub = sub.assign(floor_share=floor_share)
        sub["rkl_bin"] = pd.cut(sub["rkl_before"], bins=rkl_bins, labels=bin_centers)
        mean_share = sub.groupby("rkl_bin", observed=False)["floor_share"].mean()

        short = pair.replace("___", "→").replace("_", "/", 2)[:35]
        ax7.plot(range(len(bin_centers)), mean_share.values,
                 marker="o", lw=1.5, alpha=0.75, label=short)

    ax7.set_xticks(range(len(bin_centers)))
    ax7.set_xticklabels(bin_centers, rotation=30, ha="right")
    ax7.set_xlabel("RKL_before bin", fontsize=10)
    ax7.set_ylabel("mean floor / RKL_before", fontsize=10)
    ax7.set_ylim(0, 1.05)
    ax7.axhline(0.5, color="red", ls=":", lw=1, alpha=0.5, label="50% floor")
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=7, loc="upper right", ncol=2)
    ax7.set_title("Floor share of initial RKL (per pair, binned by RKL magnitude)\n"
                  "High share → mass mismatch dominates → unlearnable under nucleus RKL",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    out7 = os.path.join(out_dir, "floor_share_by_rkl_bin.png")
    plt.savefig(out7, dpi=150, bbox_inches="tight")
    plt.close(fig7)
    print(f"Saved {out7}")

    # === Figure 8: FKL_before vs floor, colored by different features ===
    # Each pair one row, each feature one column. Reveals which features
    # separate the "floor-dominated" region from the "learnable" region.
    COLOR_FEATS = [
        ("teacher_entropy", "Teacher entropy", "viridis", None, None),
        ("entropy_S", "Student entropy", "plasma", None, None),
        ("student_max_pi", "max π_S", "magma", 0.0, 1.0),
        ("beta", "β (T-mass on S-nucleus)", "RdYlGn", 0.0, 1.0),
        ("overlap_ratio_k8", "Overlap ratio k8", "RdYlBu", 0.0, 1.0),
        ("A_value", "A = log π_T(u) − log π_S(u)", "coolwarm", None, None),
    ]
    n_feat = len(COLOR_FEATS)

    fig8, axes8 = plt.subplots(n_pairs, n_feat, figsize=(4.2 * n_feat, 3.7 * n_pairs), squeeze=False)

    for row, pair in enumerate(pairs_with_data):
        sub = all_df[all_df["pair"] == pair].dropna(subset=["fkl_before"])
        if len(sub) < 10:
            for col in range(n_feat):
                axes8[row, col].set_visible(False)
            continue

        x = sub["fkl_before"].values
        y = sub["kl2_mu_beta"].values  # RKL floor: exact binary KL, always ≤ true FKL when μ > β
        lim = max(np.percentile(x, 99), np.percentile(y, 99), 1e-3)

        for col, (feat, feat_label, cmap_name, vmin, vmax) in enumerate(COLOR_FEATS):
            ax = axes8[row, col]
            c_vals = sub[feat].values
            vmin_use = np.percentile(c_vals, 2) if vmin is None else vmin
            vmax_use = np.percentile(c_vals, 98) if vmax is None else vmax

            sc = ax.scatter(x, y, c=c_vals, cmap=cmap_name,
                            vmin=vmin_use, vmax=vmax_use,
                            s=7, alpha=0.5, edgecolors="none")
            ax.plot([0, lim], [0, lim], "k--", lw=0.6, alpha=0.5)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(feat_label, fontsize=9, fontweight="bold")
            if col == 0:
                short = pair.replace("___", "\n→ ")
                ax.set_ylabel(f"{short}\nkl2(μ,β) — RKL floor", fontsize=6, fontweight="bold")
            if row == n_pairs - 1:
                ax.set_xlabel("FKL_before", fontsize=8)

            if row == 0:
                cbar = fig8.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
                cbar.ax.tick_params(labelsize=6)

    fig8.suptitle("FKL_before vs RKL floor kl2(μ,β), colored by features",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out8 = os.path.join(out_dir, "fkl_before_vs_floor_colored.png")
    plt.savefig(out8, dpi=150, bbox_inches="tight")
    plt.close(fig8)
    print(f"Saved {out8}")

    # === Figure 9: Feature distribution by FKL_before bin ===
    # Bin positions by FKL_before magnitude, plot mean/median of features per bin.
    # Reveals: as FKL grows, which features change how?
    FEAT_BINS = [
        ("teacher_entropy", "Teacher entropy", None),
        ("entropy_S", "Student entropy", None),
        ("student_max_pi", "max π_S", (0, 1)),
        ("beta", "β (T-mass on S-nucleus)", (0, 1)),
        ("student_mass_on_teacher_nuc", "π_S mass on T-nucleus", (0, 1)),
        ("overlap_ratio_k8", "Overlap ratio k8", (0, 1)),
        ("l_rkl", "L_RKL (learnable)", None),
        ("kl2_mu_beta", "kl2(μ,β) — RKL floor", None),
        ("is_single_token_nucleus", "Frozen fraction", (0, 1)),
    ]

    fkl_bins = np.array([0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf])
    bin_labels = ["<0.05", "0.05-0.1", "0.1-0.3", "0.3-0.5", "0.5-1", "1-2", "2-5", ">5"]

    n_feat = len(FEAT_BINS)
    n_c = 3
    n_r = (n_feat + n_c - 1) // n_c
    fig9, axes9 = plt.subplots(n_r, n_c, figsize=(6 * n_c, 4 * n_r), squeeze=False)

    for f_idx, (feat, feat_label, ylim) in enumerate(FEAT_BINS):
        ax = axes9[f_idx // n_c, f_idx % n_c]

        for pair in pairs_with_data:
            sub = all_df[all_df["pair"] == pair].dropna(subset=["fkl_before"])
            sub = sub[sub["fkl_before"] > 1e-6]
            if len(sub) < 10:
                continue
            sub = sub.copy()
            sub["fkl_bin"] = pd.cut(sub["fkl_before"], bins=fkl_bins, labels=bin_labels)
            vals = sub[feat].astype(float) if feat != "is_single_token_nucleus" else sub[feat].astype(float)
            sub["_val"] = vals
            mean_by_bin = sub.groupby("fkl_bin", observed=False)["_val"].mean()
            count_by_bin = sub.groupby("fkl_bin", observed=False)["_val"].count()
            # Suppress bins with <20 samples to avoid noisy tails
            mean_by_bin = mean_by_bin.where(count_by_bin >= 20)

            short = pair.replace("___", "→").replace("_", "/", 2)[:35]
            ax.plot(range(len(bin_labels)), mean_by_bin.values,
                    marker="o", lw=1.4, alpha=0.75, label=short)

        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=7)
        ax.set_xlabel("FKL_before bin", fontsize=9)
        ax.set_ylabel(f"mean {feat_label}", fontsize=9)
        ax.set_title(feat_label, fontsize=10, fontweight="bold")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        if f_idx == 0:
            ax.legend(fontsize=6, loc="best", ncol=1)

    for idx in range(n_feat, n_r * n_c):
        axes9[idx // n_c, idx % n_c].set_visible(False)

    fig9.suptitle("Feature means vs FKL_before bin (per pair)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out9 = os.path.join(out_dir, "feature_by_fkl_bin.png")
    plt.savefig(out9, dpi=150, bbox_inches="tight")
    plt.close(fig9)
    print(f"Saved {out9}")

    # === Summary statistics ===
    print(f"\n{'='*60}")
    print("SUMMARY per pair")
    print(f"{'='*60}")
    for pair in pairs_with_data:
        sub = all_df[all_df["pair"] == pair]
        print(f"\n  {pair}")
        print(f"    N={len(sub)}, frozen (max π > 0.95): {sub.is_single_token_nucleus.mean():.1%}")
        print(f"    μ:       median={sub.mu.median():.4f}, mean={sub.mu.mean():.4f}")
        print(f"    β:       median={sub.beta.median():.4f}, mean={sub.beta.mean():.4f}")
        print(f"    kl2(μ,β): median={sub.kl2_mu_beta.median():.4f}, mean={sub.kl2_mu_beta.mean():.4f}")
        print(f"    L_RKL:   median={sub.l_rkl.median():.4f}, mean={sub.l_rkl.mean():.4f}")
        print(f"    J_A:     median={sub.j_a.median():.4f}, mean={sub.j_a.mean():.4f}")
        print(f"    ε:       median={sub.epsilon.median():.4f}, mean={sub.epsilon.mean():.4f}")

        # Sanity check: L_RKL + kl2 ≈ full RKL?
        non_frozen = sub[~sub.is_single_token_nucleus]
        if len(non_frozen) > 10:
            approx_rkl = non_frozen["l_rkl"] + non_frozen["kl2_mu_beta"]
            print(f"    [non-frozen] L_RKL+kl2 ≈ RKL: median={approx_rkl.median():.4f}")
            if "fkl_before" in non_frozen.columns:
                corr = non_frozen[["l_rkl", "delta_fkl"]].dropna().corr().iloc[0, 1]
                print(f"    [non-frozen] corr(L_RKL, ΔFKL) = {corr:.3f}")


if __name__ == "__main__":
    main()
