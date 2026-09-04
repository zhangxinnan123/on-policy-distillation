"""Test whether student's mode DISAPPEARS under sharp mismatch.

Hypothesis: with sharp teacher + sharp student + disjoint modes, s0 collapses
uniformly, so all other tokens should stay roughly equal → mode disappears.

But in practice: teacher's Zipf tail is nonzero, so sampling noise + positive
feedback creates a random attractor. To verify:

  Panel A: Sharp mismatch with Zipf teacher, 24 seeds — show winner distribution
  Panel B: Sharp mismatch with TRULY DELTA teacher (mass only on t*, rest ε)
           — does student's mode actually disappear (entropy → log V)?
  Panel C: Entropy trajectory: Zipf teacher vs delta teacher, over 24 seeds
  Panel D: Final π_S(top-5 tokens) distribution across seeds
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = 100
N_STEPS = 500
LR = 0.05
N_SEEDS = 24

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "mismatch_tracking")
os.makedirs(OUT_DIR, exist_ok=True)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def make_zipf_logits(alpha, top_id, rng):
    z = -alpha * np.log(np.arange(1, V + 1).astype(float))
    perm = rng.permutation(V)
    perm[np.where(perm == top_id)[0][0]], perm[0] = perm[0], perm[np.where(perm == top_id)[0][0]]
    z_perm = np.empty_like(z)
    z_perm[perm] = z
    return z_perm


def make_delta_teacher(t_star, mass=0.999):
    """Truly-delta teacher: mass on t*, ε uniform on rest."""
    p = np.full(V, (1 - mass) / (V - 1))
    p[t_star] = mass
    return np.log(p)


def run_single(z_T_init, z_S_init, seed):
    rng = np.random.default_rng(seed)
    z_S = z_S_init.copy()
    P = softmax(z_T_init)
    pi = softmax(z_S)
    ent_hist = np.zeros(N_STEPS + 1)
    ent_hist[0] = -np.sum(pi * np.log(pi + 1e-30))
    for t in range(1, N_STEPS + 1):
        u = int(rng.choice(V, p=pi))
        A_u = float(np.log(P[u] + 1e-30) - np.log(pi[u] + 1e-30))
        z_S = z_S + LR * A_u * (np.eye(V)[u] - pi)
        pi = softmax(z_S)
        ent_hist[t] = -np.sum(pi * np.log(pi + 1e-30))
    return pi, ent_hist


def main():
    t_star, s0 = 3, 71
    setup_rng = np.random.default_rng(0)

    # Case A: Zipf teacher, sharp student
    z_T_zipf = make_zipf_logits(alpha=2.5, top_id=t_star, rng=setup_rng)
    z_S_zipf = make_zipf_logits(alpha=2.5, top_id=s0, rng=setup_rng)
    # Case B: Delta teacher
    z_T_delta = make_delta_teacher(t_star, mass=0.999)
    z_S_delta = z_S_zipf.copy()

    P_zipf = softmax(z_T_zipf)
    P_delta = softmax(z_T_delta)
    teacher_topk_zipf = np.argsort(-P_zipf)[:8]
    print(f"Zipf teacher top-8 tokens (ranked): {teacher_topk_zipf.tolist()}")
    print(f"  their π_T: {P_zipf[teacher_topk_zipf].round(3).tolist()}")

    # Run many seeds
    final_pi_zipf, ent_traj_zipf = [], []
    final_pi_delta, ent_traj_delta = [], []
    for s in range(N_SEEDS):
        pi_z, ent_z = run_single(z_T_zipf, z_S_zipf, seed=s + 1)
        final_pi_zipf.append(pi_z); ent_traj_zipf.append(ent_z)
        pi_d, ent_d = run_single(z_T_delta, z_S_delta, seed=s + 1)
        final_pi_delta.append(pi_d); ent_traj_delta.append(ent_d)
    final_pi_zipf = np.stack(final_pi_zipf)     # (N_SEEDS, V)
    final_pi_delta = np.stack(final_pi_delta)
    ent_traj_zipf = np.stack(ent_traj_zipf)     # (N_SEEDS, T+1)
    ent_traj_delta = np.stack(ent_traj_delta)

    winner_zipf = final_pi_zipf.argmax(axis=1)   # (N_SEEDS,)
    winner_delta = final_pi_delta.argmax(axis=1)
    max_prob_zipf = final_pi_zipf.max(axis=1)
    max_prob_delta = final_pi_delta.max(axis=1)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    # === Panel A: winner distribution (Zipf teacher) ===
    ax = fig.add_subplot(gs[0, 0])
    win_ids, win_cnt = np.unique(winner_zipf, return_counts=True)
    colors = ["#2563EB" if w in teacher_topk_zipf else
              ("#DC2626" if w == s0 else "#9CA3AF") for w in win_ids]
    labels = [f"tok {w}\nπ_T={P_zipf[w]:.3f}" for w in win_ids]
    ax.bar(range(len(win_ids)), win_cnt, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(win_ids)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("# seeds", fontsize=9)
    ax.set_title(f"A: Zipf teacher — winning attractor across {N_SEEDS} seeds\n"
                 "(blue = teacher top-8, red = s0, gray = other)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    # === Panel B: winner distribution (Delta teacher) ===
    ax = fig.add_subplot(gs[0, 1])
    win_ids_d, win_cnt_d = np.unique(winner_delta, return_counts=True)
    colors_d = ["#2563EB" if w == t_star else
                ("#DC2626" if w == s0 else "#9CA3AF") for w in win_ids_d]
    labels_d = [f"tok {w}" + (" =t*" if w == t_star else "")
                for w in win_ids_d]
    ax.bar(range(len(win_ids_d)), win_cnt_d, color=colors_d, alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(win_ids_d)))
    ax.set_xticklabels(labels_d, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("# seeds", fontsize=9)
    ax.set_title(f"B: DELTA teacher (mass 0.999 on t*) — winning attractor\n"
                 "(blue = t*, red = s0, gray = other)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    # === Panel C: entropy trajectory ===
    ax = fig.add_subplot(gs[0, 2])
    steps = np.arange(N_STEPS + 1)
    ent_z_mean = ent_traj_zipf.mean(axis=0); ent_z_std = ent_traj_zipf.std(axis=0)
    ent_d_mean = ent_traj_delta.mean(axis=0); ent_d_std = ent_traj_delta.std(axis=0)
    ax.plot(steps, ent_z_mean, color="#2563EB", lw=1.8, label="Zipf teacher")
    ax.fill_between(steps, ent_z_mean - ent_z_std, ent_z_mean + ent_z_std,
                    color="#2563EB", alpha=0.2)
    ax.plot(steps, ent_d_mean, color="#DC2626", lw=1.8, label="Delta teacher")
    ax.fill_between(steps, ent_d_mean - ent_d_std, ent_d_mean + ent_d_std,
                    color="#DC2626", alpha=0.2)
    ax.axhline(np.log(V), color="black", ls="--", lw=1, label=f"log V = {np.log(V):.2f} (uniform)")
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("entropy(π_S)", fontsize=9)
    ax.set_title(f"C: entropy trajectory (mean ± std over {N_SEEDS} seeds)\n"
                 "Delta teacher → higher entropy = 'mode disappears'?",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # === Panel D: final max π_S distribution (does mode really vanish?) ===
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(max_prob_zipf, bins=20, alpha=0.6, color="#2563EB",
            label=f"Zipf teacher (mean={max_prob_zipf.mean():.2f})", edgecolor="black")
    ax.hist(max_prob_delta, bins=20, alpha=0.6, color="#DC2626",
            label=f"Delta teacher (mean={max_prob_delta.mean():.2f})", edgecolor="black")
    ax.axvline(1 / V, color="black", ls="--", lw=1, label=f"uniform = 1/V = {1/V:.2f}")
    ax.set_xlabel("final max π_S", fontsize=9)
    ax.set_ylabel("# seeds", fontsize=9)
    ax.set_title("D: final max π_S — is there still a mode?",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # === Panel E: top-5 π_S values across seeds ===
    ax = fig.add_subplot(gs[1, 1])
    top5_zipf = -np.sort(-final_pi_zipf, axis=1)[:, :5]    # (N_SEEDS, 5)
    top5_delta = -np.sort(-final_pi_delta, axis=1)[:, :5]
    ranks = np.arange(1, 6)
    ax.errorbar(ranks - 0.1, top5_zipf.mean(axis=0), yerr=top5_zipf.std(axis=0),
                fmt="o-", color="#2563EB", capsize=4, label="Zipf teacher", lw=1.8, ms=8)
    ax.errorbar(ranks + 0.1, top5_delta.mean(axis=0), yerr=top5_delta.std(axis=0),
                fmt="s-", color="#DC2626", capsize=4, label="Delta teacher", lw=1.8, ms=8)
    ax.axhline(1 / V, color="black", ls="--", lw=1, label="uniform")
    ax.set_xlabel("final rank", fontsize=9)
    ax.set_ylabel("π_S(rank r)", fontsize=9)
    ax.set_title(f"E: final top-5 π_S values (mean ± std over {N_SEEDS} seeds)\n"
                 "Zipf: sharp attractor. Delta: flatter, mode 'weakens'.",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_yscale("log")

    # === Panel F: winning-rate for each token id (Zipf case) ===
    ax = fig.add_subplot(gs[1, 2])
    # For each token, average final π_S across seeds
    avg_pi_zipf = final_pi_zipf.mean(axis=0)
    avg_pi_delta = final_pi_delta.mean(axis=0)
    # Show sorted by teacher rank
    sort_by_P_zipf = np.argsort(-P_zipf)
    ranks_zipf = np.arange(V)
    ax.plot(ranks_zipf, avg_pi_zipf[sort_by_P_zipf], "o", color="#2563EB",
            ms=3, alpha=0.6, label="Zipf: avg π_S")
    ax.plot(ranks_zipf, avg_pi_delta[sort_by_P_zipf], "s", color="#DC2626",
            ms=3, alpha=0.6, label="Delta: avg π_S")
    ax.plot(ranks_zipf, P_zipf[sort_by_P_zipf], "-", color="black", lw=1,
            alpha=0.6, label="Zipf π_T (reference)")
    ax.set_xlabel("token (sorted by Zipf π_T desc)", fontsize=9)
    ax.set_ylabel("π (log)", fontsize=9)
    ax.set_yscale("log")
    ax.set_title(f"F: avg final π_S over {N_SEEDS} seeds vs teacher rank",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(1e-4, 1)

    fig.suptitle("Does student's mode DISAPPEAR under sharp mismatch?\n"
                 f"(Zipf α=2.5 vs Delta teacher; sharp student α=2.5; LR={LR}, {N_STEPS} steps, u ~ π_S)",
                 fontsize=13, fontweight="bold")
    out = os.path.join(OUT_DIR, "mode_disappear_test.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
