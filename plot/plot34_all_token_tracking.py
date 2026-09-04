"""Track ALL V tokens' probability evolution under REINFORCE-RKL.

Sample u ~ pi_S(·) each step, then update pi_S. Plot every token's
π_S(tok) trajectory. Highlight teacher's top-K and s0 with color, everything
else in faint gray.

This exposes:
  - Are all tokens "swimming" or does the system have a clear attractor?
  - Do rare tokens ever spike?
  - How fast does mass leak away from s0 vs. accumulate on other tokens?
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = 100
N_STEPS = 500
LR = 0.05
TOP_K_TEACHER = 5

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


def make_pair(scenario, rng):
    if scenario == "sharp_mismatch":
        t_star, s0 = 3, 71
        z_T = make_zipf_logits(alpha=2.5, top_id=t_star, rng=rng)
        z_S = make_zipf_logits(alpha=2.5, top_id=s0, rng=rng)
    elif scenario == "sharp_S_flat_T":
        t_star, s0 = 3, 71
        z_T = make_zipf_logits(alpha=0.8, top_id=t_star, rng=rng)
        z_S = make_zipf_logits(alpha=2.5, top_id=s0, rng=rng)
    elif scenario == "partial_overlap":
        t_star, s0 = 3, 5
        z_T = make_zipf_logits(alpha=2.0, top_id=t_star, rng=rng)
        z_S = make_zipf_logits(alpha=2.0, top_id=s0, rng=rng)
        z_T[s0] = z_T[t_star] - 1.0
    elif scenario == "flat_S_sharp_T":
        t_star, s0 = 3, 71
        z_T = make_zipf_logits(alpha=2.5, top_id=t_star, rng=rng)
        z_S = make_zipf_logits(alpha=0.5, top_id=s0, rng=rng)
    else:
        raise ValueError(scenario)
    return z_T, z_S, t_star, s0


def run_single(z_T_init, z_S_init, seed):
    rng = np.random.default_rng(seed)
    z_S = z_S_init.copy()
    P = softmax(z_T_init)
    pi = softmax(z_S)
    traj = np.zeros((N_STEPS + 1, V))
    traj[0] = pi
    sampled = np.full(N_STEPS + 1, -1, dtype=int)
    for t in range(1, N_STEPS + 1):
        # Random sample per current pi_S
        u = int(rng.choice(V, p=pi))
        A_u = float(np.log(P[u] + 1e-30) - np.log(pi[u] + 1e-30))
        z_S = z_S + LR * A_u * (np.eye(V)[u] - pi)
        pi = softmax(z_S)
        traj[t] = pi
        sampled[t] = u
    return traj, sampled, P


SCENARIOS = [
    ("sharp_mismatch",  "Sharp mismatch"),
    ("sharp_S_flat_T",  "Overconfident S, flat T"),
    ("partial_overlap", "Partial overlap"),
    ("flat_S_sharp_T",  "Flat S, sharp T"),
]


def main():
    fig, axes = plt.subplots(len(SCENARIOS), 3, figsize=(20, 4.5 * len(SCENARIOS)))

    for row, (sc, title) in enumerate(SCENARIOS):
        seed_rng = np.random.default_rng(0)
        z_T, z_S, t_star, s0 = make_pair(sc, seed_rng)
        traj, sampled, P = run_single(z_T, z_S, seed=1)
        steps = np.arange(N_STEPS + 1)
        teacher_topk = np.argsort(-P)[:TOP_K_TEACHER]

        # ============ Panel 1: linear scale — all V tokens ============
        ax = axes[row, 0]
        # Background: all tokens in faint gray
        for tok in range(V):
            if tok in teacher_topk or tok == s0:
                continue
            ax.plot(steps, traj[:, tok], color="#D1D5DB", lw=0.4, alpha=0.4)
        # Foreground: teacher top-K (blue gradient by rank) + s0 (red)
        cmap_T = plt.get_cmap("Blues")
        for rank_i, tok in enumerate(teacher_topk):
            color = cmap_T(1 - 0.13 * rank_i)
            ax.plot(steps, traj[:, tok], color=color, lw=1.8,
                    label=f"T rank-{rank_i+1} (tok {tok}, π_T={P[tok]:.3f})")
        ax.plot(steps, traj[:, s0], color="#DC2626", lw=1.8,
                label=f"s0={s0} (π_T={P[s0]:.3f})")
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("π_S(tok)", fontsize=9)
        ax.set_title(f"All {V} tokens (linear)\ngray=other, blue=T top-{TOP_K_TEACHER}, red=s0",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.85)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(bottom=0)

        # ============ Panel 2: log-scale — see tail dynamics ============
        ax = axes[row, 1]
        for tok in range(V):
            if tok in teacher_topk or tok == s0:
                continue
            ax.plot(steps, traj[:, tok], color="#D1D5DB", lw=0.4, alpha=0.4)
        for rank_i, tok in enumerate(teacher_topk):
            color = cmap_T(1 - 0.13 * rank_i)
            ax.plot(steps, traj[:, tok], color=color, lw=1.8,
                    label=f"T rank-{rank_i+1}")
        ax.plot(steps, traj[:, s0], color="#DC2626", lw=1.8, label=f"s0={s0}")
        ax.set_yscale("log")
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("π_S(tok) [log]", fontsize=9)
        ax.set_title("All V tokens (log scale)\n(tail dynamics visible)",
                     fontsize=9, fontweight="bold")
        ax.set_ylim(1e-6, 1.5)
        ax.grid(True, alpha=0.2, which="both")
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)

        # ============ Panel 3: heatmap (V × T) ============
        ax = axes[row, 2]
        # Sort tokens by teacher prob desc for readable rows
        sort_by_P = np.argsort(-P)
        traj_sorted = traj[:, sort_by_P].T   # (V, T+1)
        im = ax.imshow(traj_sorted, aspect="auto", origin="upper",
                       cmap="viridis", vmin=0, vmax=min(0.5, traj.max()),
                       extent=[0, N_STEPS, V, 0])
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="π_S(tok)")
        # Mark t_star and s0 positions
        rank_of_t = int(np.where(sort_by_P == t_star)[0][0])
        rank_of_s = int(np.where(sort_by_P == s0)[0][0])
        ax.axhline(rank_of_t, color="cyan", ls="--", lw=1,
                   label=f"t* (T rank {rank_of_t+1})")
        ax.axhline(rank_of_s, color="red", ls="--", lw=1,
                   label=f"s0 (T rank {rank_of_s+1})")
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("token (sorted by π_T desc)", fontsize=9)
        ax.set_title("Heatmap: π_S(tok) over time\n(y = teacher rank; look for bright rows)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

        # Row label
        axes[row, 0].annotate(title, xy=(-0.32, 0.5), xycoords="axes fraction",
                              ha="right", va="center", fontsize=11,
                              fontweight="bold", rotation=0)

    fig.suptitle(f"All-{V}-token tracking under REINFORCE-RKL (LR={LR}, {N_STEPS} steps, sample u ~ π_S)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0.04, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "all_token_tracking.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
