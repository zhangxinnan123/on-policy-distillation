"""Toy simulation: track student mode token evolution under mismatch.

Setup:
  - vocab size V=100
  - Teacher has mode at token t*, sharpness controlled by alpha_T
  - Student initially has mode at s0 ≠ t*, overconfidence controlled by alpha_S
  - Optionally: s0 is completely outside teacher's top-K (full disjoint)

Update: single-sample REINFORCE RKL
    u ~ pi_S(·)
    A_u = log pi_T(u) - log pi_S(u)
    pi_S ← softmax(z + LR · A_u · (e_u - pi_S))

Tracking per step:
    - argmax(pi_S) (student's current mode)
    - pi_S(t*)     (mass on teacher's mode)
    - pi_S(s0)     (mass on initial wrong mode)
    - entropy of pi_S
    - FKL(pi_T || pi_S), RKL(pi_S || pi_T)

Scenarios (rows in figure):
  A: sharp mismatch     — both sharp, disjoint modes (worst case)
  B: soft mismatch      — student sharp, teacher flat
  C: partial overlap    — student mode is teacher's rank-2
  D: mild mismatch      — student flat, teacher sharp
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = 100
N_STEPS = 500
LR = 0.05
N_SEEDS = 32   # average trajectories over seeds

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "mismatch_tracking")
os.makedirs(OUT_DIR, exist_ok=True)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def make_zipf_logits(alpha, top_id, rng):
    """Zipf-like logits with peak at top_id."""
    z = -alpha * np.log(np.arange(1, V + 1).astype(float))
    # place peak at top_id: identity perm, then swap so ranked-0 sits at top_id
    perm = rng.permutation(V)
    # ensure top_id gets rank 0
    perm[np.where(perm == top_id)[0][0]], perm[0] = perm[0], perm[np.where(perm == top_id)[0][0]]
    z_perm = np.empty_like(z)
    z_perm[perm] = z
    return z_perm


def make_pair(scenario, rng):
    """Return (z_T, z_S, teacher_mode, student_init_mode)."""
    if scenario == "sharp_mismatch":
        # Both sharp; disjoint modes far apart
        t_star, s0 = 3, 71
        z_T = make_zipf_logits(alpha=2.5, top_id=t_star, rng=rng)
        z_S = make_zipf_logits(alpha=2.5, top_id=s0, rng=rng)
    elif scenario == "sharp_S_flat_T":
        # Student overconfident on wrong token, teacher spread out
        t_star, s0 = 3, 71
        z_T = make_zipf_logits(alpha=0.8, top_id=t_star, rng=rng)
        z_S = make_zipf_logits(alpha=2.5, top_id=s0, rng=rng)
    elif scenario == "partial_overlap":
        # Student mode is teacher's rank-2 (there is a bridge)
        t_star, s0 = 3, 5
        z_T = make_zipf_logits(alpha=2.0, top_id=t_star, rng=rng)
        z_S = make_zipf_logits(alpha=2.0, top_id=s0, rng=rng)
        # Boost pi_T at s0 so it's rank-2 (still much less than t*)
        z_T[s0] = z_T[t_star] - 1.0
    elif scenario == "flat_S_sharp_T":
        # Student is fairly diffuse, teacher sharp
        t_star, s0 = 3, 71
        z_T = make_zipf_logits(alpha=2.5, top_id=t_star, rng=rng)
        z_S = make_zipf_logits(alpha=0.5, top_id=s0, rng=rng)
    else:
        raise ValueError(scenario)
    return z_T, z_S, t_star, s0


def run_single(z_T_init, z_S_init, t_star, s0, seed):
    rng = np.random.default_rng(seed)
    z_T = z_T_init.copy()
    z_S = z_S_init.copy()
    P = softmax(z_T)
    pi = softmax(z_S)

    hist = {
        "step": np.arange(N_STEPS + 1),
        "mode": np.zeros(N_STEPS + 1, dtype=int),
        "pi_t_star": np.zeros(N_STEPS + 1),
        "pi_s0": np.zeros(N_STEPS + 1),
        "entropy_S": np.zeros(N_STEPS + 1),
        "fkl": np.zeros(N_STEPS + 1),
        "rkl": np.zeros(N_STEPS + 1),
        "sampled": np.full(N_STEPS + 1, -1, dtype=int),
        "A_at_sampled": np.zeros(N_STEPS + 1),
    }

    def snap(t):
        hist["mode"][t] = int(np.argmax(pi))
        hist["pi_t_star"][t] = pi[t_star]
        hist["pi_s0"][t] = pi[s0]
        hist["entropy_S"][t] = float(-np.sum(pi * np.log(pi + 1e-30)))
        hist["fkl"][t] = float(np.sum(P * np.log((P + 1e-30) / (pi + 1e-30))))
        hist["rkl"][t] = float(np.sum(pi * np.log((pi + 1e-30) / (P + 1e-30))))

    snap(0)
    for t in range(1, N_STEPS + 1):
        u = int(rng.choice(V, p=pi))
        A_u = float(np.log(P[u] + 1e-30) - np.log(pi[u] + 1e-30))
        z_S = z_S + LR * A_u * (np.eye(V)[u] - pi)
        pi = softmax(z_S)
        hist["sampled"][t] = u
        hist["A_at_sampled"][t] = A_u
        snap(t)
    return hist


def avg_over_seeds(scenario):
    trajs = []
    seed_rng = np.random.default_rng(0)
    z_T, z_S, t_star, s0 = make_pair(scenario, seed_rng)   # fixed pair per scenario
    for s in range(N_SEEDS):
        h = run_single(z_T, z_S, t_star, s0, seed=s + 1)
        trajs.append(h)
    keys = ["pi_t_star", "pi_s0", "entropy_S", "fkl", "rkl"]
    mean = {k: np.mean([h[k] for h in trajs], axis=0) for k in keys}
    std = {k: np.std([h[k] for h in trajs], axis=0) for k in keys}
    # Mode: how often each seed's mode == t_star vs s0 vs "other"
    mode_arr = np.stack([h["mode"] for h in trajs])   # (S, T+1)
    frac_t = (mode_arr == t_star).mean(axis=0)
    frac_s = (mode_arr == s0).mean(axis=0)
    frac_other = 1 - frac_t - frac_s
    return {"mean": mean, "std": std, "frac_t": frac_t, "frac_s": frac_s,
            "frac_other": frac_other, "t_star": t_star, "s0": s0}


SCENARIOS = [
    ("sharp_mismatch",  "Sharp mismatch\n(both peaky, disjoint modes)"),
    ("sharp_S_flat_T",  "Overconfident student, flat teacher\n(worst case: A<0 huge)"),
    ("partial_overlap", "Partial overlap\n(student's mode is teacher rank-2)"),
    ("flat_S_sharp_T",  "Flat student, sharp teacher\n(easy case)"),
]


def main():
    fig, axes = plt.subplots(len(SCENARIOS), 4, figsize=(20, 4.5 * len(SCENARIOS)))

    for row, (sc, title) in enumerate(SCENARIOS):
        R = avg_over_seeds(sc)
        steps = np.arange(N_STEPS + 1)
        m = R["mean"]; s = R["std"]

        # --- Panel 1: probability mass on teacher-mode t* and student-init-mode s0 ---
        ax = axes[row, 0]
        ax.plot(steps, m["pi_t_star"], color="#2563EB", lw=1.8,
                label=f"π_S(t*={R['t_star']})  teacher mode")
        ax.fill_between(steps, m["pi_t_star"] - s["pi_t_star"],
                        m["pi_t_star"] + s["pi_t_star"], color="#2563EB", alpha=0.2)
        ax.plot(steps, m["pi_s0"], color="#DC2626", lw=1.8,
                label=f"π_S(s0={R['s0']})  initial wrong mode")
        ax.fill_between(steps, m["pi_s0"] - s["pi_s0"],
                        m["pi_s0"] + s["pi_s0"], color="#DC2626", alpha=0.2)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("probability mass", fontsize=9)
        ax.set_title(f"Mass on t* vs s0\n(mean ± std over {N_SEEDS} seeds)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)

        # --- Panel 2: fraction of seeds where argmax == t* / s0 / other ---
        ax = axes[row, 1]
        ax.stackplot(steps, R["frac_s"], R["frac_other"], R["frac_t"],
                     colors=["#DC2626", "#9CA3AF", "#2563EB"],
                     labels=["mode = s0 (stuck)", "mode = other",
                             "mode = t* (learned)"], alpha=0.85)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("fraction of seeds", fontsize=9)
        ax.set_title("Where is student's argmax?", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="center right")
        ax.set_ylim(0, 1)

        # --- Panel 3: FKL, RKL, entropy over time ---
        ax = axes[row, 2]
        ax2 = ax.twinx()
        ax.plot(steps, m["fkl"], color="#DC2626", lw=1.5, label="FKL (T||S)")
        ax.plot(steps, m["rkl"], color="#2563EB", lw=1.5, label="RKL (S||T)")
        ax2.plot(steps, m["entropy_S"], color="#059669", lw=1.5, ls="--",
                 label="entropy_S")
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("KL (nats)", fontsize=9)
        ax2.set_ylabel("entropy_S", fontsize=9, color="#059669")
        ax.set_title("Divergences & entropy", fontsize=9, fontweight="bold")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

        # --- Panel 4: mode-switch event visualization (single seed 0) ---
        ax = axes[row, 3]
        seed_rng = np.random.default_rng(0)
        z_T, z_S, t_star, s0 = make_pair(sc, seed_rng)
        h = run_single(z_T, z_S, t_star, s0, seed=1)
        ax.scatter(np.arange(1, N_STEPS + 1), h["sampled"][1:],
                   c=h["A_at_sampled"][1:], cmap="RdBu", s=6, alpha=0.6,
                   vmin=-4, vmax=4)
        ax.axhline(t_star, color="#2563EB", ls="--", lw=1.2, label=f"t*={t_star}")
        ax.axhline(s0, color="#DC2626", ls="--", lw=1.2, label=f"s0={s0}")
        ax.plot(np.arange(N_STEPS + 1), h["mode"], color="black", lw=1.0,
                alpha=0.7, label="argmax(pi_S)")
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("token id", fontsize=9)
        ax.set_title("Sampled token & mode\n(color = A_u; single seed)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(-2, V + 2)

        # Row label on the left
        axes[row, 0].annotate(title, xy=(-0.35, 0.5), xycoords="axes fraction",
                              ha="right", va="center", fontsize=11,
                              fontweight="bold", rotation=0)

    fig.suptitle(f"Student mode-token tracking under mismatch (V={V}, LR={LR}, {N_STEPS} steps)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0.04, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "mismatch_mode_tracking.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
