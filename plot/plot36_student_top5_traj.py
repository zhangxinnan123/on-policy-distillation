"""Step vs π_S(tok): track student's top-5 tokens under FULL mismatch.

Full mismatch = student's top-K and teacher's top-K are completely disjoint
(no shared token in top-8 of either).

Track:
  - The 5 tokens with highest final π_S ("winners" — see who ends up on top)
  - The 5 tokens with highest initial π_S ("initial modes" — see how s0 leaks)
  - Optionally overlay teacher top-5 as reference

Sample u ~ π_S(·) each step, apply REINFORCE-RKL update.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = 100
N_STEPS = 500
LR = 0.05
SEED = 1
TEACHER_ALPHA = 2.5
STUDENT_ALPHA = 2.5

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "mismatch_tracking")
os.makedirs(OUT_DIR, exist_ok=True)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def make_zipf_logits(alpha, top_ids_ordered):
    """Zipf logits where top_ids_ordered[i] is placed at rank i."""
    z = -alpha * np.log(np.arange(1, V + 1).astype(float))
    perm = np.empty(V, dtype=int)
    used = np.array(top_ids_ordered, dtype=int)
    remaining = np.setdiff1d(np.arange(V), used, assume_unique=False)
    perm[: len(used)] = used
    perm[len(used):] = remaining
    z_perm = np.empty_like(z)
    z_perm[perm] = z
    return z_perm


def make_fully_disjoint_pair():
    """Teacher's top-16 and student's top-16 are completely disjoint."""
    # Teacher supports tokens 0..15 (rank order)
    teacher_topk = np.arange(16)
    # Student supports tokens 50..65 — no overlap with teacher's support at all
    student_topk = np.arange(50, 66)
    z_T = make_zipf_logits(TEACHER_ALPHA, teacher_topk.tolist())
    z_S = make_zipf_logits(STUDENT_ALPHA, student_topk.tolist())
    return z_T, z_S, teacher_topk, student_topk


def run(z_T_init, z_S_init, seed):
    rng = np.random.default_rng(seed)
    z_S = z_S_init.copy()
    P = softmax(z_T_init)
    pi = softmax(z_S)
    traj = np.zeros((N_STEPS + 1, V))
    traj[0] = pi
    for t in range(1, N_STEPS + 1):
        u = int(rng.choice(V, p=pi))
        A_u = float(np.log(P[u] + 1e-30) - np.log(pi[u] + 1e-30))
        z_S = z_S + LR * A_u * (np.eye(V)[u] - pi)
        pi = softmax(z_S)
        traj[t] = pi
    return traj, P


def main():
    z_T, z_S, teacher_topk, student_topk = make_fully_disjoint_pair()
    traj, P = run(z_T, z_S, seed=SEED)
    steps = np.arange(N_STEPS + 1)

    # Student's top-5 at step 0 (initial modes)
    initial_top5 = np.argsort(-traj[0])[:5]
    # Student's top-5 at final step (winners)
    final_top5 = np.argsort(-traj[-1])[:5]
    # Teacher's top-5 (reference)
    teacher_top5 = np.argsort(-P)[:5]

    # ==============================================================
    # Figure: 3 panels side-by-side
    # ==============================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: student's FINAL top-5 trajectories ---
    ax = axes[0]
    colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, 5))
    for r, tok in enumerate(final_top5):
        pi_T_tok = P[tok]
        in_T = tok in teacher_topk
        marker = "★" if in_T else ""
        ax.plot(steps, traj[:, tok], color=colors[r], lw=2.0,
                label=f"rank-{r+1}: tok {tok}  π_T={pi_T_tok:.3f} {marker}")
    ax.set_xlabel("step", fontsize=10)
    ax.set_ylabel("π_S(tok)", fontsize=10)
    ax.set_title("Student's FINAL top-5 tokens\n(★ = also in teacher top-16)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 2: student's INITIAL top-5 trajectories ---
    ax = axes[1]
    colors2 = plt.get_cmap("Reds")(np.linspace(0.4, 0.9, 5))
    for r, tok in enumerate(initial_top5):
        pi_T_tok = P[tok]
        ax.plot(steps, traj[:, tok], color=colors2[r], lw=2.0,
                label=f"rank-{r+1}: tok {tok}  π_T={pi_T_tok:.3f}  π_S₀={traj[0, tok]:.3f}")
    ax.set_xlabel("step", fontsize=10)
    ax.set_ylabel("π_S(tok)", fontsize=10)
    ax.set_title("Student's INITIAL top-5 tokens\n(all disjoint from teacher's top-16)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 3: teacher's top-5 tokens (their π_S trajectory) ---
    ax = axes[2]
    colors3 = plt.get_cmap("Blues")(np.linspace(0.4, 0.9, 5))
    for r, tok in enumerate(teacher_top5):
        pi_T_tok = P[tok]
        ax.plot(steps, traj[:, tok], color=colors3[r], lw=2.0,
                label=f"T rank-{r+1}: tok {tok}  π_T={pi_T_tok:.3f}")
    ax.set_xlabel("step", fontsize=10)
    ax.set_ylabel("π_S(tok)", fontsize=10)
    ax.set_title("Teacher's top-5 tokens\n(does student ever learn them?)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(f"Full mismatch: student top-K ∩ teacher top-K = ∅  "
                 f"(V={V}, LR={LR}, {N_STEPS} steps, u ~ π_S)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "student_top5_full_mismatch.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # Summary print
    print(f"\nTeacher top-5:  {teacher_top5.tolist()}  π_T={P[teacher_top5].round(3).tolist()}")
    print(f"Initial top-5:  {initial_top5.tolist()}")
    print(f"Final top-5:    {final_top5.tolist()}  π_S={traj[-1, final_top5].round(3).tolist()}")
    print(f"Final top-5 ∩ teacher top-16: "
          f"{[t for t in final_top5 if t in teacher_topk]}")


if __name__ == "__main__":
    main()
