"""Single-panel figure: track student's INITIAL top-5 tokens through
REINFORCE-RKL under full mismatch.

Full mismatch: student's initial top-K and teacher's top-K are disjoint.
Student's initial top-5 have non-trivial probability (moderately sharp init,
not extreme peak), so we can see how each mode-token evolves.
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
TEACHER_ALPHA = 2.0
STUDENT_ALPHA = 1.2  # softer than before → initial top-5 all have non-trivial mass

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "mismatch_tracking")
os.makedirs(OUT_DIR, exist_ok=True)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def make_zipf_logits(alpha, top_ids_ordered):
    z = -alpha * np.log(np.arange(1, V + 1).astype(float))
    perm = np.empty(V, dtype=int)
    used = np.array(top_ids_ordered, dtype=int)
    remaining = np.setdiff1d(np.arange(V), used, assume_unique=False)
    perm[: len(used)] = used
    perm[len(used):] = remaining
    z_perm = np.empty_like(z)
    z_perm[perm] = z
    return z_perm


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
    # Teacher supports [0..15]; student supports [50..65] — fully disjoint
    teacher_topk = list(range(16))
    student_topk = list(range(50, 66))
    z_T = make_zipf_logits(TEACHER_ALPHA, teacher_topk)
    z_S = make_zipf_logits(STUDENT_ALPHA, student_topk)
    traj, P = run(z_T, z_S, seed=SEED)

    initial_top5 = np.argsort(-traj[0])[:5]
    steps = np.arange(N_STEPS + 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, 5))
    for r, tok in enumerate(initial_top5):
        ax.plot(steps, traj[:, tok], color=colors[r], lw=2.2,
                label=f"student rank-{r+1}: tok {tok}  "
                      f"π_S₀={traj[0, tok]:.3f}  π_T={P[tok]:.4f}")

    ax.set_xlabel("step", fontsize=11)
    ax.set_ylabel("π_S(tok)", fontsize=11)
    ax.set_title(f"Student INITIAL top-5 under full mismatch\n"
                 f"(student top-16 = {student_topk[0]}..{student_topk[-1]}, "
                 f"teacher top-16 = {teacher_topk[0]}..{teacher_topk[-1]}, LR={LR}, {N_STEPS} steps)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    out = os.path.join(OUT_DIR, "student_top5_single.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    print(f"\nInitial top-5 tokens: {initial_top5.tolist()}")
    print(f"  π_S₀: {traj[0, initial_top5].round(3).tolist()}")
    print(f"  π_S final: {traj[-1, initial_top5].round(3).tolist()}")
    print(f"  π_T: {P[initial_top5].round(4).tolist()}")


if __name__ == "__main__":
    main()
