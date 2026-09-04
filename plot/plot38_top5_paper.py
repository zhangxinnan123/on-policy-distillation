"""Publication-quality single-panel figure:
Student's initial top-5 token probabilities under REINFORCE-RKL with a fully
mismatched teacher (disjoint top-K supports, random token indices).

Reproduces the "modes collapse but student cannot find teacher's support"
phenomenon in a clean, paper-ready form.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Publication style ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
})

# ---------------- Config ----------------
V = 100
N_STEPS = 500
LR = 0.01
TOP_K_SUPPORT = 16
SEED_INIT = 0        # seed for teacher/student support selection
SEED_RUN = 1         # seed for sampling trajectory
TEACHER_ALPHA = 2.0
STUDENT_ALPHA = 1.2

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "mismatch_tracking")
os.makedirs(OUT_DIR, exist_ok=True)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def zipf_over_support(alpha, support_tokens):
    """Zipf logits over the given support_tokens (in rank order); rest very low."""
    z = np.full(V, -50.0)
    for r, tok in enumerate(support_tokens):
        z[tok] = -alpha * np.log(r + 1)
    return z


def make_disjoint_supports(rng):
    """Randomly pick two disjoint size-K token subsets."""
    all_ids = np.arange(V)
    rng.shuffle(all_ids)
    teacher_supp = all_ids[:TOP_K_SUPPORT]
    student_supp = all_ids[TOP_K_SUPPORT: 2 * TOP_K_SUPPORT]
    # Shuffle within each to break the sorted order (visual clarity)
    teacher_supp = rng.permutation(teacher_supp)
    student_supp = rng.permutation(student_supp)
    return teacher_supp, student_supp


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
    init_rng = np.random.default_rng(SEED_INIT)
    teacher_supp, student_supp = make_disjoint_supports(init_rng)
    z_T = zipf_over_support(TEACHER_ALPHA, teacher_supp)
    z_S = zipf_over_support(STUDENT_ALPHA, student_supp)

    traj, P = run(z_T, z_S, seed=SEED_RUN)
    steps = np.arange(N_STEPS + 1)
    initial_top5 = np.argsort(-traj[0])[:5]

    # ---------------- Figure ----------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Muted, colorblind-friendly palette
    palette = ["#1F4E79", "#2E75B6", "#548235", "#BF9000", "#C00000"]

    for r, tok in enumerate(initial_top5):
        ax.plot(
            steps,
            traj[:, tok],
            color=palette[r],
            lw=1.9,
            alpha=0.95,
            label=(
                rf"rank {r + 1}: token {tok}   "
                rf"$\pi_S^{{(0)}}={traj[0, tok]:.2f}$"
            ),
        )

    ax.set_xlabel("Update step $t$")
    ax.set_ylabel(r"Student probability $\pi_S(v)$")
    ax.set_title(
        r"Evolution of student's initial top-5 tokens under REINFORCE-RKL "
        r"(disjoint supports)",
        pad=10,
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#333333",
        handlelength=1.2,
        fontsize=7,
        borderpad=0.3,
        labelspacing=0.25,
        handletextpad=0.4,
    )
    leg.get_frame().set_linewidth(0.5)

    # Small annotation box: setup summary
    info = (
        rf"$V={V}$, $K={TOP_K_SUPPORT}$, $\eta={LR}$"
        "\n"
        r"$\mathrm{supp}(\pi_T) \cap \mathrm{supp}(\pi_S) = \varnothing$"
    )
    ax.text(
        0.98, 0.63, info,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#888888", linewidth=0.5),
    )

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "top5_paper.png")
    out_pdf = os.path.join(OUT_DIR, "top5_paper.pdf")
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")

    # ---------- console diagnostic ----------
    print(f"\nTeacher support: {sorted(teacher_supp.tolist())}")
    print(f"Student support: {sorted(student_supp.tolist())}")
    print(f"Initial top-5 tokens: {initial_top5.tolist()}")
    print(f"  pi_S^0: {traj[0, initial_top5].round(3).tolist()}")
    print(f"  pi_S^T: {traj[-1, initial_top5].round(4).tolist()}")


if __name__ == "__main__":
    main()
