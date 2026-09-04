"""Two-panel paper figure: student initial top-5 evolution under REINFORCE-RKL
with disjoint supports, comparing HIGH vs LOW initial student entropy.

Left panel  : high initial entropy (flat student → mass spread across top-K)
Right panel : low  initial entropy (sharp student → one dominant mode)
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
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 7,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.7,
    "figure.dpi": 150,
})

# ---------------- Config ----------------
V = 100
N_STEPS = 200
LR = 0.005
TOP_K_SUPPORT = 16
SEED_INIT = 0
SEED_RUN = 1
TEACHER_ALPHA = 2.0
STUDENT_ALPHA_HIGH_ENT = 0.4    # small α → flat distribution → high entropy
STUDENT_ALPHA_LOW_ENT  = 3.2    # large α → sharp distribution → low entropy (max π_S ≈ 0.85)
N_TEACHER_TRACKED = 3           # how many teacher-preferred tokens to overlay

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "mismatch_tracking")
os.makedirs(OUT_DIR, exist_ok=True)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def zipf_over_support(alpha, support_tokens):
    z = np.full(V, -50.0)
    for r, tok in enumerate(support_tokens):
        z[tok] = -alpha * np.log(r + 1)
    return z


def make_disjoint_supports(rng):
    all_ids = np.arange(V)
    rng.shuffle(all_ids)
    teacher_supp = all_ids[:TOP_K_SUPPORT]
    student_supp = all_ids[TOP_K_SUPPORT: 2 * TOP_K_SUPPORT]
    return rng.permutation(teacher_supp), rng.permutation(student_supp)


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


def entropy(p):
    return float(-np.sum(p * np.log(p + 1e-30)))


def plot_panel(ax, traj, P, alpha_S, title):
    steps = np.arange(N_STEPS + 1)
    initial_top5 = np.argsort(-traj[0])[:5]
    teacher_top = np.argsort(-P)[:N_TEACHER_TRACKED]
    # Colors: student ranks 1..5 = cool sequential; teacher tokens = warm dashes
    stu_palette = ["#1F4E79", "#2E75B6", "#548235", "#BF9000", "#7F6000"]
    tea_palette = ["#C00000", "#E97132", "#843C0C"]

    # Student initial top-5 (solid)
    for r, tok in enumerate(initial_top5):
        ax.plot(steps, traj[:, tok], color=stu_palette[r], lw=1.6, alpha=0.95,
                label=rf"S-rank {r+1}: tok {tok}   $\pi_S^{{(0)}}={traj[0, tok]:.2f}$")

    # Teacher-preferred tokens (dashed) — tracked in student's distribution
    for r, tok in enumerate(teacher_top):
        ax.plot(steps, traj[:, tok], color=tea_palette[r], lw=1.4, ls="--",
                alpha=0.9,
                label=rf"T-rank {r+1}: tok {tok}   $\pi_T={P[tok]:.2f}$")

    H_init = entropy(traj[0])
    H_final = entropy(traj[-1])
    ax.set_xlabel("Update step $t$")
    ax.set_ylabel(r"Student probability $\pi_S(v)$")
    ax.set_title(rf"{title}   ($\alpha_S={alpha_S}$, $H(\pi_S^{{(0)}})={H_init:.2f}$ $\to$ $H(\pi_S^{{(T)}})={H_final:.2f}$)",
                 pad=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0, top=1.0)

    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.95,
                    edgecolor="#333333", handlelength=1.4, fontsize=6.5,
                    borderpad=0.3, labelspacing=0.22, handletextpad=0.4,
                    ncol=1)
    leg.get_frame().set_linewidth(0.5)


def main():
    init_rng = np.random.default_rng(SEED_INIT)
    teacher_supp, student_supp = make_disjoint_supports(init_rng)
    z_T = zipf_over_support(TEACHER_ALPHA, teacher_supp)

    # Two students share the same support but different α
    z_S_high_ent = zipf_over_support(STUDENT_ALPHA_HIGH_ENT, student_supp)
    z_S_low_ent  = zipf_over_support(STUDENT_ALPHA_LOW_ENT,  student_supp)

    traj_high, P = run(z_T, z_S_high_ent, seed=SEED_RUN)
    traj_low,  _ = run(z_T, z_S_low_ent,  seed=SEED_RUN)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    plot_panel(axes[0], traj_high, P, STUDENT_ALPHA_HIGH_ENT, "(a) High initial entropy")
    plot_panel(axes[1], traj_low,  P, STUDENT_ALPHA_LOW_ENT,  "(b) Low initial entropy")

    fig.suptitle(
        rf"Student's top-5 evolution under REINFORCE-RKL, disjoint supports "
        rf"($V={V}$, $K={TOP_K_SUPPORT}$, $\eta={LR}$)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "top5_two_entropy.png")
    out_pdf = os.path.join(OUT_DIR, "top5_two_entropy.pdf")
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
