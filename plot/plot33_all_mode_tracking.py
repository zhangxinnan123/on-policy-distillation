"""Track all mode-candidate tokens across REINFORCE-RKL evolution.

Each step, sample 1 token from pi_S and update. Then for the whole trajectory:
  1. For each token that was EVER student's argmax at any step, plot its
     probability trajectory over time.
  2. Highlight teacher's top-K tokens (fixed) with distinct color.
  3. Mark mode-switch events (when argmax changes).

Rows: same 4 mismatch scenarios as plot32.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

V = 100
N_STEPS = 500
LR = 0.05
TOP_K_TEACHER = 5     # how many teacher-top-K tokens to highlight
SHOW_TOP_STUDENT_MODES = 8   # up to how many "ever-been-mode" tokens to plot

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
    """Run 1 seed, return full-vocab pi_S trajectory (T+1, V), sampled ids, and A_u."""
    rng = np.random.default_rng(seed)
    z_S = z_S_init.copy()
    P = softmax(z_T_init)
    pi = softmax(z_S)
    traj = np.zeros((N_STEPS + 1, V))
    traj[0] = pi
    sampled = np.full(N_STEPS + 1, -1, dtype=int)
    A_at = np.zeros(N_STEPS + 1)
    for t in range(1, N_STEPS + 1):
        u = int(rng.choice(V, p=pi))
        A_u = float(np.log(P[u] + 1e-30) - np.log(pi[u] + 1e-30))
        z_S = z_S + LR * A_u * (np.eye(V)[u] - pi)
        pi = softmax(z_S)
        traj[t] = pi
        sampled[t] = u
        A_at[t] = A_u
    return traj, sampled, A_at, P


SCENARIOS = [
    ("sharp_mismatch",  "Sharp mismatch"),
    ("sharp_S_flat_T",  "Overconfident S, flat T"),
    ("partial_overlap", "Partial overlap (s0 = T rank-2)"),
    ("flat_S_sharp_T",  "Flat S, sharp T"),
]


def main():
    fig, axes = plt.subplots(len(SCENARIOS), 3, figsize=(18, 4.5 * len(SCENARIOS)))

    for row, (sc, title) in enumerate(SCENARIOS):
        seed_rng = np.random.default_rng(0)
        z_T, z_S, t_star, s0 = make_pair(sc, seed_rng)
        traj, sampled, A_at, P = run_single(z_T, z_S, seed=1)   # single seed for token-level detail
        argmax_over_time = traj.argmax(axis=1)                   # (T+1,)

        # Collect every token that was ever student's mode
        ever_mode = np.unique(argmax_over_time)
        # If too many, keep the ones with longest tenure
        if len(ever_mode) > SHOW_TOP_STUDENT_MODES:
            tenure = np.array([(argmax_over_time == tok).sum() for tok in ever_mode])
            ever_mode = ever_mode[np.argsort(tenure)[::-1][:SHOW_TOP_STUDENT_MODES]]

        teacher_topk = np.argsort(-P)[:TOP_K_TEACHER]
        # Mode-switch events: steps where argmax changes
        switches = np.where(np.diff(argmax_over_time) != 0)[0] + 1

        # =============== Panel 1: track all EVER-mode tokens ===============
        ax = axes[row, 0]
        steps = np.arange(N_STEPS + 1)
        # Color by teacher rank if in teacher-topK, gray otherwise
        cmap_T = plt.get_cmap("Blues")
        for tok in ever_mode:
            if tok in teacher_topk:
                t_rank = int(np.where(teacher_topk == tok)[0][0])
                color = cmap_T(1 - 0.15 * t_rank)  # T rank-1 darkest
                label = f"tok {tok} (T-rank {t_rank+1})"
                lw = 2.0
            elif tok == s0:
                color = "#DC2626"
                label = f"tok {tok} (s0 init)"
                lw = 2.0
            else:
                color = "#9CA3AF"
                label = f"tok {tok}"
                lw = 1.0
            ax.plot(steps, traj[:, tok], color=color, lw=lw, label=label, alpha=0.85)

        # Mark switches with faint vertical lines
        for sw in switches[::max(1, len(switches) // 30)]:  # subsample if too many
            ax.axvline(sw, color="black", lw=0.3, alpha=0.15)

        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("π_S(token)", fontsize=9)
        ax.set_title(f"π_S trajectory for every token that was ever argmax\n"
                     f"({len(ever_mode)} shown; teacher top-{TOP_K_TEACHER} in blue)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.85)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)

        # =============== Panel 2: argmax over time (mode ID trace) ===============
        ax = axes[row, 1]
        ax.plot(steps, argmax_over_time, drawstyle="steps-post", color="black",
                lw=1.5, label="argmax(π_S)")
        # Overlay sampled tokens as scatter, color=A_u
        sc_pts = ax.scatter(steps[1:], sampled[1:], c=A_at[1:], cmap="RdBu",
                            s=6, alpha=0.5, vmin=-4, vmax=4)
        # Teacher top-K as horizontal ref lines
        for tk_i, tok in enumerate(teacher_topk):
            ax.axhline(tok, color=plt.get_cmap("Blues")(1 - 0.15 * tk_i),
                       ls="--", lw=0.8, alpha=0.8,
                       label=f"T rank-{tk_i+1} (tok {tok})" if tk_i == 0 or tk_i == len(teacher_topk)-1 else None)
        ax.axhline(s0, color="#DC2626", ls="--", lw=1.0, alpha=0.7, label=f"s0={s0}")
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("token id", fontsize=9)
        ax.set_title(f"argmax trajectory + sampled tokens\n"
                     f"({len(switches)} mode switches; color = A_u at sample)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_ylim(-2, V + 2)
        plt.colorbar(sc_pts, ax=ax, fraction=0.03, pad=0.02, label="A_u")

        # =============== Panel 3: mode-tenure summary + teacher mass ===============
        ax = axes[row, 2]
        # Bar chart: how many steps each ever-mode token was argmax
        tenure = np.array([(argmax_over_time == tok).sum() for tok in ever_mode])
        # Also plot teacher mass on each of these tokens (fixed)
        teacher_mass_on_mode = P[ever_mode]
        order = np.argsort(tenure)[::-1]
        ever_mode_sorted = ever_mode[order]
        tenure_sorted = tenure[order]
        tmass_sorted = teacher_mass_on_mode[order]
        bar_colors = ["#2563EB" if tok in teacher_topk else
                      ("#DC2626" if tok == s0 else "#9CA3AF")
                      for tok in ever_mode_sorted]
        xpos = np.arange(len(ever_mode_sorted))
        ax.bar(xpos, tenure_sorted, color=bar_colors, alpha=0.85)
        ax2 = ax.twinx()
        ax2.plot(xpos, tmass_sorted, "o-", color="black", lw=1.2, ms=6,
                 label="teacher π_T(tok)")
        ax2.set_ylabel("teacher mass π_T(tok)", fontsize=9)
        ax2.legend(fontsize=7, loc="upper right")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{t}" for t in ever_mode_sorted], fontsize=7)
        ax.set_xlabel("token id (sorted by tenure)", fontsize=9)
        ax.set_ylabel("# steps as argmax", fontsize=9)
        ax.set_title("Time spent as mode + teacher's mass\n"
                     "(blue=teacher top-K, red=s0, gray=other)",
                     fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y")

        # Row label
        axes[row, 0].annotate(title, xy=(-0.25, 0.5), xycoords="axes fraction",
                              ha="right", va="center", fontsize=11,
                              fontweight="bold", rotation=0)

    fig.suptitle(f"All-mode-token tracking under mismatch (V={V}, LR={LR}, {N_STEPS} steps)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0.04, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "all_mode_tracking.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
