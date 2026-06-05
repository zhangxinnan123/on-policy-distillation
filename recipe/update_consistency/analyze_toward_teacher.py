# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""B=1 single-token — "does the update move each token TOWARD the teacher?"

For each target position (single_token_mode), in (student π, teacher π) space:
  - sampled token u  → star marker at (π_S(u)_before, π_T(u))
  - unsampled c (teacher top-K) → circle at (π_S(c)_before, π_T(c))
Color each point by whether the ACTUAL update moved it toward the teacher:
  teacher direction  = sign(π_T − π_S_before)   (above diagonal: want ↑; below: want ↓)
  actual direction   = sign(π_S_after − π_S_before)
  GREEN  = aligned    (update moved toward teacher)
  RED    = not aligned (update moved AWAY from teacher)

  python -m recipe.update_consistency.analyze_toward_teacher \\
      --measurements .../measurements.parquet \\
      --output-dir   recipe/update_consistency/figures \\
      --suffix _b1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLOOR = 1e-12  # plotting floor for log axes


def _alignment(sub: pd.DataFrame):
    """Return (aligned_bool, valid_mask). Aligned = update moved toward teacher."""
    pis_b = sub["pi_S_before"].to_numpy()
    pis_a = sub["pi_S_after"].to_numpy()
    pit = sub["pi_T"].to_numpy()
    teacher_dir = np.sign(pit - pis_b)        # +1: teacher wants more; -1: less
    actual_dir = np.sign(pis_a - pis_b)       # +1: student increased; -1: decreased
    valid = (teacher_dir != 0) & (actual_dir != 0)
    aligned = teacher_dir == actual_dir
    return aligned, valid


def make_figure(t: pd.DataFrame, out: Path):
    u = t[t["is_sampled_u"]].copy()
    c = t[~t["is_sampled_u"]].copy()

    fig, ax = plt.subplots(figsize=(9.5, 9))

    # diagonal y = x (student already matches teacher)
    lo = FLOOR
    hi = 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1,
            label="π_T = π_S (no gap)")

    green = "#2ca02c"
    red = "#d62728"

    # ---- unsampled c (circles) ----
    al_c, val_c = _alignment(c)
    xs = np.clip(c["pi_S_before"].to_numpy(), FLOOR, None)
    ys = np.clip(c["pi_T"].to_numpy(), FLOOR, None)
    ax.scatter(xs[val_c & al_c], ys[val_c & al_c], s=20, alpha=0.55,
               c=green, marker="o", edgecolor="none", zorder=2,
               label=f"unsampled c — toward teacher ({int((val_c&al_c).sum())})")
    ax.scatter(xs[val_c & ~al_c], ys[val_c & ~al_c], s=20, alpha=0.55,
               c=red, marker="o", edgecolor="none", zorder=2,
               label=f"unsampled c — AWAY from teacher ({int((val_c&~al_c).sum())})")

    # ---- sampled u (stars) ----
    al_u, val_u = _alignment(u)
    xu = np.clip(u["pi_S_before"].to_numpy(), FLOOR, None)
    yu = np.clip(u["pi_T"].to_numpy(), FLOOR, None)
    ax.scatter(xu[val_u & al_u], yu[val_u & al_u], s=180, alpha=0.9,
               c=green, marker="*", edgecolor="black", linewidth=0.6, zorder=4,
               label=f"sampled u — toward teacher ({int((val_u&al_u).sum())})")
    ax.scatter(xu[val_u & ~al_u], yu[val_u & ~al_u], s=180, alpha=0.9,
               c=red, marker="*", edgecolor="black", linewidth=0.6, zorder=4,
               label=f"sampled u — AWAY from teacher ({int((val_u&~al_u).sum())})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(FLOOR, 1.3)
    ax.set_ylim(FLOOR, 1.3)
    ax.set_xlabel("student probability  π_S(token)  [before update]")
    ax.set_ylabel("teacher probability  π_T(token)")

    frac_c = (al_c[val_c].mean()) if val_c.sum() else float("nan")
    frac_u = (al_u[val_u].mean()) if val_u.sum() else float("nan")
    ax.set_title(
        "Does the OPD update move each token toward the teacher?\n"
        f"sampled u: {frac_u:.1%} toward teacher  |  "
        f"unsampled c: {frac_c:.1%} toward teacher  "
        "(green=toward, red=away)")

    # annotate regions
    ax.text(0.02, 0.97, "above diagonal:\nteacher wants ↑ (π_T > π_S)",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            color="gray")
    ax.text(0.97, 0.03, "below diagonal:\nteacher wants ↓ (π_T < π_S)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            color="gray")

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return frac_u, frac_c


def _decode(tokenizer, tid):
    """Decode one token id to a short display string with whitespace made visible."""
    s = tokenizer.decode([int(tid)])
    return repr(s)[1:-1] or " "  # strip the repr quotes; show \n, spaces, etc.


def make_per_token_grid(t: pd.DataFrame, out: Path, linear=False, delta_pi_floor=0.0,
                        tokenizer=None):
    """One subplot per sampled token (= per step in B=1 single-token mode).
    Each subplot: the sampled u (star) + its teacher-top-K unsampled c (circles),
    colored green=toward / red=away from teacher.

    delta_pi_floor: keep only tokens whose |π_S_after − π_S_before| > floor
    (drop tiny, near-bf16-floor changes). Positions left with no token are skipped.
    tokenizer: if given, the sampled token is shown in each title and every
    circle is annotated with its decoded text.
    """
    green = "#2ca02c"
    red = "#d62728"
    if delta_pi_floor > 0:
        dpi = np.abs(t["pi_S_after"].to_numpy() - t["pi_S_before"].to_numpy())
        t = t[dpi > delta_pi_floor].copy()
    # group by (step, sequence_id, response_pos) — each is one sampled-token event
    keys = ["step", "sequence_id", "response_pos"]
    groups = [(k, g) for k, g in t.groupby(keys) if len(g)]
    n = len(groups)
    ncols = 7
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.3, nrows * 2.3),
                             squeeze=False)

    for idx, (key, g) in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        u = g[g["is_sampled_u"]]
        c = g[~g["is_sampled_u"]]

        if linear:
            xs = np.clip(c["pi_S_before"].to_numpy(), 0, None)
            ys = np.clip(c["pi_T"].to_numpy(), 0, None)
            xu = np.clip(u["pi_S_before"].to_numpy(), 0, None)
            yu = np.clip(u["pi_T"].to_numpy(), 0, None)
            hi = float(max(np.r_[xs, ys, xu, yu].max(), 1e-6)) * 1.1  # per-subplot autoscale
            ax.plot([0, hi], [0, hi], "k--", lw=0.6, zorder=1)
        else:
            xs = np.clip(c["pi_S_before"].to_numpy(), FLOOR, None)
            ys = np.clip(c["pi_T"].to_numpy(), FLOOR, None)
            xu = np.clip(u["pi_S_before"].to_numpy(), FLOOR, None)
            yu = np.clip(u["pi_T"].to_numpy(), FLOOR, None)
            ax.plot([FLOOR, 1.0], [FLOOR, 1.0], "k--", lw=0.6, zorder=1)

        # unsampled c
        al_c, val_c = _alignment(c)
        ax.scatter(xs[val_c & al_c], ys[val_c & al_c], s=14, alpha=0.7,
                   c=green, marker="o", edgecolor="none", zorder=2)
        ax.scatter(xs[val_c & ~al_c], ys[val_c & ~al_c], s=14, alpha=0.7,
                   c=red, marker="o", edgecolor="none", zorder=2)

        # sampled u
        al_u, val_u = _alignment(u)
        u_color = green if (val_u.any() and al_u[val_u][0]) else red
        ax.scatter(xu, yu, s=120, alpha=0.95, c=u_color, marker="*",
                   edgecolor="black", linewidth=0.6, zorder=4)

        # annotate each token with its decoded text
        if tokenizer is not None and "candidate_token_id" in g:
            for xx, yy, tid in zip(xs, ys, c["candidate_token_id"].to_numpy()):
                ax.annotate(_decode(tokenizer, tid), (xx, yy),
                            fontsize=3.6, alpha=0.8, color="0.25",
                            xytext=(2, 1), textcoords="offset points", zorder=5)
            if len(u):
                ax.annotate(_decode(tokenizer, int(u["candidate_token_id"].iloc[0])),
                            (xu[0], yu[0]), fontsize=4.5, fontweight="bold",
                            color="black", xytext=(2, 2),
                            textcoords="offset points", zorder=6)

        # τ watershed: x = τ (same units as student π). Theory says the c-update
        # sign flips at π_c = τ, so draw it only when τ>0 (i.e. plottable / in-range).
        tau_val = float(g["tau"].iloc[0]) if "tau" in g and len(g) else float("nan")
        if np.isfinite(tau_val) and tau_val > 0:
            ax.axvline(tau_val, color="#7b1fa2", lw=1.0, ls="--", alpha=0.8,
                       zorder=3)

        if linear:
            ax.set_xlim(0, hi); ax.set_ylim(0, hi)
            ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        else:
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(FLOOR, 1.3); ax.set_ylim(FLOOR, 1.3)
            ax.set_xticks([1e-9, 1e-3]); ax.set_yticks([1e-9, 1e-3])
        ax.tick_params(labelsize=6)
        a_val = float(u["A_value"].iloc[0]) if len(u) else float("nan")
        frac = al_c[val_c].mean() if val_c.sum() else float("nan")
        step = key[0]
        # τ>0 marks the regime where the τ-watershed sign rule has bite; flag it.
        tau_pos = np.isfinite(tau_val) and tau_val > 0
        tau_tag = "  τ>0 ✓" if tau_pos else ""
        u_tok = ""
        if tokenizer is not None and len(u):
            u_tok = f"  u='{_decode(tokenizer, int(u['candidate_token_id'].iloc[0]))}'"
        ax.set_title(
            f"step {step}  A={a_val:.1f}{tau_tag}{u_tok}\n"
            f"τ={tau_val:.1e}  →teacher {frac:.0%}",
            fontsize=6.5,
            color=("#7b1fa2" if tau_pos else "black"))

    # hide unused cells
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.supxlabel("student probability  π_S  [before]", fontsize=11)
    fig.supylabel("teacher probability  π_T", fontsize=11)
    fig.suptitle(
        f"Per-sampled-token view (n={n})  •  star=sampled u, circles=unsampled c  "
        "•  green=toward teacher, red=away",
        fontsize=12, y=1.005)
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return n


def make_tau_figure(t: pd.DataFrame, out: Path):
    """Two-panel τ analysis (x-axis = τ).

    (a) per-position: x = τ (= ‖π_S‖² − π_S(u)), y = fraction of the position's
        unsampled-c tokens that moved toward teacher. Colored by sign(A).
        Tests whether τ>0 positions recover the c-sign rule better.
    (b) per-candidate watershed: x = (τ − π_c), y = binned c→teacher fraction.
        Theory predicts the c-update sign flips at τ − π_c = 0, so alignment
        should be a step around x=0 (high on the side matching sign(A)).
    """
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    green = "#2ca02c"

    # ---------- (a) per-position alignment vs τ ----------
    keys = ["step", "sequence_id", "response_pos"]
    rows = []
    for _, g in t.groupby(keys):
        c = g[~g["is_sampled_u"]]
        al_c, val_c = _alignment(c)
        if val_c.sum() == 0:
            continue
        rows.append({
            "tau": float(g["tau"].iloc[0]),
            "A": float(g[g["is_sampled_u"]]["A_value"].iloc[0]) if g["is_sampled_u"].any() else float("nan"),
            "frac": float(al_c[val_c].mean()),
            "n": int(val_c.sum()),
        })
    pos = pd.DataFrame(rows)
    a_pos = pos[pos["A"] > 0]
    a_neg = pos[pos["A"] < 0]
    axA.axvline(0, color="0.6", lw=1, ls="--", zorder=1)
    axA.axhline(0.5, color="0.8", lw=1, ls=":", zorder=1)
    axA.scatter(a_pos["tau"], a_pos["frac"], s=18 + 4 * a_pos["n"], alpha=0.7,
                c="#1f77b4", edgecolor="none", label=f"A>0 ({len(a_pos)})")
    axA.scatter(a_neg["tau"], a_neg["frac"], s=18 + 4 * a_neg["n"], alpha=0.7,
                c="#d62728", edgecolor="none", label=f"A<0 ({len(a_neg)})")
    axA.set_xlabel("τ  = ‖π_S‖² − π_S(u)   [per position]")
    axA.set_ylabel("unsampled-c fraction → teacher")
    axA.set_ylim(-0.03, 1.03)
    axA.set_title(f"(a) per-position c-alignment vs τ   (n={len(pos)} positions)\n"
                  f"τ>0: {pos[pos.tau>0]['frac'].mean():.1%}  |  "
                  f"τ≤0: {pos[pos.tau<=0]['frac'].mean():.1%}  toward teacher")
    axA.legend(fontsize=8)

    # ---------- (b) per-candidate watershed vs (τ − π_c) ----------
    c = t[~t["is_sampled_u"]].copy()
    al_c, val_c = _alignment(c)
    c = c[val_c].copy()
    c["aligned"] = al_c[val_c].astype(float)
    c["x"] = c["tau"].to_numpy() - c["pi_S_before"].to_numpy()  # τ − π_c
    # bin on a signed-log x so the watershed at 0 is visible
    xb = c["x"].to_numpy()
    edges = np.r_[-np.logspace(0, -6, 13), 0.0, np.logspace(-6, 0, 13)]
    centers, fracs, counts = [], [], []
    for i in range(len(edges) - 1):
        m = (xb >= edges[i]) & (xb < edges[i + 1])
        if m.sum() >= 3:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            fracs.append(c["aligned"].to_numpy()[m].mean())
            counts.append(int(m.sum()))
    axB.axvline(0, color="0.6", lw=1.2, ls="--", zorder=1, label="τ − π_c = 0 (watershed)")
    axB.axhline(0.5, color="0.8", lw=1, ls=":", zorder=1)
    sc = axB.scatter(centers, fracs, s=[10 + c_ for c_ in counts], c=fracs,
                     cmap="RdYlGn", vmin=0, vmax=1, edgecolor="black", lw=0.4, zorder=3)
    axB.plot(centers, fracs, color="0.4", lw=0.8, zorder=2)
    axB.set_xscale("symlog", linthresh=1e-6)
    axB.set_xlabel("τ − π_c   (symlog; theory sign flips at 0)")
    axB.set_ylabel("c fraction → teacher  (binned)")
    axB.set_ylim(-0.03, 1.03)
    left = c[c["x"] < 0]["aligned"].mean()
    right = c[c["x"] > 0]["aligned"].mean()
    axB.set_title(f"(b) per-candidate watershed   (n={len(c)} c-tokens)\n"
                  f"τ−π_c<0: {left:.1%}  |  τ−π_c>0: {right:.1%}  toward teacher")
    fig.colorbar(sc, ax=axB, label="→ teacher fraction", fraction=0.046, pad=0.04)

    fig.suptitle("Does τ govern the unsampled-c update sign?", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return len(pos), len(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--name", type=str, default="",
                   help="Subfolder under output-dir for this run. "
                        "Defaults to the parquet's parent dir name.")
    p.add_argument("--per-token", action="store_true",
                   help="Also emit a grid with one subplot per sampled token.")
    p.add_argument("--linear", action="store_true",
                   help="Use linear (per-subplot autoscaled) axes instead of log.")
    p.add_argument("--delta-pi-floor", type=float, default=0.0,
                   help="Grid only: keep tokens with |Δπ_S| > floor (drop tiny changes).")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer model path to decode token IDs for visualization.")
    args = p.parse_args()
    df = pd.read_parquet(args.measurements)
    n_target = int(df["is_target"].sum())
    t = df[df["is_target"]].copy() if n_target > 0 else df.copy()
    print(f"rows used: {len(t)} (target={n_target})")

    # Load tokenizer if provided
    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"Loaded tokenizer from: {args.tokenizer}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer from {args.tokenizer}: {e}")

    run_name = args.name or args.measurements.parent.name
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    out = run_dir / f"toward_teacher{args.suffix}"
    frac_u, frac_c = make_figure(t, out)
    print(f"sampled u toward teacher: {frac_u:.3f}")
    print(f"unsampled c toward teacher: {frac_c:.3f}")
    print(f"Wrote: {out}.png (+ .pdf)")
    if args.per_token:
        floor_tag = f"_dpi{args.delta_pi_floor:.0e}" if args.delta_pi_floor > 0 else ""
        token_tag = "_tokens" if tokenizer else ""
        gname = f"toward_teacher_grid{'_linear' if args.linear else ''}{floor_tag}{token_tag}{args.suffix}"
        out_grid = run_dir / gname
        n = make_per_token_grid(t, out_grid, linear=args.linear,
                                delta_pi_floor=args.delta_pi_floor,
                                tokenizer=tokenizer)
        print(f"Wrote: {out_grid}.png (+ .pdf)  [{n} subplots]")

    out_tau = run_dir / f"toward_teacher_tau{args.suffix}"
    n_pos, n_c = make_tau_figure(t, out_tau)
    print(f"Wrote: {out_tau}.png (+ .pdf)  [{n_pos} positions, {n_c} c-tokens]")


if __name__ == "__main__":
    main()
