import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Plot teacher vs student token log-probability analysis.")
    p.add_argument("--input", default="lzy_eval_result/teacher_inference/20260408-223418_results/20260409-172843_Qwen3-32B_scored.jsonl", help="Path to scored .jsonl file.")
    p.add_argument("--teacher", default="Qwen3-32B", help="Teacher model name (e.g. OpenThinker3-7B).")
    p.add_argument("--student", default="Qwen3-4B-SFT", help="Student model name (e.g. Qwen3-4B-SFT).")
    p.add_argument("--out", default=None, help="Output PNG path. Default: <input_dir>/logprob_analysis.png")
    args = p.parse_args()

    filepath = args.input
    teacher_label = args.teacher or "Teacher"
    student_label = args.student or "Student"

    student_logps = []
    teacher_logps = []
    token_positions = []

    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            s = data['response_token_logprobs']
            t = data['teacher_logp_response_token_logprobs']
            for pos, (sv, tv) in enumerate(zip(s, t)):
                if sv is not None and tv is not None and np.isfinite(sv) and np.isfinite(tv):
                    student_logps.append(sv)
                    teacher_logps.append(tv)
                    token_positions.append(pos)

    student_logps = np.array(student_logps)
    teacher_logps = np.array(teacher_logps)
    token_positions = np.array(token_positions)
    diff = teacher_logps - student_logps
    abs_diff = np.abs(diff)

    print(f"Total tokens: {len(student_logps)}")
    print(f"Student logp: min={student_logps.min():.3f}, max={student_logps.max():.3f}, mean={student_logps.mean():.3f}, median={np.median(student_logps):.3f}")
    print(f"Teacher logp: min={teacher_logps.min():.3f}, max={teacher_logps.max():.3f}, mean={teacher_logps.mean():.3f}, median={np.median(teacher_logps):.3f}")
    print(f"Diff (teacher-student): min={diff.min():.3f}, max={diff.max():.3f}, mean={diff.mean():.3f}, median={np.median(diff):.3f}")
    print(f"Abs diff: min={abs_diff.min():.3f}, max={abs_diff.max():.3f}, mean={abs_diff.mean():.3f}, median={np.median(abs_diff):.3f}")
    print(f"P(diff > 0): {(diff > 0).mean():.4f}")
    print(f"P(|diff| > 1): {(abs_diff > 1).mean():.4f}")
    print(f"P(|diff| > 2): {(abs_diff > 2).mean():.4f}")
    print(f"P(|diff| > 5): {(abs_diff > 5).mean():.4f}")

    for q in [0.50, 0.80, 0.90, 0.95, 0.99]:
        print(f"|diff| q{int(q*100):02d}: {np.quantile(abs_diff, q):.4f}")

    # -------------------------
    # Hard-coded clipping limits
    # -------------------------
    logp_low = -10.0
    logp_high = 0.0

    student_c = np.clip(student_logps, logp_low, logp_high)
    teacher_c = np.clip(teacher_logps, logp_low, logp_high)

    diff_low = -10.0
    diff_high = 4.0
    diff_c = np.clip(diff, diff_low, diff_high)

    abs_diff_high = 10.0
    abs_diff_c = np.clip(abs_diff, 0, abs_diff_high)

    corr = np.corrcoef(student_logps, teacher_logps)[0, 1]
    mean_diff = diff.mean()
    median_diff = np.median(diff)
    q05, q25, q75, q95 = np.quantile(diff, [0.05, 0.25, 0.75, 0.95])
    p80, p90, p95_abs, p99 = np.quantile(abs_diff, [0.80, 0.90, 0.95, 0.99])

    # -------------------------
    # Plot styling
    # -------------------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        f"Token Log-Probability Analysis  |  Teacher: {teacher_label}  vs  Student: {student_label}",
        fontsize=15, fontweight="bold",
    )

    # -------------------------
    # 1) Teacher vs Student hexbin
    # -------------------------
    ax = axes[0, 0]
    hb1 = ax.hexbin(student_c, teacher_c, gridsize=70, mincnt=1, bins='log', cmap='viridis')
    lims = [logp_low, 0.0]
    ax.plot(lims, lims, '--', linewidth=1.5, color='red')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{student_label} token log-prob")
    ax.set_ylabel(f"{teacher_label} token log-prob")
    ax.set_title(f"{teacher_label} vs {student_label} (r = {corr:.3f})")
    cb1 = fig.colorbar(hb1, ax=ax)
    cb1.set_label("log10(count)")

    # -------------------------
    # 2) Signed diff histogram
    # -------------------------
    ax = axes[0, 1]
    bins = np.linspace(diff_low, diff_high, 180)
    neg_mask = diff_c <= 0
    pos_mask = diff_c > 0
    ax.hist(diff_c[neg_mask], bins=bins, alpha=0.85, color='steelblue', label=f'{teacher_label} < {student_label}')
    ax.hist(diff_c[pos_mask], bins=bins, alpha=0.85, color='darkorange', label=f'{teacher_label} > {student_label}')
    ax.axvline(0, linestyle='--', linewidth=1.5, color='red', label='0')
    ax.axvline(mean_diff, linestyle='--', linewidth=1.5, color='purple', label=f'mean={mean_diff:.2f}')
    ax.axvline(median_diff, linestyle='--', linewidth=1.5, color='green', label=f'median={median_diff:.2f}')
    ax.axvspan(q05, q95, alpha=0.30, color='gray', label='5%-95%')
    ax.set_yscale('log')
    ax.set_xlabel(f"{teacher_label} logp - {student_label} logp")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Signed Difference Distribution")
    ax.legend(frameon=False)

    # -------------------------
    # 3) |diff| histogram with percentiles
    # -------------------------
    ax = axes[0, 2]
    ax.hist(abs_diff_c, bins=180, alpha=0.85)
    ax.set_yscale('log')
    ax.axvline(p80, linestyle='--', linewidth=1.5, color='#4daf4a', label=f'80%={p80:.2f}')
    ax.axvline(p90, linestyle='--', linewidth=1.5, color='#377eb8', label=f'90%={p90:.2f}')
    ax.axvline(p95_abs, linestyle='--', linewidth=1.5, color='#ff7f00', label=f'95%={p95_abs:.2f}')
    ax.axvline(p99, linestyle='--', linewidth=1.5, color='#e41a1c', label=f'99%={p99:.2f}')
    ax.set_xlabel(f"|{teacher_label} logp - {student_label} logp|")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Absolute Difference Distribution")
    ax.legend(frameon=False)

    # -------------------------
    # 4) Teacher logp vs diff
    # -------------------------
    ax = axes[1, 0]
    hb2 = ax.hexbin(teacher_c, diff_c, gridsize=70, mincnt=1, bins='log', cmap='plasma')
    ax.axhline(0, linestyle='--', linewidth=1.5, color='red')
    ax.axhline(mean_diff, linestyle='--', linewidth=1.2, color='orange')
    ax.set_xlabel(f"{teacher_label} token log-prob")
    ax.set_ylabel(f"{teacher_label} - {student_label} log-prob")
    ax.set_title(f"Difference vs {teacher_label} Confidence")
    cb2 = fig.colorbar(hb2, ax=ax)
    cb2.set_label("log10(count)")

    # -------------------------
    # 5) Student logp vs diff
    # -------------------------
    ax = axes[1, 1]
    hb3 = ax.hexbin(student_c, diff_c, gridsize=70, mincnt=1, bins='log', cmap='inferno')
    ax.axhline(0, linestyle='--', linewidth=1.5, color='red')
    ax.axhline(mean_diff, linestyle='--', linewidth=1.2, color='orange')
    ax.set_xlabel(f"{student_label} token log-prob")
    ax.set_ylabel(f"{teacher_label} - {student_label} log-prob")
    ax.set_title(f"Difference vs {student_label} Confidence")
    cb3 = fig.colorbar(hb3, ax=ax)
    cb3.set_label("log10(count)")

    # -------------------------
    # 6) Diff vs token position
    # -------------------------
    ax = axes[1, 2]
    max_pos = token_positions.max()
    bin_size = max(1, max_pos // 100)
    bins = np.arange(0, max_pos + bin_size, bin_size)
    bin_idx = np.digitize(token_positions, bins) - 1
    bin_centers = bins[:-1] + bin_size / 2

    mean_diff_pos = np.array([diff[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(len(bins) - 1)])
    mean_abs_diff_pos = np.array([np.abs(diff[bin_idx == i]).mean() if (bin_idx == i).any() else np.nan for i in range(len(bins) - 1)])

    ax.plot(bin_centers, mean_diff_pos, color='steelblue', linewidth=1.5, label='mean diff')
    ax.plot(bin_centers, mean_abs_diff_pos, color='darkorange', linewidth=1.5, label='mean |diff|')
    ax.axhline(0, linestyle='--', linewidth=1.0, color='red')
    ax.set_xlabel("Token position in sequence")
    ax.set_ylabel(f"Mean diff ({teacher_label} - {student_label})")
    ax.set_title("Diff vs Token Position")
    ax.legend(frameon=False)

    out = args.out or str(Path(filepath).parent / f"logprob_analysis_{teacher_label}_vs_{student_label}.png")
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
