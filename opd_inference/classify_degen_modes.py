"""Classify degeneration modes across rollouts and quantify teacher agreement per mode.

Two modes were observed by hand on 5 rollouts and are separated automatically here:

  loop      a periodic tail — the model emits the same answer block forever.
            `repeat_onset` finds the period, so the loop is exactly identifiable.
  drift     no periodic tail, but the rollout still never terminates (finish=length).
            Locally fluent maths that never converges; the failure is semantic, not
            character-level. This is the stronger evidence: the teacher cannot excuse
            its silence by "the prefix is already degenerate".
  complete  finish != length, i.e. the model actually stopped.

For each rollout it reports teacher agreement on the *degenerate region* (the loop for
`loop`, the last third for `drift`) so the two modes can be compared on equal footing.

Usage:
    python opd_inference/classify_degen_modes.py \
        --jsonl /fsx/xinnanzh/eval_out/degen_vis/n20_tokens.jsonl \
        --tokenizer /fsx/xinnanzh/ckpts/degen_merged/step_200 \
        --out-md /fsx/xinnanzh/eval_out/degen_vis/n20_modes.md
"""

import argparse
import json

import numpy as np

from opd_inference.vis_token_advantage import repeat_onset


def agreement(A):
    A = np.asarray(A, dtype=float)
    A = A[np.isfinite(A)]
    if not len(A):
        return dict(n=0, mean=np.nan, share05=np.nan, share2=np.nan, mx=np.nan)
    return dict(n=len(A), mean=float(A.mean()),
                share05=100.0 * float(np.mean(np.abs(A) < 0.5)),
                share2=100.0 * float(np.mean(np.abs(A) > 2.0)),
                mx=float(np.abs(A).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--probe", type=int, default=48)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows = [json.loads(l) for l in open(args.jsonl)]
    recs = []
    for i, r in enumerate(rows, 1):
        ids = list(r["token_ids"])
        n = len(ids)
        onset, period = repeat_onset(ids, probe=args.probe)
        if r["finish_reason"] != "length":
            mode, lo = "complete", max(0, n - n // 3)
        elif onset is not None:
            mode, lo = "loop", onset
        else:
            mode, lo = "drift", max(0, n - n // 3)
        recs.append(dict(
            i=i, n=n, finish=r["finish_reason"], mode=mode, onset=onset, period=period,
            loop_frac=(100.0 * (n - onset) / n) if onset is not None else np.nan,
            cycle=(tok.decode(ids[onset:onset + period]) if onset is not None else ""),
            full=agreement(r["A"]), degen=agreement(np.asarray(r["A"])[lo:]),
        ))

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Degeneration modes, {len(recs)} rollouts")
    emit()
    counts = {}
    for m in ("loop", "drift", "complete"):
        counts[m] = sum(1 for r in recs if r["mode"] == m)
    emit(f"`loop` {counts['loop']} &middot; `drift` {counts['drift']} "
         f"&middot; `complete` {counts['complete']}")
    emit()
    emit("| # | mode | tokens | finish | onset | period | loop % | "
         "degen mean A | \\|A\\|<0.5 | \\|A\\|>2 | max\\|A\\| |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        d = r["degen"]
        lf = "—" if np.isnan(r["loop_frac"]) else f"{r['loop_frac']:.0f}"
        emit(f"| {r['i']} | {r['mode']} | {r['n']} | {r['finish']} | "
             f"{r['onset'] if r['onset'] is not None else '—'} | "
             f"{r['period'] if r['period'] is not None else '—'} | "
             f"{lf} | "
             f"{d['mean']:+.3f} | {d['share05']:.1f}% | {d['share2']:.1f}% | {d['mx']:.2f} |")

    emit()
    emit("## Teacher agreement by mode (degenerate region only)")
    emit()
    emit("| mode | n rollouts | mean A | \\|A\\|<0.5 | \\|A\\|>2 |")
    emit("|---|---|---|---|---|")
    for m in ("loop", "drift", "complete"):
        sub = [r for r in recs if r["mode"] == m]
        if not sub:
            continue
        emit(f"| {m} | {len(sub)} | "
             f"{np.mean([r['degen']['mean'] for r in sub]):+.3f} | "
             f"{np.mean([r['degen']['share05'] for r in sub]):.1f}% | "
             f"{np.mean([r['degen']['share2'] for r in sub]):.1f}% |")

    emit()
    emit("## Repeated cycles")
    emit()
    for r in recs:
        if r["mode"] == "loop":
            emit(f"- **{r['i']}** period {r['period']}: `{r['cycle'][:160]!r}`")

    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
