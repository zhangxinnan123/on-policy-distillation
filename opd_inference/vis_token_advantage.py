"""Token-level visualization of teacher-vs-student log-prob gap on student rollouts.

Generates N rollouts from a student checkpoint, then scores every generated token with
the teacher, and writes an HTML page where each token is shaded by

    A = log pi_T(u) - log pi_S(u)

the same per-token advantage the OPD masks route on (`hybrid_masks.py`, `A_value` in the
update_consistency dumps). Blue = teacher likes the token *more* than the student
(A > 0, student under-confident). Red = student over-confident (A < 0) — the regime
REINFORCE-RKL is documented to handle badly, so degenerate spans should light up red.

Two vLLM passes, run as SEPARATE PROCESSES via --stage. In-process `del llm` does not
release vLLM's memory pool (a first attempt OOM'd with 77.6 GiB still resident when the
teacher loaded), so the stages exchange an intermediate .rollouts.json:
  1. --stage student : generate, `logprobs=0` -> log pi_S at each sampled token
  2. --stage teacher : `prompt_logprobs=0` over prompt+response -> log pi_T, then HTML

Usage (on the cluster, inside an salloc/sbatch allocation):
    python opd_inference/vis_token_advantage.py \
        --student /fsx/xinnanzh/ckpts/degen_merged/step_200 \
        --teacher Qwen/Qwen3-8B \
        --n 5 --max-tokens 8192 \
        --out /fsx/xinnanzh/eval_out/degen_step200_tokens.html
"""

import argparse
import gc
import html
import json
import os

import numpy as np
import pandas as pd


def build_prompts(parquet, tokenizer, n, enable_thinking, seed):
    df = pd.read_parquet(parquet)
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    prompts, metas = [], []
    for i in idx:
        raw = df.iloc[int(i)][col]
        # verl parquets store prompt as a chat list [{role, content}, ...]
        if isinstance(raw, (list, np.ndarray)):
            msgs = [dict(m) for m in raw]
        else:
            msgs = [{"role": "user", "content": str(raw)}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(text)
        metas.append({"row": int(i), "question": msgs[-1]["content"]})
    return prompts, metas


def student_pass(args, prompts):
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.student, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enforce_eager=True,
              tensor_parallel_size=args.tp)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, logprobs=0)
    outs = llm.generate(prompts, sp)
    rollouts = []
    for o in outs:
        c = o.outputs[0]
        lp = []
        for tid, d in zip(c.token_ids, c.logprobs):
            lp.append(float(d[tid].logprob))
        rollouts.append({
            "prompt_ids": list(o.prompt_token_ids),
            "token_ids": list(c.token_ids),
            "logp_student": lp,
            "finish_reason": c.finish_reason,
        })
    return rollouts


def teacher_pass(args, rollouts):
    """Score the student's tokens with the teacher via prompt_logprobs."""
    from vllm import LLM, SamplingParams

    # prompt_logprobs materializes a (chunk_tokens, vocab) fp32 tensor. At 8k tokens x
    # 151936 vocab that is ~4.7 GiB in one allocation, which OOM'd on top of an engine
    # sized at 0.85. Cap the prefill chunk and leave headroom.
    llm = LLM(model=args.teacher, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enforce_eager=True,
              tensor_parallel_size=args.tp,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_num_seqs=1, enable_prefix_caching=False)
    # 1 sampled token is the minimum; we only want prompt_logprobs.
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    seqs = [{"prompt_token_ids": r["prompt_ids"] + r["token_ids"]} for r in rollouts]
    outs = llm.generate(seqs, sp)
    for r, o in zip(rollouts, outs):
        npr = len(r["prompt_ids"])
        pl = o.prompt_logprobs  # list aligned with the full id sequence; [0] is None
        tl = []
        for pos in range(npr, npr + len(r["token_ids"])):
            tid = r["token_ids"][pos - npr]
            d = pl[pos] if pos < len(pl) else None
            tl.append(float(d[tid].logprob) if d and tid in d else float("nan"))
        r["logp_teacher"] = tl
    return rollouts


def color_seq(lp, mid=0.5, floor=-6.0):
    """Diverging ramp on a probability: red = unlikely, white = `mid`, blue = confident.

    Same red/blue vocabulary as the A ramp so the two panels of --color-by paired can be
    read with one mental model. White sits at pi = `mid` (default 0.5), blue saturates at
    pi = 1, red saturates at log pi <= `floor`.
    """
    if not np.isfinite(lp):
        return "#ffffff", "#999999"
    lmid = float(np.log(mid))
    if lp >= lmid:
        t = float(np.clip((lp - lmid) / (0.0 - lmid), 0.0, 1.0))
        r, g, b = 1 - 0.75 * t, 1 - 0.45 * t, 1.0
    else:
        t = -float(np.clip((lp - lmid) / (floor - lmid), 0.0, 1.0))
        s_ = -t
        r, g, b = 1.0, 1 - 0.55 * s_, 1 - 0.60 * s_
    bg = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    return bg, ("#ffffff" if abs(t) > 0.75 else "#111111")


# diverging blue -> grey -> red, A>0 blue (teacher wants more), A<0 red (student overconfident)
def color(a, lim):
    if not np.isfinite(a):
        return "#ffffff", "#999999"
    t = float(np.clip(a / lim, -1.0, 1.0))
    if t >= 0:
        r, g, b = 1 - 0.75 * t, 1 - 0.45 * t, 1.0
    else:
        s = -t
        r, g, b = 1.0, 1 - 0.55 * s, 1 - 0.60 * s
    bg = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    fg = "#111111" if abs(t) < 0.75 else "#ffffff"
    return bg, fg


def ngram_onset(ids, k=12, run=200, frac=0.9):
    """First position from which k-grams are overwhelmingly recurrences of earlier text.

    `repeat_onset` only locks onto a *strictly* periodic tail. The text is usually
    near-repetitive long before that (e.g. "**Answer:**" alternating with "**Final
    Answer:**"), so anchoring on the strict period puts the whole window inside the
    degenerate region and every panel comes out blank. This detector fires as soon as
    `frac` of the k-grams in a `run`-token window have already appeared — on these
    rollouts that is up to 7069 tokens earlier.
    """
    n = len(ids)
    if n < run + k:
        return None
    seen, dup = {}, np.zeros(n, dtype=bool)
    for i in range(n - k + 1):
        g = tuple(ids[i:i + k])
        if g in seen:
            dup[i] = True
        else:
            seen[g] = i
    c = np.cumsum(np.concatenate([[0], dup.astype(int)]))
    rate = (c[run:] - c[:-run]) / run
    hit = np.where(rate >= frac)[0]
    return int(hit[0]) if len(hit) else None


def repeat_onset(ids, probe=48, min_period=4):
    """First index from which the tail is periodic — i.e. where the model starts looping.

    Take the last `probe` tokens as a signature, find its previous occurrence to get the
    period p, then walk backwards while ids[i] == ids[i + p]. Returns None if the tail is
    not periodic.
    """
    n = len(ids)
    if n < 2 * probe + min_period:
        return None, None
    tail = ids[n - probe:]
    # last earlier occurrence of the tail signature
    prev = None
    for start in range(n - probe - min_period, -1, -1):
        if ids[start:start + probe] == tail:
            prev = start
            break
    if prev is None:
        return None, None
    period = (n - probe) - prev
    if period < min_period:
        return None, None
    i = n - 1
    while i - period >= 0 and ids[i] == ids[i - period]:
        i -= 1
    return i + 1, period


def colorbar_html(args, mode=None):
    """Self-contained colour key with numeric ticks, sampled from the actual ramp.

    Rendered directly above each text block so one crop for the paper captures both
    the key and the tokens.
    """
    n = 64
    cb = mode or args.color_by
    if cb in ("A", "dual"):
        vals = np.linspace(-args.clip, args.clip, n)
        stops = ", ".join(color(v, args.clip)[0] for v in vals)
        ticks = [-args.clip, -args.clip / 2, 0.0, args.clip / 2, args.clip]
        tlab = "".join(f'<span>{t:+.2g}</span>' for t in ticks)
        lo_txt, hi_txt = "student over-confident", "teacher wants it more"
        title = ("A = log &pi;<sub>T</sub>(u) &minus; log &pi;<sub>S</sub>(u)"
                 "&nbsp;&nbsp;<span style='color:#6b6b6b'>white = identical log-probs"
                 "</span>")
        if cb == "dual":
            title += ("&nbsp;&nbsp;&middot;&nbsp;&nbsp;underline = &pi;<sub>S</sub> "
                      "<span style='box-shadow:inset 0 -3px 0 rgba(20,20,20,.55);"
                      "padding:0 10px'>&nbsp;</span> thick = confident")
    else:
        who = "T" if cb == "logpT" else "S"
        vals = np.linspace(-6.0, 0.0, n)
        stops = ", ".join(color_seq(v)[0] for v in vals)
        tlab = "".join(f'<span>{t:g}</span>' for t in [0.0025, 0.05, 0.5, 1.0])
        lo_txt, hi_txt = "unlikely", "confident"
        title = (f"&pi;<sub>{who}</sub>(u)&nbsp;&nbsp;<span style='color:#6b6b6b'>"
                 "white = 0.5</span>")
    return (f'<div class="key"><div class="keytitle">{title}</div>'
            f'<div class="keyrow"><span class="keyend">{lo_txt}</span>'
            f'<div class="keywrap"><div class="keybar" style="background:'
            f'linear-gradient(90deg, {stops})"></div>'
            f'<div class="keyticks">{tlab}</div></div>'
            f'<span class="keyend">{hi_txt}</span></div></div>')


def render(rollouts, metas, tokenizer, args):
    lim = args.clip
    if args.color_by == "paired":
        legend = ""
        shade = ("<b>A</b> in panel (i) and <b>&pi;<sub>S</sub></b> in panel (ii), over the "
                 "same tokens. Degeneration = panel (i) blank (teacher agrees) while panel "
                 "(ii) is saturated (student confident)")
    elif args.color_by == "dual":
        legend = ""
        shade = ("background = <b>A = log &pi;<sub>T</sub> &minus; log &pi;<sub>S</sub></b> "
                 f"(clip &plusmn;{lim:g}, white = teacher agrees), underline weight = "
                 "<b>&pi;<sub>S</sub></b> (thick = student confident). Degenerate spans read "
                 "as <i>white background + thick underline</i>: confident and unopposed")
    elif args.color_by == "A":
        legend = ('<span>A &lt; 0 &nbsp;student over-confident</span><div class="bar"></div>'
                  '<span>A &gt; 0 &nbsp;teacher wants it more</span>')
        shade = ("<b>A = log &pi;<sub>T</sub>(u) &minus; log &pi;<sub>S</sub>(u)</b>, "
                 f"clipped at &plusmn;{lim:g}")
    else:
        who = "T" if args.color_by == "logpT" else "S"
        legend = (f'<span>&pi;<sub>{who}</sub> &asymp; 0</span>'
                  '<div class="bar" style="background:linear-gradient(90deg,#ffffff,#3980eb)">'
                  f'</div><span>&pi;<sub>{who}</sub> = 1 &nbsp;(model confident)</span>')
        shade = (f"<b>log &pi;<sub>{who}</sub>(u)</b>, white at &pi;=0, "
                 "deep blue at &pi;=1 (ramp saturates below &pi;=e<sup>&minus;6</sup>)")
    parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(args.title)}</title><style>
body{{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
margin:0;padding:32px 40px;color:#1a1a1a;background:#fff;max-width:1180px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:32px 0 8px}}
.sub{{color:#6b6b6b;font-size:12.5px;margin:0 0 20px}}
.legend{{display:flex;align-items:center;gap:10px;margin:16px 0 26px;font-size:12px;color:#6b6b6b}}
.bar{{height:12px;width:320px;border-radius:2px;
background:linear-gradient(90deg,#ff7366,#ffffff,#4080ff)}}
.meta{{font-size:12px;color:#6b6b6b;margin:0 0 10px}}
.q{{background:#f6f7f9;border-left:3px solid #d0d4da;padding:9px 12px;margin:0 0 12px;
font-size:12.5px;white-space:pre-wrap;color:#333}}
.gen{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
line-height:2.0;white-space:pre-wrap;word-break:break-word;
border:1px solid #e6e8eb;border-radius:4px;padding:14px}}
tok{{padding:1.5px 0.5px;border-radius:2px}}
nl{{color:#b0b4ba;font-size:11px}}
byte{{color:#c8ccd1;font-size:9px;letter-spacing:-1px}}
.key{{margin:0 0 9px}}
.keytitle{{font-size:11.5px;margin:0 0 3px}}
.keyrow{{display:flex;align-items:center;gap:9px}}
.keyend{{font-size:10.5px;color:#6b6b6b;white-space:nowrap}}
.keywrap{{width:300px}}
.keybar{{height:11px;border:1px solid #d6d9dd;border-radius:2px}}
.panelcap{{font-size:12px;font-weight:600;margin:12px 0 5px}}
.keyticks{{display:flex;justify-content:space-between;font-size:10px;
color:#6b6b6b;margin-top:1px}}
</style></head><body>
<h1>{html.escape(args.title)}</h1>
<p class="sub">Each token shaded by {shade}. Student <code>{html.escape(args.student)}</code>,
teacher <code>{html.escape(args.teacher)}</code>. Sampling T={args.temperature},
top_p={args.top_p}, top_k={args.top_k}, max {args.max_tokens} tokens.
Hover a token for its exact values.</p>
"""]

    cap = args.max_display_tokens
    mode = args.color_by
    pairs = list(enumerate(zip(rollouts, metas), 1))
    if args.only:
        want = [int(x) for x in args.only.split(",")]
        byidx = {k: v for k, v in pairs}
        pairs = [(k, byidx[k]) for k in want if k in byidx]
    for k, (r, m) in pairs:
        A = np.array(r["logp_teacher"]) - np.array(r["logp_student"])
        fin = np.isfinite(A)
        lo = 0
        onset_note = ""
        if args.excerpt_len:
            L = min(args.excerpt_len, len(A))
            if args.excerpt_start is not None:
                lo = max(0, min(args.excerpt_start, len(A) - L))
            elif args.excerpt_at in ("repeat-onset", "ngram-onset"):
                if args.excerpt_at == "ngram-onset":
                    onset = ngram_onset(list(r["token_ids"]))
                    _, period = repeat_onset(list(r["token_ids"]))
                else:
                    onset, period = repeat_onset(list(r["token_ids"]))
                if onset is None:
                    lo = max(0, len(A) - L)
                    onset_note = " (no periodic tail found; showing the last window)"
                else:
                    lo = max(0, onset - args.pre_onset)
                    onset_note = (f" &middot; repetition onset at token {onset}, "
                                  f"period {period}")
            else:
                # window where the teacher is most confident = the strongest
                # "both models agree" evidence
                pt = np.array(r["logp_teacher"], dtype=float)
                pt = np.where(np.isfinite(pt), pt, -20.0)
                c = np.cumsum(np.concatenate([[0.0], pt]))
                means = (c[L:] - c[:-L]) / L
                lo = int(np.argmax(means))
            hi_ = min(lo + L, len(A))
            shown = hi_ - lo
            trunc = f" &middot; <b>excerpt [{lo}:{hi_}]</b>" + onset_note
        else:
            shown = len(A) if cap is None else min(cap, len(A))
            hi_ = shown
            trunc = ("" if shown == len(A) else
                     f" &middot; <b>showing first {shown}</b>")
        stats = (f"{len(A)} tokens{trunc} &middot; finish=<code>{r['finish_reason']}</code> "
                 f"&middot; mean A {np.nanmean(A):+.3f} &middot; "
                 f"share A&lt;0 {100 * np.mean(A[fin] < 0):.1f}% &middot; "
                 f"share A&lt;&minus;2 {100 * np.mean(A[fin] < -2):.1f}%")
        onset_k, period_k = repeat_onset(list(r["token_ids"]))
        if r["finish_reason"] != "length":
            mlabel = "terminated normally &mdash; control"
        elif onset_k is not None:
            mlabel = f"token-level loop (period {period_k})"
        else:
            mlabel = "semantic drift &mdash; no repetition, never terminates"
        parts.append(f'<h2>Rollout {k} &mdash; {mlabel} '
                     f'<span style="font-weight:400;color:#6b6b6b">'
                     f'(dataset row {m["row"]})</span></h2>')
        parts.append(f'<p class="meta">{stats}</p>')
        parts.append(f'<div class="q">{html.escape(m["question"][:900])}</div>')
        seg = list(zip(r["token_ids"], A, r["logp_student"], r["logp_teacher"]))[lo:hi_]
        pt_seg = np.array([x[3] for x in seg], dtype=float)
        a_seg = np.array([x[1] for x in seg], dtype=float)
        parts.append(
            f'<p class="meta">excerpt: median &pi;<sub>T</sub> = '
            f'{np.exp(np.nanmedian(pt_seg)):.3f} &middot; median &pi;<sub>S</sub> = '
            f'{np.exp(np.nanmedian([x[2] for x in seg])):.3f} &middot; '
            f'share |A| &lt; 0.5 = {100*np.nanmean(np.abs(a_seg) < 0.5):.1f}%</p>')

        # Multi-byte characters (emoji) are split across several BPE tokens, so decoding
        # one id at a time yields U+FFFD. Decode incrementally instead: each token shows
        # only the text it completes, so the character lands on its final byte-token and
        # the earlier ones render as a thin placeholder. Per-token colour is preserved.
        ids_seg = [int(x[0]) for x in seg]
        pieces, acc = [], []
        for j, tid in enumerate(ids_seg):
            acc.append(tid)
            txt_now = tokenizer.decode(acc, errors="replace")
            if "\ufffd" in txt_now and j + 1 < len(ids_seg):
                pieces.append(None)          # incomplete byte sequence, wait
            else:
                pieces.append(txt_now)
                acc = []

        def block(cmode):
            spans = []
            for (tid, a, ls, lt), piece in zip(seg, pieces):
                extra = ""
                if cmode == "logpT":
                    bg, fg = color_seq(lt)
                elif cmode == "logpS":
                    bg, fg = color_seq(ls)
                elif cmode == "dual":
                    bg, fg = color(a, lim)
                    w = float(np.clip(1.0 + (ls if np.isfinite(ls) else -6.0) / 4.0, 0.0, 1.0))
                    extra = f";box-shadow:inset 0 -{w*3.0:.1f}px 0 rgba(20,20,20,.55)"
                else:
                    bg, fg = color(a, lim)
                nlmark = '<nl>\\n</nl>' + ('<br>' if args.break_newlines else '')
                if piece is None:
                    txt = '<byte>\u2039\u203a</byte>'   # partial byte of a multi-byte char
                else:
                    txt = (html.escape(piece).replace("\n", nlmark)
                           .replace("\t", '<nl>\\t</nl>'))
                tip = f"A={a:+.3f}  logPs={ls:.3f}  logPt={lt:.3f}  id={tid}"
                spans.append(f'<tok style="background:{bg};color:{fg}{extra}" '
                             f'title="{html.escape(tip)}">{txt}</tok>')
            if hi_ < len(A) or lo > 0:
                spans.append('<span style="color:#6b6b6b">  \u2026 '
                             f'{len(A) - (hi_ - lo)} of {len(A)} tokens not shown</span>')
            return '<div class="gen">' + "".join(spans) + "</div>"

        if mode == "paired":
            for sub, cap_txt in (("A", "(i) teacher disagreement"),
                                 ("logpS", "(ii) student confidence")):
                parts.append(f'<p class="panelcap">{cap_txt}</p>')
                parts.append(colorbar_html(args, sub))
                parts.append(block(sub))
        else:
            parts.append(colorbar_html(args))
            parts.append(block(mode))

    parts.append("</body></html>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--parquet", default=os.path.expanduser(
        "~/data/dapo_17k_aime2426-suffix/test.parquet"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="degen was trained non-think; leave off to match it")
    ap.add_argument("--clip", type=float, default=5.0, help="A value mapped to full colour")
    ap.add_argument("--color-by", choices=["A", "logpT", "logpS", "dual", "paired"],
                    default="A",
                    help="logpT: absolute teacher confidence (use for the agreement figure)")
    ap.add_argument("--excerpt-len", type=int, default=None,
                    help="render a window of this many tokens instead of a prefix")
    ap.add_argument("--excerpt-start", type=int, default=None,
                    help="explicit window start; overrides --excerpt-at")
    ap.add_argument("--excerpt-at",
                    choices=["ngram-onset", "repeat-onset", "max-teacher-conf"],
                    default="ngram-onset",
                    help="ngram-onset: start before near-repetition begins (recommended); "
                         "repeat-onset: strictly periodic tail only, usually far too late")
    ap.add_argument("--only", default=None,
                    help="comma-separated 1-based rollout indices, in the order to render")
    ap.add_argument("--break-newlines", action="store_true",
                    help="also insert a real line break at each newline; default keeps the "
                         "text flowing so the figure stays compact")
    ap.add_argument("--pre-onset", type=int, default=64,
                    help="tokens of pre-repetition context to include before the onset")
    ap.add_argument("--jsonl", default=None,
                    help="--stage render: source .jsonl (default: derived from --out)")
    ap.add_argument("--max-display-tokens", type=int, default=None,
                    help="render only the first N tokens of each rollout (stats stay full-length)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048,
                    help="teacher prefill chunk; bounds the (tokens, vocab) logprob tensor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Token-level teacher-student gap on student rollouts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["student", "teacher", "render"], required=True,
                    help="run as two processes; vLLM does not free its pool in-process")
    args = ap.parse_args()
    if args.max_model_len is None:
        args.max_model_len = args.max_tokens + 2048

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    mid = os.path.splitext(args.out)[0] + ".rollouts.json"

    if args.stage == "student":
        prompts, metas = build_prompts(args.parquet, tok, args.n, args.enable_thinking, args.seed)
        print(f"{len(prompts)} prompts")
        rollouts = student_pass(args, prompts)
        for i, r in enumerate(rollouts, 1):
            print(f"  rollout {i}: {len(r['token_ids'])} tokens, finish={r['finish_reason']}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(mid, "w") as f:
            json.dump({"metas": metas, "rollouts": rollouts}, f)
        print(f"wrote {mid}")
        return

    if args.stage == "render":
        # rebuild the HTML from an existing .jsonl — no GPU, no model reload
        jl = args.jsonl or (os.path.splitext(args.out)[0] + ".jsonl")
        metas, rollouts = [], []
        with open(jl) as f:
            for line in f:
                d = json.loads(line)
                metas.append({"row": d["row"], "question": d["question"]})
                rollouts.append({k: d[k] for k in
                                 ("token_ids", "logp_student", "logp_teacher", "finish_reason")})
        with open(args.out, "w") as f:
            f.write(render(rollouts, metas, tok, args))
        print(f"wrote {args.out} from {jl}")
        return

    with open(mid) as f:
        blob = json.load(f)
    metas, rollouts = blob["metas"], blob["rollouts"]
    rollouts = teacher_pass(args, rollouts)

    with open(args.out, "w") as f:
        f.write(render(rollouts, metas, tok, args))
    print(f"wrote {args.out}")

    jsonl = os.path.splitext(args.out)[0] + ".jsonl"
    with open(jsonl, "w") as f:
        for r, m in zip(rollouts, metas):
            A = (np.array(r["logp_teacher"]) - np.array(r["logp_student"])).tolist()
            f.write(json.dumps({**m, "finish_reason": r["finish_reason"],
                                "n_tokens": len(r["token_ids"]),
                                "token_ids": r["token_ids"],
                                "logp_student": r["logp_student"],
                                "logp_teacher": r["logp_teacher"], "A": A}) + "\n")
    print(f"wrote {jsonl}")

    for i, r in enumerate(rollouts, 1):
        A = np.array(r["logp_teacher"]) - np.array(r["logp_student"])
        fin = np.isfinite(A)
        print(f"  rollout {i}: mean A={np.nanmean(A):+.3f}  "
              f"A<0 {100*np.mean(A[fin]<0):.1f}%  A<-2 {100*np.mean(A[fin]<-2):.1f}%")


if __name__ == "__main__":
    main()
