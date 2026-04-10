#!/usr/bin/env python3
"""
Visualize teacher - student logprob differences per token.
Green = teacher > student (teacher more confident), Red = teacher < student.
"""

import json
import math
import html as html_module
from transformers import AutoTokenizer

JSONL_PATH = "/projects/standard/mhong/zhan9359/work/on-policy-distillation/verl/lzy_eval_result/teacher_inference/20260408-195742_results/20260408-201149_scored.jsonl"
OUTPUT_PATH = "/projects/standard/mhong/zhan9359/work/on-policy-distillation/verl/lzy_eval_result/teacher_inference/20260408-195742_results/logprob_diff_viz.html"
TOKENIZER_NAME = "Qwen/Qwen3-4B"
# Asymmetric color scale: 0=white, +POS_MAX=green, -NEG_MAX=red
POS_MAX = 2.0
NEG_MAX = 5.0


def diff_to_color(diff: float) -> str:
    """Map diff (teacher - student) to an RGB background color.
    0 = pure white, +2 = full green, -5 = full red. Asymmetric scale.
    """
    if diff >= 0:
        ratio = min(diff / POS_MAX, 1.0)  # 0..1
        # white -> green
        r = int(255 * (1 - ratio))
        g = 255
        b = int(255 * (1 - ratio))
    else:
        ratio = min(-diff / NEG_MAX, 1.0)  # 0..1
        # white -> red
        r = 255
        g = int(255 * (1 - ratio))
        b = int(255 * (1 - ratio))

    return f"rgb({r},{g},{b})"


def build_sample_html(idx: int, record: dict, tokenizer) -> str:
    token_ids = record["response_token_ids"]
    student_lp = record["response_token_logprobs"]
    teacher_lp = record["teacher_logp_response_token_logprobs"]

    assert len(token_ids) == len(student_lp) == len(teacher_lp), (
        f"Length mismatch: {len(token_ids)}, {len(student_lp)}, {len(teacher_lp)}"
    )

    diffs = [t - s for t, s in zip(teacher_lp, student_lp)]
    min_diff = min(diffs)
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)

    # Build token spans
    spans = []
    for tid, s_lp, t_lp, diff in zip(token_ids, student_lp, teacher_lp, diffs):
        text = tokenizer.decode([tid])
        # Show newlines/tabs as literal escape sequences (no actual line break)
        display = text.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "\u00a0")
        if not display:
            display = f"[{tid}]"
        esc = html_module.escape(display)
        bg = diff_to_color(diff)
        tooltip = (
            f"token_id={tid} | "
            f"student={s_lp:.4f} | "
            f"teacher={t_lp:.4f} | "
            f"diff={diff:+.4f}"
        )
        spans.append(
            f'<span style="background:{bg};border-radius:2px;padding:1px 0;" '
            f'title="{html_module.escape(tooltip)}">{esc}</span>'
        )

    token_html = "".join(spans)

    question_text = ""
    if record.get("prompt"):
        for msg in record["prompt"]:
            if msg.get("role") == "user":
                question_text = html_module.escape(msg["content"])
                break

    correct = record.get("ground_truth", "?")
    source = record.get("data_source", "?")

    return f"""
<details open>
  <summary style="cursor:pointer;font-size:1.05em;font-weight:bold;padding:6px 0;">
    Sample {idx} &mdash; source: {html_module.escape(str(source))} &mdash; answer: {html_module.escape(str(correct))}
    &nbsp;<span style="font-weight:normal;font-size:0.9em;">
      [diff: min={min_diff:+.2f}, max={max_diff:+.2f}, mean={mean_diff:+.2f}, tokens={len(token_ids)}]
    </span>
  </summary>

  <div style="margin:6px 0 4px 0;padding:8px;background:#f8f8f8;border-left:3px solid #aaa;font-size:0.88em;white-space:pre-wrap;">{question_text}</div>

  <div style="margin-bottom:16px;line-height:1.8;font-family:monospace;font-size:0.85em;word-break:break-all;">{token_html}</div>
</details>
<hr style="border:none;border-top:1px solid #ddd;margin:8px 0;">
"""


def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Reading: {JSONL_PATH}")
    records = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Processing {len(records)} samples...")

    sample_htmls = []
    for i, rec in enumerate(records):
        print(f"  sample {i} ({len(rec['response_token_ids'])} tokens)...")
        sample_htmls.append(build_sample_html(i, rec, tokenizer))

    legend_html = """
<div style="display:flex;gap:8px;align-items:center;margin-bottom:16px;font-size:0.85em;flex-wrap:wrap;">
  <strong>Legend (teacher &minus; student logprob):</strong>
  <span style="background:rgb(0,255,0);padding:2px 10px;border-radius:3px;">+2 (teacher&gt;&gt;student)</span>
  <span style="background:rgb(128,255,128);padding:2px 10px;border-radius:3px;">+1</span>
  <span style="background:rgb(255,255,255);border:1px solid #ccc;padding:2px 10px;border-radius:3px;">0</span>
  <span style="background:rgb(255,179,179);padding:2px 10px;border-radius:3px;">-2.5</span>
  <span style="background:rgb(255,0,0);padding:2px 10px;border-radius:3px;">-5 (teacher&lt;&lt;student)</span>
  <span style="color:#666;">(hover token for details)</span>
</div>
"""

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Teacher vs Student LogProb Diff Visualization</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #222; }}
    h1 {{ font-size: 1.3em; margin-bottom: 4px; }}
    summary::-webkit-details-marker {{ color: #555; }}
    details {{ margin-bottom: 4px; }}
    span[title]:hover {{ outline: 1px solid #666; cursor: default; }}
  </style>
</head>
<body>
  <h1>Teacher &minus; Student LogProb per Token</h1>
  <p style="font-size:0.85em;color:#555;">
    Teacher model: <code>{html_module.escape(records[0].get('teacher_logp_model','?'))}</code> &nbsp;|&nbsp;
    Color range: +{POS_MAX}/&minus;{NEG_MAX} nats &nbsp;|&nbsp;
    {len(records)} samples
  </p>
  {legend_html}
  {"".join(sample_htmls)}
</body>
</html>
"""

    with open(OUTPUT_PATH, "w") as f:
        f.write(page)

    print(f"\nDone! Visualization saved to:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
