#!/usr/bin/env python3
"""
Visualize teacher - student logprob differences per token.
Two color modes, switchable via button in the HTML:

  diff mode:  green = teacher_lp - student_lp > 0 (up to POS_MAX)
              red   = teacher_lp - student_lp < 0 (down to -NEG_MAX)

  ratio mode: green = teacher_prob / student_prob > UPPER_THRESHOLD
              red   = teacher_prob / student_prob < LOWER_THRESHOLD
              white = in between
"""

import json
import math
import os
import re
import html as html_module
from typing import List
from transformers import AutoTokenizer

JSONL_PATH = "sdpo_result/Qwen3-4B/deepmath_diff6to8_verified/teacher_inference/20260417-013031_results/20260417-022234_Qwen3-4B_expert2_scored.jsonl"
OUTPUT_PATH = "sdpo_result/Qwen3-4B/deepmath_diff6to8_verified/teacher_inference/20260417-013031_results/vis/logprob_diff_2.html"
TOKENIZER_NAME = "Qwen/Qwen3-4B"
NUM_SAMPLES = 5       # number of samples to visualize; set to None to include all
SAMPLE_INDICES = None # e.g. [0, 3, 7] to pick specific samples; overrides NUM_SAMPLES if set
SELECT_WRONG_BOXED = False  # if True, auto-select wrong-answer samples (uses math_boxed scorer); overrides both above
MAX_GEN_TOKENS = 14000     # when SELECT_WRONG_BOXED=True, only include samples shorter than this
PROMPT_FILTER = True       # e.g. "Aya goes for a walk" — keep only samples whose user prompt contains this substring

# --- diff mode ---
POS_MAX = 2.0   # diff at which green is fully saturated
NEG_MAX = 5.0   # |diff| at which red is fully saturated

# --- ratio mode ---
UPPER_THRESHOLD = 2.0   # ratio above this → starts turning green
LOWER_THRESHOLD = 0.5   # ratio below this → starts turning red
UPPER_MAX = 8.0         # ratio at which green is fully saturated
LOWER_MIN = 0.125       # ratio at which red is fully saturated


def _load_math_boxed():
    import importlib.util, sys
    from pathlib import Path
    base = Path(__file__).parent.parent / "verl/utils/reward_score"
    spec = importlib.util.spec_from_file_location("math_boxed", base / "math_boxed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_wrong_and_short(record: dict, math_boxed, max_tokens: int) -> bool:
    """True if response has \\boxed{}, is wrong per math_boxed scorer, and is shorter than max_tokens."""
    response = record.get("response", "")
    gt = record.get("ground_truth")
    token_ids = record.get("response_token_ids", [])
    if not gt or not re.search(r'\\boxed\{', response):
        return False
    if len(token_ids) >= max_tokens:
        return False
    res = math_boxed.compute_score(response, gt)
    return not res.get("acc", False)


def diff_to_color(diff: float) -> str:
    """Original mode: map teacher_lp - student_lp to RGB.
    0 = white, +POS_MAX = full green, -NEG_MAX = full red.
    """
    if diff >= 0:
        scale = min(diff / POS_MAX, 1.0)
        r = int(255 * (1 - scale))
        g = 255
        b = int(255 * (1 - scale))
    else:
        scale = min(-diff / NEG_MAX, 1.0)
        r = 255
        g = int(255 * (1 - scale))
        b = int(255 * (1 - scale))
    return f"rgb({r},{g},{b})"


def ratio_to_color(ratio: float) -> str:
    """Ratio mode: map teacher_prob / student_prob to RGB.
    ratio > UPPER_THRESHOLD → green, ratio < LOWER_THRESHOLD → red, else white.
    """
    if ratio >= UPPER_THRESHOLD:
        scale = min((ratio - UPPER_THRESHOLD) / (UPPER_MAX - UPPER_THRESHOLD), 1.0)
        r = int(255 * (1 - scale))
        g = 255
        b = int(255 * (1 - scale))
    elif ratio <= LOWER_THRESHOLD:
        scale = min((LOWER_THRESHOLD - ratio) / (LOWER_THRESHOLD - LOWER_MIN), 1.0)
        r = 255
        g = int(255 * (1 - scale))
        b = int(255 * (1 - scale))
    else:
        return "rgb(255,255,255)"
    return f"rgb({r},{g},{b})"


def safe_incremental_decode(tokenizer, token_ids: List[int]) -> List[str]:
    """Decode each token's visible contribution using full-prefix decode.

    Decoding tokens individually breaks multi-byte sequences (Chinese, emoji,
    math symbols). This approach decodes the growing prefix and diffs successive
    strings. When an incomplete multi-byte sequence produces \ufffd, we emit an
    empty string for that token and defer the character(s) to the completing token.
    """
    pieces = []
    prev_len = 0  # character length of the last clean (no \ufffd) decoded prefix
    for i in range(len(token_ids)):
        cur_text = tokenizer.decode(
            token_ids[: i + 1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        piece = cur_text[prev_len:]
        if "\ufffd" in piece:
            # Incomplete multi-byte sequence — defer; don't advance prev_len
            pieces.append("")
        else:
            pieces.append(piece)
            prev_len = len(cur_text)
    return pieces


def build_sample_html(idx: int, record: dict, tokenizer) -> str:
    token_ids = record["response_token_ids"]
    student_lp = record["response_token_logprobs"]
    teacher_lp = record["teacher_logp_response_token_logprobs"]

    assert len(token_ids) == len(student_lp) == len(teacher_lp), (
        f"Length mismatch: {len(token_ids)}, {len(student_lp)}, {len(teacher_lp)}"
    )

    diffs = [t - s for t, s in zip(teacher_lp, student_lp)]
    ratios = [math.exp(d) for d in diffs]

    min_diff, max_diff = min(diffs), max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    min_ratio, max_ratio = min(ratios), max(ratios)
    mean_ratio = sum(ratios) / len(ratios)

    token_texts = safe_incremental_decode(tokenizer, token_ids)
    prefix_decoded = [""]
    for piece in token_texts:
        prefix_decoded.append(prefix_decoded[-1] + piece)

    topk_per_pos = record.get("teacher_logp_topk_logprobs")  # [[token_id, lp], ...] per position, or None

    spans = []
    for i, (tid, s_lp, t_lp, diff, ratio, text) in enumerate(zip(token_ids, student_lp, teacher_lp, diffs, ratios, token_texts)):
        display = text.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "\u00a0")
        esc = html_module.escape(display) if display else ""
        bg_diff = diff_to_color(diff)
        bg_ratio = ratio_to_color(ratio)

        base = (f"token_id={tid} | student={s_lp:.4f} | teacher={t_lp:.4f} | "
                f"diff={diff:+.4f} | ratio={ratio:.4f}")
        if topk_per_pos and i < len(topk_per_pos) and topk_per_pos[i]:
            topk_lines = []
            for rank, (tk_id, tk_lp) in enumerate(topk_per_pos[i][:10], 1):
                # Decode with prefix context to handle byte-level tokens correctly.
                # If still incomplete UTF-8, extract raw byte value and show as \xNN.
                tk_full = tokenizer.decode(token_ids[:i] + [tk_id], skip_special_tokens=False)
                tk_text = tk_full[len(prefix_decoded[i]):]
                if '\ufffd' in tk_text:
                    raw = tokenizer.convert_ids_to_tokens(tk_id) or ""
                    m = re.match(r'<0x([0-9A-Fa-f]{2})>', raw)
                    tk_text = f'\\x{m.group(1)}' if m else raw
                tk_text = tk_text.replace("\n", "\\n").replace("\t", "\\t")
                marker = " <--" if tk_id == tid else ""
                topk_lines.append(f"  top-{rank}: {repr(tk_text)} {tk_lp:.4f}{marker}")
            base += "\nteacher top-k:\n" + "\n".join(topk_lines)
        tooltip = html_module.escape(base)
        # data-rawdiff/rawratio let JS recompute colors when thresholds change
        spans.append(
            f'<span class="tok" '
            f'data-diff="{bg_diff}" data-ratio="{bg_ratio}" '
            f'data-rawdiff="{diff:.6f}" data-rawratio="{ratio:.6f}" '
            f'style="background:{bg_diff};border-radius:2px;padding:1px 0;white-space:pre;" '
            f'title="{tooltip}">{esc}</span>'
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
      [diff: min={min_diff:+.2f}, max={max_diff:+.2f}, mean={mean_diff:+.2f}]
      &nbsp;
      [ratio: min={min_ratio:.3f}, max={max_ratio:.3f}, mean={mean_ratio:.3f}]
      &nbsp;tokens={len(token_ids)}
    </span>
  </summary>

  <div style="margin:6px 0 4px 0;padding:8px;background:#f8f8f8;border-left:3px solid #aaa;font-size:0.88em;white-space:pre-wrap;">{question_text}</div>

  <div style="margin-bottom:16px;line-height:1.8;font-family:monospace;font-size:0.85em;word-break:normal;overflow-wrap:anywhere;white-space:normal;">{token_html}</div>
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

    def _get_user_prompt(r):
        prompt = r.get("prompt")
        if isinstance(prompt, list):
            for m in prompt:
                if m.get("role") == "user":
                    return m.get("content", "")
        # Fallback for JSONLs where prompt is null: use original_question (or question).
        q = r.get("original_question") or r.get("question")
        return q if isinstance(q, str) else ""

    if PROMPT_FILTER is not None and isinstance(PROMPT_FILTER, str):
        records = [r for r in records if PROMPT_FILTER in _get_user_prompt(r)]
        print(f"After PROMPT_FILTER: {len(records)} records match")

    # Deduplicate: keep only one sample per unique prompt
    seen_prompts = set()
    deduped = []
    for r in records:
        prompt = _get_user_prompt(r)
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            deduped.append(r)
    records = deduped

    if SELECT_WRONG_BOXED:
        print("Filtering: wrong answer + has \\boxed{} + < {MAX_GEN_TOKENS} tokens ...")
        math_boxed = _load_math_boxed()
        records = [r for r in records if _is_wrong_and_short(r, math_boxed, MAX_GEN_TOKENS)]
        if NUM_SAMPLES is not None:
            records = records[:NUM_SAMPLES]
    elif SAMPLE_INDICES is not None:
        records = [records[i] for i in SAMPLE_INDICES if i < len(records)]
    elif NUM_SAMPLES is not None:
        records = records[:NUM_SAMPLES]
    print(f"Processing {len(records)} samples...")

    sample_htmls = []
    for i, rec in enumerate(records):
        print(f"  sample {i} ({len(rec['response_token_ids'])} tokens)...")
        sample_htmls.append(build_sample_html(i, rec, tokenizer))

    legend_diff = f"""
<div id="legend-diff" style="display:flex;gap:8px;align-items:center;font-size:0.85em;flex-wrap:wrap;">
  <strong>Legend (diff mode &mdash; teacher&minus;student logprob):</strong>
  <span style="background:rgb(0,255,0);padding:2px 10px;border-radius:3px;">+{POS_MAX} (teacher&gt;&gt;student)</span>
  <span style="background:rgb(128,255,128);padding:2px 10px;border-radius:3px;">+{POS_MAX/2:.1f}</span>
  <span style="background:rgb(255,255,255);border:1px solid #ccc;padding:2px 10px;border-radius:3px;">0</span>
  <span style="background:rgb(255,128,128);padding:2px 10px;border-radius:3px;">-{NEG_MAX/2:.1f}</span>
  <span style="background:rgb(255,0,0);padding:2px 10px;border-radius:3px;">-{NEG_MAX} (student&gt;&gt;teacher)</span>
  <span style="color:#666;">(hover for details)</span>
</div>"""

    legend_ratio = f"""
<div id="legend-ratio" style="display:none;gap:8px;align-items:center;font-size:0.85em;flex-wrap:wrap;">
  <strong>Legend (ratio mode &mdash; teacher/student prob):</strong>
  <span style="background:rgb(0,255,0);padding:2px 10px;border-radius:3px;">&ge;{UPPER_MAX} (teacher&gt;&gt;student)</span>
  <span style="background:rgb(128,255,128);padding:2px 10px;border-radius:3px;">&gt;{UPPER_THRESHOLD}</span>
  <span style="background:rgb(255,255,255);border:1px solid #ccc;padding:2px 10px;border-radius:3px;">{LOWER_THRESHOLD}&ndash;{UPPER_THRESHOLD} (neutral)</span>
  <span style="background:rgb(255,128,128);padding:2px 10px;border-radius:3px;">&lt;{LOWER_THRESHOLD}</span>
  <span style="background:rgb(255,0,0);padding:2px 10px;border-radius:3px;">&le;{LOWER_MIN} (student&gt;&gt;teacher)</span>
  <span style="color:#666;">(hover for details)</span>
</div>"""

    js = f"""
<script>
var currentMode = 'diff';

// thresholds (initialised from Python constants, editable at runtime)
var POS_MAX = {POS_MAX};
var NEG_MAX = {NEG_MAX};
var UPPER_THRESHOLD = {UPPER_THRESHOLD};
var LOWER_THRESHOLD = {LOWER_THRESHOLD};
var UPPER_MAX = {UPPER_MAX};
var LOWER_MIN = {LOWER_MIN};

function diffToColor(diff) {{
  var r, g, b;
  if (diff >= 0) {{
    var scale = Math.min(diff / POS_MAX, 1.0);
    r = Math.round(255 * (1 - scale)); g = 255; b = Math.round(255 * (1 - scale));
  }} else {{
    var scale = Math.min(-diff / NEG_MAX, 1.0);
    r = 255; g = Math.round(255 * (1 - scale)); b = Math.round(255 * (1 - scale));
  }}
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}}

function ratioToColor(ratio) {{
  if (ratio >= UPPER_THRESHOLD) {{
    var scale = Math.min((ratio - UPPER_THRESHOLD) / (UPPER_MAX - UPPER_THRESHOLD), 1.0);
    var r = Math.round(255 * (1 - scale)), g = 255, b = Math.round(255 * (1 - scale));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }} else if (ratio <= LOWER_THRESHOLD) {{
    var scale = Math.min((LOWER_THRESHOLD - ratio) / (LOWER_THRESHOLD - LOWER_MIN), 1.0);
    var r = 255, g = Math.round(255 * (1 - scale)), b = Math.round(255 * (1 - scale));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }}
  return 'rgb(255,255,255)';
}}

function recolor() {{
  var tokens = document.querySelectorAll('.tok');
  for (var i = 0; i < tokens.length; i++) {{
    var diff  = parseFloat(tokens[i].dataset.rawdiff);
    var ratio = parseFloat(tokens[i].dataset.rawratio);
    tokens[i].dataset.diff  = diffToColor(diff);
    tokens[i].dataset.ratio = ratioToColor(ratio);
    if (currentMode === 'diff')  tokens[i].style.background = tokens[i].dataset.diff;
    if (currentMode === 'ratio') tokens[i].style.background = tokens[i].dataset.ratio;
  }}
}}

function applyThresholds() {{
  var posMax = parseFloat(document.getElementById('in-pos-max').value);
  var negMax = parseFloat(document.getElementById('in-neg-max').value);
  var upper  = parseFloat(document.getElementById('in-upper').value);
  var lower  = parseFloat(document.getElementById('in-lower').value);
  var upperMax = parseFloat(document.getElementById('in-upper-max').value);
  var lowerMin = parseFloat(document.getElementById('in-lower-min').value);
  if ([posMax,negMax,upper,lower,upperMax,lowerMin].some(isNaN)) {{ alert('Invalid number'); return; }}
  POS_MAX = posMax; NEG_MAX = negMax;
  UPPER_THRESHOLD = upper; LOWER_THRESHOLD = lower;
  UPPER_MAX = upperMax; LOWER_MIN = lowerMin;
  recolor();
}}

function switchMode(mode) {{
  currentMode = mode;
  var tokens = document.querySelectorAll('.tok');
  for (var i = 0; i < tokens.length; i++) {{
    tokens[i].style.background = tokens[i].dataset[mode];
  }}
  document.getElementById('legend-diff').style.display  = (mode === 'diff')  ? 'flex' : 'none';
  document.getElementById('legend-ratio').style.display = (mode === 'ratio') ? 'flex' : 'none';
  document.getElementById('ctrl-diff').style.display    = (mode === 'diff')  ? 'flex' : 'none';
  document.getElementById('ctrl-ratio').style.display   = (mode === 'ratio') ? 'flex' : 'none';
  document.getElementById('btn-diff').style.fontWeight  = (mode === 'diff')  ? 'bold' : 'normal';
  document.getElementById('btn-ratio').style.fontWeight = (mode === 'ratio') ? 'bold' : 'normal';
}}
</script>
"""

    teacher_model = html_module.escape(records[0].get('teacher_logp_model', '?')) if records else '?'

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Teacher vs Student LogProb Visualization</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #222; }}
    h1 {{ font-size: 1.3em; margin-bottom: 4px; }}
    summary::-webkit-details-marker {{ color: #555; }}
    details {{ margin-bottom: 4px; }}
    span[title]:hover {{ outline: 1px solid #666; cursor: default; }}
    .mode-btn {{
      padding: 4px 14px; border: 1px solid #999; border-radius: 4px;
      background: #f0f0f0; cursor: pointer; font-size: 0.9em;
    }}
    .mode-btn:hover {{ background: #e0e0e0; }}
  </style>
</head>
<body>
  <h1>Teacher &minus; Student LogProb per Token</h1>
  <p style="font-size:0.85em;color:#555;">
    Teacher model: <code>{teacher_model}</code> &nbsp;|&nbsp; {len(records)} samples
  </p>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <span style="font-size:0.9em;font-weight:bold;">Color mode:</span>
    <button id="btn-diff"  class="mode-btn" style="font-weight:bold;" onclick="switchMode('diff')">Diff (logprob)</button>
    <button id="btn-ratio" class="mode-btn" onclick="switchMode('ratio')">Ratio (prob)</button>
  </div>
  <!-- threshold controls -->
  <div id="ctrl-diff" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px;font-size:0.85em;">
    <strong>Diff thresholds:</strong>
    <label>pos_max <input id="in-pos-max" type="number" step="0.1" value="{POS_MAX}" style="width:60px;"></label>
    <label>neg_max <input id="in-neg-max" type="number" step="0.1" value="{NEG_MAX}" style="width:60px;"></label>
    <button class="mode-btn" onclick="applyThresholds()">Apply</button>
  </div>
  <div id="ctrl-ratio" style="display:none;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px;font-size:0.85em;">
    <strong>Ratio thresholds:</strong>
    <label>upper <input id="in-upper" type="number" step="0.1" value="{UPPER_THRESHOLD}" style="width:60px;"></label>
    <label>lower <input id="in-lower" type="number" step="0.1" value="{LOWER_THRESHOLD}" style="width:60px;"></label>
    <label>upper_max <input id="in-upper-max" type="number" step="0.5" value="{UPPER_MAX}" style="width:60px;"></label>
    <label>lower_min <input id="in-lower-min" type="number" step="0.01" value="{LOWER_MIN}" style="width:60px;"></label>
    <button class="mode-btn" onclick="applyThresholds()">Apply</button>
  </div>
  <div style="margin-bottom:16px;">
    {legend_diff}
    {legend_ratio}
  </div>
  {"".join(sample_htmls)}
  {js}
</body>
</html>
"""

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"\nDone! Visualization saved to:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
