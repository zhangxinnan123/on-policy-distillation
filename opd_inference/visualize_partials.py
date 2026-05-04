"""Render a (bundle, continuations) pair as a single self-contained HTML page.

Inputs
------
--bundle         Path to the JSON written by `prepare_partials_file.py`
                 (contains: prompt, original_response, partials[]).
--continuations  Path to the JSONL written by `run_continue_from_partial.py`
                 (one record per partial: prompt, partial, completions[]).

Output
------
A single HTML file with:
  - the prompt
  - the original response, split inline at every partial-cutoff char position
  - at each cutoff, an inline <details> button that toggles open to show
    the model's continuation(s) generated from that partial.

The HTML uses no external assets — open it directly in any browser.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_continuations(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group continuation JSONL records by tag → list of records (one per row)."""
    by_tag: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tag = rec.get("tag", "?")
            by_tag.setdefault(tag, []).append(rec)
    return by_tag


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1080px; margin: 24px auto; padding: 0 18px; line-height: 1.5; color: #1f1f23; }
h1 { font-size: 18px; margin: 0 0 12px; }
h2 { font-size: 14px; margin: 26px 0 8px; border-bottom: 1px solid #e0e0e3; padding-bottom: 4px;
     color: #444; text-transform: uppercase; letter-spacing: 0.5px; }
.meta { background: #f5f5f7; border: 1px solid #e0e0e3; padding: 10px 14px; border-radius: 6px;
        font-size: 12.5px; }
.meta b { color: #555; }
pre.block { white-space: pre-wrap; word-wrap: break-word; font-family: ui-monospace, SFMono-Regular,
            Menlo, monospace; font-size: 12.5px; background: #fafafa; padding: 12px 14px;
            border-left: 3px solid #c0c0c8; border-radius: 4px; margin: 0; }
.response { white-space: pre-wrap; word-wrap: break-word; font-family: ui-monospace, SFMono-Regular,
            Menlo, monospace; font-size: 12.5px; background: #fcfcfd; padding: 14px 16px;
            border: 1px solid #ececef; border-radius: 6px; }
details.cutoff { display: inline; }
details.cutoff > summary { cursor: pointer; display: inline-block; padding: 1px 8px; margin: 2px 4px;
                           background: #eaf2ff; border: 1px solid #b8cdee; border-radius: 4px;
                           font-size: 11.5px; color: #14467a; user-select: none;
                           font-family: -apple-system, sans-serif; }
details.cutoff[open] > summary { background: #cfe0ff; }
details.cutoff .body { display: block; margin: 8px 0 12px; padding: 10px 12px;
                       background: #f3f7fc; border-left: 3px solid #4d80c0; border-radius: 4px; }
.cont-label { color: #666; font-size: 11.5px; margin: 0 0 6px; }
.cont-text { white-space: pre-wrap; word-wrap: break-word; font-family: ui-monospace, monospace;
             font-size: 12.5px; background: #ffffff; padding: 10px 12px; border-radius: 4px;
             border: 1px solid #dde4ec; }
.sample-divider { color: #888; font-size: 11px; margin: 10px 0 4px; }
.warn { color: #a04020; background: #fff4e8; border: 1px solid #f0c8a0; padding: 8px 10px;
        border-radius: 4px; font-size: 12.5px; margin: 8px 0; }
.controls { margin: 6px 0 12px; font-size: 12px; color: #555; }
.controls button { padding: 3px 10px; margin-right: 6px; font-size: 12px; cursor: pointer;
                   background: #fff; border: 1px solid #c0c0c8; border-radius: 4px; }
"""

JS = """
function toggleAll(open) {
  document.querySelectorAll('details.cutoff').forEach(d => d.open = open);
}
"""


def render(bundle: Dict[str, Any], by_tag: Dict[str, List[Dict[str, Any]]],
           title: str) -> str:
    prompt = bundle.get("prompt", "")
    original = bundle.get("original_response", "")
    partials = bundle.get("partials", [])

    # Sort by where the partial ends in the original response (= len(partial)).
    # Drop the empty / start-of-response partial since it has no inline anchor.
    indexed = []
    warnings: List[str] = []
    for entry in partials:
        ptext = entry.get("partial", "") or ""
        cutoff_char = len(ptext)
        if cutoff_char == 0:
            # Render the start-of-response continuation as a leading button.
            indexed.append({"cutoff_char": 0, "entry": entry, "leading": True})
            continue
        if not original.startswith(ptext):
            warnings.append(
                f"partial for tag={entry.get('tag')!r} is not a strict prefix of "
                f"original_response (len={cutoff_char}); rendering may be off."
            )
        indexed.append({"cutoff_char": cutoff_char, "entry": entry, "leading": False})
    indexed.sort(key=lambda x: x["cutoff_char"])

    parts: List[str] = ['<!doctype html><html lang="en"><head><meta charset="utf-8">']
    parts.append(f"<title>{html_lib.escape(title)}</title>")
    parts.append("<style>" + CSS + "</style>")
    parts.append("<script>" + JS + "</script>")
    parts.append("</head><body>")
    parts.append(f"<h1>{html_lib.escape(title)}</h1>")

    # Metadata block.
    cont_models = sorted({recs[0].get("model", "") for recs in by_tag.values() if recs})
    sp = next(iter(by_tag.values()), [{}])[0].get("sampling_params", {}) if by_tag else {}
    parts.append('<div class="meta">')
    parts.append(f"<b>source:</b> {html_lib.escape(str(bundle.get('source_jsonl', '')))}<br>")
    parts.append(
        f"<b>row_idx:</b> {bundle.get('source_row_idx')}"
        f" &nbsp; <b>tokenizer:</b> {html_lib.escape(str(bundle.get('tokenizer', '')))}"
        f" &nbsp; <b>orig_tokens:</b> {bundle.get('original_response_token_count')}<br>"
    )
    if cont_models:
        parts.append(f"<b>continuation model(s):</b> {html_lib.escape(', '.join(cont_models))}<br>")
    if sp:
        parts.append(
            f"<b>sampling:</b> T={sp.get('temperature')} top_p={sp.get('top_p')}"
            f" n={sp.get('n')} max_new_tokens={sp.get('max_new_tokens')}"
        )
    parts.append("</div>")

    if warnings:
        parts.append('<div class="warn"><b>warnings:</b><ul>')
        for w in warnings:
            parts.append(f"<li>{html_lib.escape(w)}</li>")
        parts.append("</ul></div>")

    parts.append("<h2>Prompt</h2>")
    parts.append(f'<pre class="block">{html_lib.escape(prompt)}</pre>')

    parts.append("<h2>Original response (click a button to reveal a continuation)</h2>")
    parts.append('<div class="controls">')
    parts.append('<button onclick="toggleAll(true)">expand all</button>')
    parts.append('<button onclick="toggleAll(false)">collapse all</button>')
    parts.append("</div>")
    parts.append('<div class="response">')

    prev_char = 0
    for item in indexed:
        cutoff_char = item["cutoff_char"]
        entry = item["entry"]
        if cutoff_char > prev_char:
            parts.append(html_lib.escape(original[prev_char:cutoff_char]))
            prev_char = cutoff_char
        parts.append(_render_button(entry, by_tag))
    if prev_char < len(original):
        parts.append(html_lib.escape(original[prev_char:]))

    parts.append("</div>")
    parts.append("</body></html>")
    return "".join(parts)


def _render_button(entry: Dict[str, Any], by_tag: Dict[str, List[Dict[str, Any]]]) -> str:
    tag = entry.get("tag", "?")
    cutoff_tokens = entry.get("cutoff_tokens")
    source = entry.get("source") or "?"
    recs = by_tag.get(tag, [])
    n_samples = sum(len(r.get("completions", [])) for r in recs)
    summary = (f"▶ continuation @ {tag} "
               f"(tokens={cutoff_tokens}, source={source}, samples={n_samples})")
    out: List[str] = []
    out.append('<details class="cutoff">')
    out.append(f"<summary>{html_lib.escape(summary)}</summary>")
    out.append('<div class="body">')
    if not recs:
        out.append('<div class="cont-label"><i>no continuation record found for this tag</i></div>')
    else:
        for ri, rec in enumerate(recs):
            comps = rec.get("completions", [])
            model = rec.get("model", "?")
            for ci, comp in enumerate(comps):
                label_bits = [f"model={model}"]
                if len(recs) > 1:
                    label_bits.append(f"record {ri}")
                if len(comps) > 1:
                    label_bits.append(f"sample {ci}")
                finish = (comp.get("vllm_extra") or {}).get("finish_reason")
                if finish:
                    label_bits.append(f"finish={finish}")
                out.append(f'<div class="cont-label">{html_lib.escape(" · ".join(label_bits))}</div>')
                out.append(f'<pre class="cont-text">{html_lib.escape(comp.get("continuation", ""))}</pre>')
                if ci < len(comps) - 1:
                    out.append('<div class="sample-divider">— next sample —</div>')
    out.append("</div></details>")
    return "".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="HTML viewer for prompt + original response with inline continuation buttons.")
    p.add_argument("--bundle", required=True)
    p.add_argument("--continuations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Partial-continuation viewer")
    args = p.parse_args()

    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    by_tag = _load_continuations(args.continuations)
    html_doc = render(bundle, by_tag, args.title)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
