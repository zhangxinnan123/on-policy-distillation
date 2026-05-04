"""Build a partials bundle file from a scored teacher-inference JSONL row.

Given a row from a JSONL like
`lzy_eval_result/teacher_inference/.../<run>_scored.jsonl`,
this script extracts the user prompt and produces several partial assistant
responses by truncating the recorded `response_token_ids` at the requested
token cutoffs.

Output is a single JSON file (the "bundle") consumed by
`run_continue_from_partial.py --input_file`.

Examples:
    # 1) List rows by recorded response length (longest first) to pick a row.
    python prepare_partials_file.py \\
        --source_jsonl path/to/20260419-220251_Qwen3-4B_scored.jsonl \\
        --list --list_top 20

    # 2) Build a bundle with explicit token cutoffs.
    python prepare_partials_file.py \\
        --source_jsonl path/to/20260419-220251_Qwen3-4B_scored.jsonl \\
        --row_idx 7 \\
        --tokenizer Qwen/Qwen3-4B \\
        --truncate_tokens "0,200,1000,4000,10000" \\
        --out continue_inputs/eight_circles_bundle.json

    # 3) Pick the longest row automatically and place cutoffs at fractions.
    python prepare_partials_file.py \\
        --source_jsonl path/to/20260419-220251_Qwen3-4B_scored.jsonl \\
        --pick_longest \\
        --tokenizer Qwen/Qwen3-4B \\
        --truncate_fracs "0,0.1,0.25,0.5,0.75,0.9" \\
        --out continue_inputs/longest_bundle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_prompt(prompt_field: Any) -> str:
    if isinstance(prompt_field, str):
        return prompt_field
    if isinstance(prompt_field, list):
        for m in prompt_field:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content
    raise ValueError("source row 'prompt' is not a string or messages list with a user role")


def _read_row(path: str, row_idx: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == row_idx:
                return json.loads(line)
    raise IndexError(f"row_idx {row_idx} not found in {path}")


def _scan_lengths(path: str) -> List[Dict[str, Any]]:
    """Return [{row_idx, n_tokens, n_chars, prompt_preview}] for every row.

    Streams the JSONL so we don't materialize the whole file (rows can hold
    massive token-id arrays).
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"row_idx": idx, "n_tokens": -1, "n_chars": -1, "prompt_preview": "<malformed>"})
                continue
            tids = rec.get("response_token_ids")
            n_tokens = len(tids) if isinstance(tids, list) else -1
            resp = rec.get("response")
            n_chars = len(resp) if isinstance(resp, str) else -1
            prompt_field = rec.get("prompt")
            try:
                prompt_text = _resolve_prompt(prompt_field)
            except ValueError:
                prompt_text = ""
            preview = prompt_text.replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."
            rows.append({
                "row_idx": idx,
                "n_tokens": n_tokens,
                "n_chars": n_chars,
                "prompt_preview": preview,
            })
    return rows


def _find_paragraph_boundaries(recorded_response: str) -> List[int]:
    """Return char offsets where each '\\n\\n' ends (i.e. position right after the boundary).

    Non-overlapping scan; an isolated triple '\\n\\n\\n' yields a single boundary
    just past the first '\\n\\n'.
    """
    boundaries: List[int] = []
    start = 0
    while True:
        i = recorded_response.find("\n\n", start)
        if i < 0:
            break
        boundaries.append(i + 2)
        start = i + 2
    return boundaries


def _token_cutoff_for_char(token_ids: List[int], tokenizer, target_char_len: int) -> int:
    """Smallest k such that len(decode(token_ids[:k])) >= target_char_len.

    Binary search over prefix length. Decodes the prefix at each step; for
    typical scored rows (≤ ~16k tokens) and ~50–500 paragraph breaks this is
    fast enough.
    """
    n = len(token_ids)
    if n == 0 or target_char_len <= 0:
        return 0
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        text = tokenizer.decode(token_ids[:mid], skip_special_tokens=False)
        if len(text) >= target_char_len:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _resolve_paragraph_indices(spec: str, total: int) -> List[int]:
    """Parse '--truncate_paragraphs' / 'all' into a list of 1-based indices."""
    spec = spec.strip()
    if spec.lower() == "all":
        return list(range(1, total + 1))
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            i = int(tok)
        except ValueError:
            raise SystemExit(f"error: --truncate_paragraphs entry '{tok}' is not an integer or 'all'")
        out.append(i)
    return out


def _print_length_table(stats: List[Dict[str, Any]], top: int) -> None:
    ranked = sorted(stats, key=lambda r: r["n_tokens"], reverse=True)
    head = ranked[:top] if top > 0 else ranked
    print(f"{'row_idx':>8} {'n_tokens':>10} {'n_chars':>10}  prompt")
    print("-" * 80)
    for r in head:
        print(f"{r['row_idx']:>8} {r['n_tokens']:>10} {r['n_chars']:>10}  {r['prompt_preview']}")
    if top > 0 and len(ranked) > top:
        print(f"... ({len(ranked) - top} more rows; pass --list_top 0 to see all)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a partials bundle from a scored JSONL row.")
    p.add_argument("--source_jsonl", required=True,
                   help="Path to the scored JSONL produced by teacher inference.")
    p.add_argument("--row_idx", type=int, default=None,
                   help="0-based row index to use. Mutually exclusive with --pick_longest.")
    p.add_argument("--pick_longest", action="store_true",
                   help="Auto-select the row with the largest response_token_ids length.")
    p.add_argument("--list", dest="list_mode", action="store_true",
                   help="List rows sorted by recorded response length (longest first), then exit.")
    p.add_argument("--list_top", type=int, default=20,
                   help="When using --list, show this many top rows (0 = all).")
    p.add_argument("--min_tokens", type=int, default=0,
                   help="Skip rows whose recorded response has fewer than this many tokens "
                        "(applies to --pick_longest and --list).")
    p.add_argument("--tokenizer",
                   help="HF tokenizer id or local path matching the model that produced "
                        "the recorded response_token_ids (e.g. Qwen/Qwen3-4B). Required when "
                        "building a bundle (i.e. not --list).")
    p.add_argument("--truncate_tokens",
                   help="Comma-separated token-count cutoffs. 0 = empty assistant turn. "
                        "Example: '0,200,1000,4000,10000'.")
    p.add_argument("--truncate_fracs",
                   help="Comma-separated fractional cutoffs in [0, 1] applied to the recorded "
                        "response token length, e.g. '0,0.1,0.25,0.5,0.75'. Useful for very "
                        "long responses. Combined with --truncate_tokens if both are given.")
    p.add_argument("--truncate_paragraphs",
                   help="Comma-separated 1-based indices of '\\n\\n' boundaries in the recorded "
                        "response (or 'all' for every boundary). For each, the cutoff is the "
                        "smallest token count k such that decoding token_ids[:k] covers that "
                        "boundary. Combined with --truncate_tokens / --truncate_fracs.")
    p.add_argument("--list_paragraphs", action="store_true",
                   help="Print the '\\n\\n' boundaries (and their token cutoffs) for the chosen "
                        "row, then exit. Use to pick indices for --truncate_paragraphs.")
    p.add_argument("--out", help="Output bundle JSON file path. Required unless --list/--list_paragraphs.")
    args = p.parse_args()

    if args.list_mode:
        stats = _scan_lengths(args.source_jsonl)
        if args.min_tokens > 0:
            stats = [s for s in stats if s["n_tokens"] >= args.min_tokens]
        _print_length_table(stats, args.list_top)
        return

    if args.tokenizer is None:
        sys.exit("error: --tokenizer is required when building a bundle")
    if args.out is None:
        sys.exit("error: --out is required when building a bundle")
    if (args.row_idx is None) == (not args.pick_longest):
        sys.exit("error: pass exactly one of --row_idx or --pick_longest")

    if args.pick_longest:
        stats = _scan_lengths(args.source_jsonl)
        if args.min_tokens > 0:
            stats = [s for s in stats if s["n_tokens"] >= args.min_tokens]
        if not stats:
            sys.exit("error: no rows matched --min_tokens filter")
        chosen = max(stats, key=lambda r: r["n_tokens"])
        row_idx = chosen["row_idx"]
        print(f"[INFO] --pick_longest selected row_idx={row_idx} "
              f"(n_tokens={chosen['n_tokens']}, n_chars={chosen['n_chars']})")
        print(f"       prompt: {chosen['prompt_preview']}")
    else:
        row_idx = args.row_idx

    row = _read_row(args.source_jsonl, row_idx)
    prompt = _resolve_prompt(row.get("prompt"))
    token_ids: Optional[List[int]] = row.get("response_token_ids")
    recorded_response: str = row.get("response") or ""
    if not isinstance(token_ids, list):
        sys.exit("error: source row missing 'response_token_ids' list")

    # Lazy-import transformers so this script works in lighter envs.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    paragraph_boundaries = _find_paragraph_boundaries(recorded_response)

    if args.list_paragraphs:
        print(f"[INFO] row_idx={row_idx} has {len(paragraph_boundaries)} '\\n\\n' boundaries")
        print(f"{'p_idx':>5} {'char_end':>10} {'tok_cut':>8}  preview (last ~40 chars before boundary)")
        print("-" * 90)
        for p_idx, char_end in enumerate(paragraph_boundaries, 1):
            k = _token_cutoff_for_char(token_ids, tokenizer, char_end)
            preview = recorded_response[max(0, char_end - 42):char_end - 2].replace("\n", "\\n")
            print(f"{p_idx:>5} {char_end:>10} {k:>8}  ...{preview}")
        return

    # Build cutoff specs as (k, tag) tuples; paragraph cutoffs preserve their index in the tag.
    raw_specs: List[Dict[str, Any]] = []
    if args.truncate_tokens is not None:
        for tok in args.truncate_tokens.split(","):
            tok = tok.strip()
            if tok == "":
                continue
            try:
                k = int(tok)
            except ValueError:
                sys.exit(f"error: --truncate_tokens entry '{tok}' is not an integer")
            raw_specs.append({"k": k, "tag": f"tok{k}", "source": "tok"})
    if args.truncate_fracs is not None:
        for tok in args.truncate_fracs.split(","):
            tok = tok.strip()
            if tok == "":
                continue
            try:
                f = float(tok)
            except ValueError:
                sys.exit(f"error: --truncate_fracs entry '{tok}' is not a number")
            if not (0.0 <= f <= 1.0):
                sys.exit(f"error: --truncate_fracs entry {f} must be in [0, 1]")
            k = int(round(f * len(token_ids)))
            raw_specs.append({"k": k, "tag": f"tok{k}", "source": "frac"})
    if args.truncate_paragraphs is not None:
        if not paragraph_boundaries:
            print("[WARN] no '\\n\\n' boundaries in recorded response; ignoring --truncate_paragraphs.")
        else:
            indices = _resolve_paragraph_indices(args.truncate_paragraphs, len(paragraph_boundaries))
            for p_idx in indices:
                if p_idx < 1 or p_idx > len(paragraph_boundaries):
                    print(f"[WARN] paragraph index {p_idx} out of range "
                          f"[1, {len(paragraph_boundaries)}]; skipping.")
                    continue
                k = _token_cutoff_for_char(token_ids, tokenizer, paragraph_boundaries[p_idx - 1])
                raw_specs.append({"k": k, "tag": f"para{p_idx}", "source": "paragraph"})

    if not raw_specs:
        sys.exit("error: provide --truncate_tokens, --truncate_fracs, and/or --truncate_paragraphs")
    if any(s["k"] < 0 for s in raw_specs):
        sys.exit("error: cutoff values must be >= 0")

    # Sort by k, dedupe by k (keep the first tag — paragraph tags will appear before
    # equivalent tok tags only if they were listed first; otherwise dedupe by value).
    raw_specs.sort(key=lambda s: s["k"])
    seen_k = set()
    cutoff_specs: List[Dict[str, Any]] = []
    for s in raw_specs:
        if s["k"] in seen_k:
            continue
        seen_k.add(s["k"])
        cutoff_specs.append(s)

    partials_out: List[Dict[str, Any]] = []
    for spec in cutoff_specs:
        k = spec["k"]
        clamped = min(k, len(token_ids))
        if clamped != k:
            print(f"[WARN] cutoff {k} > recorded length {len(token_ids)}; clamping.")

        if clamped == 0:
            partial_text = ""
        else:
            partial_text = tokenizer.decode(token_ids[:clamped], skip_special_tokens=False)

        partials_out.append({
            "tag": spec["tag"],
            "source": spec["source"],
            "cutoff_tokens": clamped,
            "partial": partial_text,
            "remaining_tokens": len(token_ids) - clamped,
        })

    bundle: Dict[str, Any] = {
        "source_jsonl": args.source_jsonl,
        "source_row_idx": row_idx,
        "tokenizer": args.tokenizer,
        "data_source": row.get("data_source"),
        "ground_truth": row.get("ground_truth"),
        "original_response_token_count": len(token_ids),
        "original_response": recorded_response,
        "prompt": prompt,
        "partials": partials_out,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote bundle: {out_path}")
    print(f"     prompt length: {len(prompt)} chars")
    print(f"     partials: {len(partials_out)} (cutoffs: "
          f"{[p['cutoff_tokens'] for p in partials_out]})")
    print(f"     recorded response length: {len(token_ids)} tokens, "
          f"{len(recorded_response)} chars")


if __name__ == "__main__":
    main()
