#!/usr/bin/env python3
"""Validate the FORMAT of generated responses against the expected Qwen3 template.

Two questions this answers:

  Test 1 (--mode nonthink): a BASE-model student, rolled out under enable_thinking=False.
      The empty <think></think> block lives in the PROMPT (prefilled by the chat
      template), so a conformant response contains NO <think>/</think> tags — it is a
      direct answer. This checks the base model actually follows the non-think frame
      (doesn't spawn its own <think>, doesn't run on, produces a non-empty answer).

  Test 2 (--mode think): an SFT'd student expected to reason.
      A conformant response STARTS with a well-formed <think> ... </think> block
      (non-empty reasoning), followed by a non-empty answer.

Input can be inline (--response ...) or your generation files from
run_generate_hf.py / run_generate_parquet.py (JSONL or parquet with a `response`
column). Reports a per-file breakdown and a pass/fail against --mode.

Run on the cluster:
    ssh sfm-science-sfm-p5-cluster "cd /fsx/xinnanzh/on-policy-distillation && \
        python opd_inference/check_response_format.py \
            --responses-file /path/to/gen.jsonl --mode nonthink --show-examples 5"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Non-greedy, DOTALL: match a single reasoning block.
BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def classify(response: str) -> dict:
    """Classify a raw generated response string.

    Returns a dict with keys:
        format:  'think' | 'nonthink' | 'malformed'
        reason:  short human string
        n_open, n_close: tag counts
        think_len: chars of reasoning inside the first block (stripped)
        answer_len: chars of answer text (stripped)
    """
    r = response or ""
    n_open = r.count(THINK_OPEN)
    n_close = r.count(THINK_CLOSE)

    # No think tags at all -> non-think candidate.
    if n_open == 0 and n_close == 0:
        answer = r.strip()
        if not answer:
            return dict(format="malformed", reason="empty response", n_open=0, n_close=0,
                        think_len=0, answer_len=0)
        return dict(format="nonthink", reason="direct answer, no think tags", n_open=0, n_close=0,
                    think_len=0, answer_len=len(answer))

    # Unbalanced tags -> malformed (e.g. truncated mid-thinking, or stray tag).
    if n_open != n_close or n_open != 1:
        return dict(format="malformed",
                    reason=f"unbalanced/multiple think tags (open={n_open}, close={n_close})",
                    n_open=n_open, n_close=n_close, think_len=0, answer_len=0)

    m = BLOCK_RE.search(r)
    if not m:
        return dict(format="malformed", reason="tags present but no well-formed block",
                    n_open=n_open, n_close=n_close, think_len=0, answer_len=0)

    think_content = m.group(1).strip()
    answer = r[m.end():].strip()

    # An empty <think></think> followed by an answer == the non-think frame
    # (this is exactly what the template prefills when enable_thinking=False).
    if not think_content:
        if not answer:
            return dict(format="malformed", reason="empty think block and empty answer",
                        n_open=n_open, n_close=n_close, think_len=0, answer_len=0)
        return dict(format="nonthink", reason="empty <think></think> + answer",
                    n_open=n_open, n_close=n_close, think_len=0, answer_len=len(answer))

    if not answer:
        return dict(format="malformed", reason="reasoning present but no answer after </think>",
                    n_open=n_open, n_close=n_close, think_len=len(think_content), answer_len=0)

    # Well-formed reasoning + answer.
    return dict(format="think", reason="well-formed <think>...</think> + answer",
                n_open=n_open, n_close=n_close, think_len=len(think_content), answer_len=len(answer))


def iter_responses(path: str, field: str):
    """Yield (idx, response_str) from a JSONL or parquet file."""
    if path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(path)
        if field not in df.columns:
            raise SystemExit(f"Column {field!r} not in parquet. Columns: {list(df.columns)}")
        for i, v in enumerate(df[field].tolist()):
            yield i, v if isinstance(v, str) else str(v)
    else:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield i, rec.get(field, "") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--response", help="A single response string to classify.")
    src.add_argument("--responses-file", help="JSONL or parquet file of generations.")
    ap.add_argument("--response-field", default="response", help="Field/column holding the response text.")
    ap.add_argument("--mode", choices=["think", "nonthink", "auto"], default="auto",
                    help="Expected format. 'auto' just reports the distribution.")
    ap.add_argument("--show-examples", type=int, default=3, help="Non-conformant examples to print.")
    ap.add_argument("--limit", type=int, default=None, help="Only check the first N responses.")
    args = ap.parse_args()

    if args.response is not None:
        items = [(0, args.response)]
    else:
        items = list(iter_responses(args.responses_file, args.response_field))
    if args.limit is not None:
        items = items[: args.limit]

    counts = Counter()
    reason_counts = Counter()
    think_lens, answer_lens = [], []
    bad_examples = []

    conform_fmt = {"think": "think", "nonthink": "nonthink"}.get(args.mode)

    for idx, resp in items:
        info = classify(resp)
        counts[info["format"]] += 1
        reason_counts[(info["format"], info["reason"])] += 1
        if info["think_len"]:
            think_lens.append(info["think_len"])
        if info["answer_len"]:
            answer_lens.append(info["answer_len"])
        if conform_fmt and info["format"] != conform_fmt and len(bad_examples) < args.show_examples:
            bad_examples.append((idx, info, resp))

    total = len(items)
    hr = "=" * 78

    if total == 1 and args.response is not None:
        idx, resp = items[0]
        info = classify(resp)
        print(f"format = {info['format']}   ({info['reason']})")
        print(f"think_chars = {info['think_len']}   answer_chars = {info['answer_len']}   "
              f"tags open/close = {info['n_open']}/{info['n_close']}")
        if conform_fmt:
            ok = info["format"] == conform_fmt
            print(f"\nExpected '{conform_fmt}' -> {'PASS ✅' if ok else 'FAIL ❌'}")
            sys.exit(0 if ok else 1)
        sys.exit(0)

    print(f"\n{hr}\nFORMAT DISTRIBUTION over {total} responses\n{hr}")
    for fmt in ("think", "nonthink", "malformed"):
        n = counts.get(fmt, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"  {fmt:<10} {n:>7}  ({pct:5.1f}%)")

    print(f"\n{hr}\nBREAKDOWN BY REASON\n{hr}")
    for (fmt, reason), n in reason_counts.most_common():
        print(f"  [{fmt:<9}] {n:>7}  {reason}")

    if think_lens:
        print(f"\n  think content chars: min={min(think_lens)} "
              f"mean={sum(think_lens)//len(think_lens)} max={max(think_lens)}")
    if answer_lens:
        print(f"  answer      chars: min={min(answer_lens)} "
              f"mean={sum(answer_lens)//len(answer_lens)} max={max(answer_lens)}")

    if conform_fmt:
        conform = counts.get(conform_fmt, 0)
        rate = 100.0 * conform / total if total else 0.0
        print(f"\n{hr}\nEXPECTED FORMAT: '{conform_fmt}'\n{hr}")
        print(f"  conformant: {conform}/{total}  ({rate:.1f}%)")
        if bad_examples:
            print(f"\n  Non-conformant examples (up to {args.show_examples}):")
            for idx, info, resp in bad_examples:
                snippet = resp[:400].replace("\n", "\\n")
                print(f"\n    [row {idx}] got '{info['format']}' ({info['reason']})")
                print(f"      {snippet!r}")
        rc = 0 if conform == total else 1
        print(f"\n{hr}\nRESULT: {'ALL CONFORMANT ✅' if rc == 0 else 'SOME NON-CONFORMANT ❌'}\n{hr}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
