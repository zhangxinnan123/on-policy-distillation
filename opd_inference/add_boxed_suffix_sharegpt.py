"""Append the boxed-answer suffix to the final user turn of a ShareGPT jsonl.

The openthoughts3-math-50k8 dataset has no instruction suffix on its prompts,
while our eval/OPD parquets (dapo_17k_aime2426-suffix) end every user turn with
"Please reason step by step, and put your final answer within \\boxed{}."
The reward scorer extracts \\boxed{}, so a model trained without the suffix can
answer correctly and still score 0.

Streams line-by-line so a ~19 GB file needs no meaningful memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def main() -> None:
    p = argparse.ArgumentParser(description="Append boxed suffix to ShareGPT user turns.")
    p.add_argument("--in", dest="in_path", required=True, help="Input ShareGPT .jsonl")
    p.add_argument("--out", dest="out_path", required=True, help="Output ShareGPT .jsonl")
    p.add_argument("--suffix", default=DEFAULT_SUFFIX)
    p.add_argument(
        "--sep",
        default="\n",
        help="Separator inserted between the prompt and the suffix (default: newline).",
    )
    p.add_argument(
        "--drop_truncated",
        action="store_true",
        help="Also drop records whose response never closes </think> or has no \\boxed{} "
             "after it (i.e. generation hit the length cap).",
    )
    args = p.parse_args()

    in_path, out_path = Path(args.in_path), Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = kept = already = dropped = no_user = 0
    with in_path.open(encoding="utf-8") as fi, out_path.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            n += 1
            r = json.loads(line)
            conv = r.get("conversations")
            if not isinstance(conv, list):
                raise ValueError(f"record {n}: expected 'conversations' list")

            if args.drop_truncated:
                g = next((t.get("value", "") for t in reversed(conv) if t.get("from") == "gpt"), "")
                tail = g.split("</think>", 1)
                if len(tail) < 2 or "boxed" not in tail[1]:
                    dropped += 1
                    continue

            # Append to the LAST human turn only.
            idx = next((i for i in range(len(conv) - 1, -1, -1) if conv[i].get("from") == "human"), None)
            if idx is None:
                no_user += 1
                continue
            val = conv[idx].get("value", "")
            if val.rstrip().endswith(args.suffix):
                already += 1
            else:
                conv[idx]["value"] = val.rstrip() + args.sep + args.suffix

            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

    print(f"read      : {n}")
    print(f"written   : {kept}")
    print(f"had suffix: {already}")
    if args.drop_truncated:
        print(f"dropped (truncated): {dropped}")
    if no_user:
        print(f"skipped (no human turn): {no_user}")


if __name__ == "__main__":
    main()
