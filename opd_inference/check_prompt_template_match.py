#!/usr/bin/env python3
"""Check that two models render the SAME prompt under a given chat-template setting.

Test 1 for OPD: confirm the base student (e.g. Qwen3-4B-Base) and the teacher
(e.g. Qwen3-8B) produce an IDENTICAL non-think prompt frame, so the student's
rollout tokens are valid, unchanged, in the teacher's frame (reapply_chat_template=False).

Renders the same messages with each tokenizer's apply_chat_template
(add_generation_prompt=True, enable_thinking=<flag>) and compares both the decoded
string and the token ids.

Run on the cluster:
    ssh sfm-science-sfm-p5-cluster "cd /fsx/xinnanzh/on-policy-distillation && \
        ~/miniconda3/bin/python opd_inference/check_prompt_template_match.py \
            --model-a Qwen/Qwen3-4B-Base --model-b Qwen/Qwen3-8B \
            --prompt 'What is 2+2?' --enable-thinking false"
"""

from __future__ import annotations

import argparse
import json
import sys

from transformers import AutoTokenizer


def _bool(s):
    if s is None:
        return None
    return str(s).strip().lower() in ("1", "true", "yes", "y", "t")


def _normalize(tokenized):
    if len(tokenized) > 0 and isinstance(tokenized[0], list):
        assert len(tokenized) == 1
        return list(tokenized[0])
    return list(tokenized)


def render(tok, messages, enable_thinking):
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    ids = _normalize(
        tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
    )
    text = tok.decode(ids, skip_special_tokens=False)
    return ids, text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", default="Qwen/Qwen3-4B-Base", help="Student / base model.")
    ap.add_argument("--model-b", default="Qwen/Qwen3-8B", help="Teacher model.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="Raw user prompt (wrapped as one user message).")
    src.add_argument("--messages-json", help="JSON list of {role, content} messages.")
    ap.add_argument("--enable-thinking", default="false", help="enable_thinking (true/false/unset).")
    args = ap.parse_args()

    if args.messages_json:
        messages = json.loads(args.messages_json)
    elif args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
    else:
        ap.error("Provide --prompt or --messages-json")

    et = _bool(args.enable_thinking)

    tok_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)

    ids_a, txt_a = render(tok_a, messages, et)
    ids_b, txt_b = render(tok_b, messages, et)

    hr = "=" * 78
    print(f"\n{hr}\nCONFIG\n{hr}")
    print(f"  model A (student/base) = {args.model_a}")
    print(f"  model B (teacher)      = {args.model_b}")
    print(f"  enable_thinking        = {et}")

    print(f"\n{hr}\nMODEL A PROMPT (decoded)\n{hr}\n{repr(txt_a)}")
    print(f"\n{hr}\nMODEL B PROMPT (decoded)\n{hr}\n{repr(txt_b)}")

    print(f"\n{hr}\nCOMPARISON\n{hr}")
    str_match = txt_a == txt_b
    ids_match = ids_a == ids_b
    print(f"  decoded string identical : {'YES ✅' if str_match else 'NO ❌'}")
    print(f"  token ids identical      : {'YES ✅' if ids_match else 'NO ❌'}  "
          f"(len A={len(ids_a)}, len B={len(ids_b)})")

    if not ids_match:
        for i, (a, b) in enumerate(zip(ids_a, ids_b)):
            if a != b:
                print(f"  first token diff at idx {i}: A={a} ({tok_a.decode([a])!r})  "
                      f"B={b} ({tok_b.decode([b])!r})")
                break
        if len(ids_a) != len(ids_b):
            print(f"  length differs: A={len(ids_a)} B={len(ids_b)}")

    ok = str_match and ids_match
    print(f"\n{hr}\nRESULT: {'SAME FORMAT ✅  base and teacher render identical non-think prompt' if ok else 'DIFFERENT ❌'}\n{hr}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
