"""Check whether a model emits <|im_end|> during generation and whether anything follows it.

The verl validation dumps are decoded with skip_special_tokens=True, so special
tokens are invisible there. This loads the model in vLLM and inspects the raw
generated token ids directly.

Reports, per sample: finish_reason, whether 151645 (<|im_end|>) / 151643
(<|endoftext|>) appear, their position, and how many tokens follow them.
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

IM_END = 151645
ENDOFTEXT = 151643


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=16384)
    p.add_argument("--max_model_len", type=int, default=18432)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[tokenizer] eos_token={tok.eos_token!r} eos_token_id={tok.eos_token_id}")

    df = pd.read_parquet(args.parquet).head(args.limit)
    convs = [list(x) for x in df["prompt"]]

    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len, dtype="bfloat16")
    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, n=1)
    outs = llm.chat(messages=convs, sampling_params=sp, chat_template_kwargs={"enable_thinking": True})

    fr = Counter()
    n_imend = n_trailing = 0
    print()
    print(f"{'#':>3} {'finish':>7} {'ntok':>6} {'im_end@':>9} {'after':>6} {'eot@':>7}  last5_ids")
    print("-" * 78)
    for i, o in enumerate(outs):
        c = o.outputs[0]
        ids = list(c.token_ids)
        fr[c.finish_reason] += 1
        pos = ids.index(IM_END) if IM_END in ids else -1
        after = (len(ids) - 1 - pos) if pos >= 0 else 0
        eot = ids.index(ENDOFTEXT) if ENDOFTEXT in ids else -1
        if pos >= 0:
            n_imend += 1
            if after > 0:
                n_trailing += 1
        print(f"{i:>3} {c.finish_reason:>7} {len(ids):>6} {pos:>9} {after:>6} {eot:>7}  {ids[-5:]}")

    n = len(outs)
    print()
    print(f"finish_reason: {dict(fr)}")
    print(f"emitted <|im_end|> (151645)      : {n_imend}/{n}")
    print(f"  ...with tokens AFTER it        : {n_trailing}/{n}   <- >0 means generation did not stop at EOS")
    print(f"tokenizer.eos_token_id used by vLLM: {tok.eos_token_id}")


if __name__ == "__main__":
    main()
