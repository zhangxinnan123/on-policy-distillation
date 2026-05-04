"""Generate a continuation from a (prompt, partial-response) pair using vLLM.

Modeled after `run_generate_parquet_with_logpi.py`, but instead of reading
prompts from a parquet file, you supply a user prompt plus one or more
partial assistant responses, and the script asks vLLM to continue from
each one.

The script applies the model's chat template with `continue_final_message=True`
and `add_generation_prompt=False` so the assistant turn is *continued* rather
than re-opened.

Three input modes (mutually exclusive):
  1. Inline / file:   --prompt + --partial      (single (prompt, partial))
  2. Multi-partial:   --prompt + --partials_jsonl  (one prompt, many partials)
  3. Bundle:          --input_file BUNDLE.json
        A bundle produced by `prepare_partials_file.py` that holds the prompt
        and a list of {tag, partial} entries (one prompt, many partials). The
        bundle's top-level metadata (including the original full recorded
        response) is propagated into the output records as `bundle_meta`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams


VLLM_MAX_ALLOWED_LOGPROBS = 20


def _extract_selected_token_logprobs(token_ids, logprobs):
    if token_ids is None or logprobs is None:
        return None
    if isinstance(logprobs, list) and len(logprobs) == len(token_ids) and all(
        isinstance(x, (int, float, type(None))) for x in logprobs
    ):
        return [float(x) if x is not None else float("nan") for x in logprobs]
    if isinstance(logprobs, list) and len(logprobs) == len(token_ids):
        out = []
        for tok, lp_entry in zip(token_ids, logprobs):
            if not isinstance(lp_entry, dict):
                out.append(float("nan"))
                continue
            val = lp_entry.get(tok)
            if val is None:
                out.append(float("nan"))
                continue
            out.append(float(val.logprob) if hasattr(val, "logprob") else float(val))
        return out
    return None


def _extract_topk_logprobs(token_ids, logprobs):
    if token_ids is None or logprobs is None:
        return None
    if not (isinstance(logprobs, list) and len(logprobs) == len(token_ids)):
        return None
    out = []
    for lp_entry in logprobs:
        if not isinstance(lp_entry, dict):
            out.append(None)
            continue
        items: List[Tuple[int, float]] = []
        for tok_id, cand_val in lp_entry.items():
            cand_lp = float(cand_val.logprob) if hasattr(cand_val, "logprob") else float(cand_val)
            items.append((int(tok_id), cand_lp))
        items.sort(key=lambda kv: kv[1], reverse=True)
        out.append([{"token_id": tid, "logprob": lp} for tid, lp in items])
    return out


def _extract_first_completion(output):
    if output is None or not hasattr(output, "outputs") or not output.outputs:
        return "", None, None, None, {"error": "missing_output"}
    comp = output.outputs[0]
    text = getattr(comp, "text", "") or ""
    token_ids = list(getattr(comp, "token_ids", []) or [])
    logprobs = getattr(comp, "logprobs", None)
    token_logprobs = _extract_selected_token_logprobs(token_ids, logprobs)
    topk_logprobs = _extract_topk_logprobs(token_ids, logprobs)
    extra: Dict[str, Any] = {}
    cum_lp = getattr(comp, "cumulative_logprob", None)
    if cum_lp is not None:
        extra["cumulative_logprob"] = float(cum_lp)
    finish = getattr(comp, "finish_reason", None)
    if finish is not None:
        extra["finish_reason"] = finish
    stop = getattr(comp, "stop_reason", None)
    if stop is not None:
        extra["stop_reason"] = stop
    return text, token_ids, token_logprobs, topk_logprobs, extra


def _read_text_arg(value: Optional[str], file_path: Optional[str], name: str) -> Optional[str]:
    if value is not None and file_path is not None:
        raise ValueError(f"Provide only one of --{name} or --{name}_file.")
    if value is not None:
        return value
    if file_path is not None:
        return Path(file_path).read_text(encoding="utf-8")
    return None


def _build_messages(system: Optional[str], prompt: str, partial: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    msgs.append({"role": "assistant", "content": partial})
    return msgs


def main() -> None:
    p = argparse.ArgumentParser(description="Continue generation from (prompt, partial response).")
    p.add_argument("--model", required=True, help="HF model path or local dir for vLLM.")
    p.add_argument("--prompt", help="User prompt (inline string).")
    p.add_argument("--prompt_file", help="Path to a file containing the user prompt.")
    p.add_argument("--partial", help="Partial assistant response (inline string).")
    p.add_argument("--partial_file", help="Path to a file containing the partial assistant response.")
    p.add_argument(
        "--partials_jsonl",
        help=(
            "Path to a JSONL file with one partial per line. "
            "Each line may be a bare string OR an object with keys "
            "{'partial': str, 'tag': optional str}. All partials share the same prompt."
        ),
    )
    p.add_argument(
        "--input_file",
        help=(
            "Path to a bundle JSON file produced by `prepare_partials_file.py`. "
            "Contains the prompt and a list of {tag, partial} entries. The bundle's "
            "other top-level fields (including the original full response) are "
            "propagated into each output record as `bundle_meta`."
        ),
    )
    p.add_argument("--system", help="Optional system prompt (inline string).")
    p.add_argument("--system_file", help="Path to a file containing the system prompt.")
    p.add_argument("--out_dir", default="./continue_results", help="Where to write the JSONL result.")
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--n", type=int, default=1, help="Number of continuations to sample.")
    p.add_argument("--logprobs_k", type=int, default=0,
                   help="Top-k logprobs per generated token (0 = disabled).")
    p.add_argument(
        "--tp",
        type=int,
        default=0,
        help="Tensor parallel size for vLLM. 0 = auto from CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--print_only", action="store_true",
                   help="Print continuation(s) to stdout and skip writing JSONL.")
    args = p.parse_args()

    system = _read_text_arg(args.system, args.system_file, "system")

    # (tag, partial) pairs.
    entries: List[Tuple[str, str]] = []
    bundle_meta: Dict[str, Any] = {}

    if args.input_file is not None:
        if any(x is not None for x in (args.partial, args.partial_file, args.partials_jsonl,
                                        args.prompt, args.prompt_file)):
            sys.exit("error: --input_file cannot be combined with --prompt/--partial/--partials_jsonl")
        with open(args.input_file, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        prompt = bundle.get("prompt")
        if not isinstance(prompt, str):
            sys.exit("error: --input_file bundle missing string 'prompt'")
        partials_arr = bundle.get("partials")
        if not isinstance(partials_arr, list) or not partials_arr:
            sys.exit("error: --input_file bundle 'partials' must be a non-empty list")
        for idx, item in enumerate(partials_arr):
            if not isinstance(item, dict) or "partial" not in item:
                sys.exit(f"error: bundle.partials[{idx}] missing 'partial'")
            tag = str(item.get("tag", f"p{idx}"))
            entries.append((tag, item["partial"]))
        bundle_meta = {k: v for k, v in bundle.items() if k not in ("prompt", "partials")}
    else:
        prompt = _read_text_arg(args.prompt, args.prompt_file, "prompt")
        if prompt is None:
            sys.exit("error: --prompt / --prompt_file / --input_file is required")
        if args.partials_jsonl is not None:
            if args.partial is not None or args.partial_file is not None:
                sys.exit("error: do not combine --partials_jsonl with --partial/--partial_file")
            with open(args.partials_jsonl, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        entries.append((f"p{idx}", obj))
                    elif isinstance(obj, dict) and "partial" in obj:
                        entries.append((str(obj.get("tag", f"p{idx}")), obj["partial"]))
                    else:
                        sys.exit(f"error: line {idx} of --partials_jsonl is malformed")
            if not entries:
                sys.exit("error: --partials_jsonl had no usable lines")
        else:
            single = _read_text_arg(args.partial, args.partial_file, "partial")
            if single is None:
                sys.exit("error: --partial / --partial_file / --partials_jsonl / --input_file is required")
            entries.append(("p0", single))

    if args.logprobs_k > VLLM_MAX_ALLOWED_LOGPROBS:
        print(f"[WARN] clamping --logprobs_k to {VLLM_MAX_ALLOWED_LOGPROBS}.")
        args.logprobs_k = VLLM_MAX_ALLOWED_LOGPROBS

    tp = args.tp
    if tp <= 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        tp = max(1, len([x for x in cvd.split(",") if x.strip()])) if cvd else 1
    print(f"[INFO] vLLM tensor_parallel_size={tp} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")

    llm = LLM(model=args.model, tensor_parallel_size=tp, max_model_len=args.max_model_len)

    sp_kwargs: Dict[str, Any] = dict(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
    )
    if args.logprobs_k > 0:
        sp_kwargs["logprobs"] = args.logprobs_k
    if args.seed is not None:
        sp_kwargs["seed"] = args.seed
    sampling_params = SamplingParams(**sp_kwargs)

    conversations = [_build_messages(system, prompt, partial) for _, partial in entries]

    # Continue the assistant turn rather than starting a fresh one.
    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
        add_generation_prompt=False,
        continue_final_message=True,
    ) or []

    results = []
    for slot_idx, ((tag, partial), output) in enumerate(zip(entries, outputs)):
        completions = []
        comps = getattr(output, "outputs", []) if output is not None else []
        for i, comp in enumerate(comps):
            class _One:
                outputs = [comp]
            text, token_ids, token_logprobs, topk_logprobs, extra = _extract_first_completion(_One())
            completions.append({
                "sample_idx": i,
                "continuation": text,
                "full_assistant": partial + text,
                "response_token_ids": token_ids,
                "response_token_logprobs": token_logprobs,
                "response_topk_logprobs": topk_logprobs if args.logprobs_k > 0 else None,
                "vllm_extra": extra,
            })
        results.append({
            "slot_idx": slot_idx,
            "tag": tag,
            "partial": partial,
            "completions": completions,
        })

    if args.print_only:
        for r in results:
            print(f"\n##### tag={r['tag']} (slot {r['slot_idx']}) #####")
            for c in r["completions"]:
                print(f"\n----- model continuation (sample {c['sample_idx']}) -----")
                print(c["continuation"])
        return

    tag = Path(args.model.rstrip("/")).name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_continue.jsonl"

    sampling_meta = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": args.n,
        "logprobs_k": args.logprobs_k,
        "seed": args.seed,
    }
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            record = {
                "model": args.model,
                "system": system,
                "prompt": prompt,
                "tag": r["tag"],
                "slot_idx": r["slot_idx"],
                "partial": r["partial"],
                "sampling_params": sampling_meta,
                "completions": r["completions"],
                "bundle_meta": bundle_meta if bundle_meta else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[OK] wrote: {out_path}")
    for r in results:
        print(f"\n##### tag={r['tag']} (slot {r['slot_idx']}) #####")
        for c in r["completions"]:
            print(f"\n----- model continuation (sample {c['sample_idx']}) -----")
            print(c["continuation"])


if __name__ == "__main__":
    main()
