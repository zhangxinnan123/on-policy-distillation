"""Generate SFT responses from a HuggingFace dataset with a vLLM teacher.

Supports:
  - HF dataset with either 'prompt' (verl-style chat messages) or
    'conversations' (ShareGPT-style [{from, value}, ...]) columns.
  - Multi-GPU data-parallel (spawn workers).
  - Sampling with top_p / top_k / min_p / temperature / n_per_prompt.
  - Optional chat_template_kwargs (e.g. enable_thinking for Qwen3).

Output: JSONL, one line per (prompt, sample) with (row_idx, sample_idx, prompt,
response, source, domain, difficulty, ground_truth).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from vllm import LLM, SamplingParams


# ShareGPT role → OpenAI chat role
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _to_chat_messages(item: Dict[str, Any], user_suffix: str = "") -> List[Dict[str, str]]:
    """Return the prompt as a list of {role, content} messages.

    Accepts:
      - item['prompt'] already a list of {role, content} (verl-style)
      - item['conversations'] a list of {from, value} (ShareGPT-style)

    For ShareGPT-style with a full user/assistant/user/... trajectory, we keep
    everything up to (but not including) the last assistant turn so the model
    is asked to regenerate the assistant response.

    If user_suffix is nonempty, it is appended to the final user turn (matches
    the format used in verl-training parquets, e.g. "Please reason step by
    step, and put your final answer within \\boxed{}.").
    """
    if isinstance(item.get("prompt"), list):
        msgs = list(item["prompt"])
    else:
        conv = item.get("conversations")
        if not isinstance(conv, list):
            raise ValueError("Expected 'prompt' or 'conversations' field.")
        msgs = []
        for turn in conv:
            role_src = (turn.get("from") or turn.get("role") or "").lower()
            role = ROLE_MAP.get(role_src, role_src or "user")
            content = turn.get("value") if "value" in turn else turn.get("content", "")
            msgs.append({"role": role, "content": content or ""})

        # Drop trailing assistant turn(s) so we regenerate the answer.
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()

    if user_suffix and msgs and msgs[-1]["role"] == "user":
        base = msgs[-1]["content"] or ""
        if not base.rstrip().endswith(user_suffix.strip()):
            msgs[-1] = {"role": "user", "content": (base.rstrip() + "\n" + user_suffix).strip()}
    return msgs


def _extract_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    """Pull dataset-specific metadata (best-effort; missing fields → None)."""
    meta = {
        "source": item.get("source") or item.get("data_source"),
        "domain": item.get("domain"),
        "difficulty": item.get("difficulty"),
    }
    # verl-style extras
    if isinstance(item.get("extra_info"), dict):
        meta["original_question"] = item["extra_info"].get("original_question")
    if isinstance(item.get("reward_model"), dict):
        meta["ground_truth"] = item["reward_model"].get("ground_truth")
    return {k: v for k, v in meta.items() if v is not None}


def _worker(
    gpu_id: int,
    model: str,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    n: int,
    dtype: str,
    conversations: List[List[Dict[str, str]]],
    row_indices: List[int],
    metas: List[Dict[str, Any]],
    tmp_path: str,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model=model, tensor_parallel_size=1, max_model_len=max_model_len, dtype=dtype)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        n=n,
    )

    print(f"[GPU {gpu_id}] generating {len(conversations)} prompts x n={n} ...", flush=True)
    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
        **({"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else {}),
    ) or []

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row_idx, meta, conv, output in zip(row_indices, metas, conversations, outputs):
            for sample_idx, comp in enumerate(output.outputs if output else []):
                rec = {
                    "row_idx": row_idx,
                    "sample_idx": sample_idx,
                    "prompt": conv,
                    "response": getattr(comp, "text", "") or "",
                    **meta,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[GPU {gpu_id}] done → {tmp_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SFT responses from a HF dataset with vLLM.")
    p.add_argument("--hf_dataset", required=True, help="HuggingFace dataset id (e.g. open-thoughts/OpenThoughts3-1.2M).")
    p.add_argument("--split", default="train")
    p.add_argument("--model", required=True, help="HF model id or local path for vLLM.")
    p.add_argument("--out_dir", default="./hf_gen_out")
    p.add_argument("--limit", type=int, default=None, help="Rows to keep (default: all).")
    p.add_argument("--start", type=int, default=0, help="Start row index (default: 0).")
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20, help="vLLM top_k (-1 disables).")
    p.add_argument("--min_p", type=float, default=0.0, help="vLLM min_p (0 disables).")
    p.add_argument("--n", type=int, default=8, help="Rollouts per prompt.")
    p.add_argument("--tp", type=int, default=1, help="Tensor-parallel per worker.")
    p.add_argument("--dp", type=int, default=0, help="Data-parallel workers (0 = auto from CUDA_VISIBLE_DEVICES).")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    p.add_argument(
        "--enable_thinking",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=None,
        help="Passed to chat_template_kwargs. Default: not set (model default).",
    )
    p.add_argument(
        "--user_suffix",
        default="",
        help="Optional text to append to the last user turn (mirrors verl "
             "training parquets, e.g. 'Please reason step by step, and put "
             "your final answer within \\boxed{}.').",
    )
    p.add_argument(
        "--domain_filter",
        default="",
        help="If set, keep only rows whose 'domain' field equals this string "
             "(e.g. 'math'). Empty = no filter.",
    )
    args = p.parse_args()

    # GPU/parallelism setup
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    all_gpu_ids = [x.strip() for x in cvd.split(",") if x.strip()] if cvd else []
    use_dp = args.dp > 0 or (args.dp == 0 and len(all_gpu_ids) > 1)

    if use_dp:
        dp = args.dp if args.dp > 0 else len(all_gpu_ids)
        gpu_ids = all_gpu_ids[:dp] if all_gpu_ids else [str(i) for i in range(dp)]
        tp = args.tp
        print(f"[INFO] Data-parallel: {dp} workers x tp={tp}   GPUs={gpu_ids}")
    else:
        dp = 1
        tp = args.tp if args.tp > 0 else max(1, len(all_gpu_ids))
        gpu_ids = all_gpu_ids or ["0"]
        print(f"[INFO] Single worker: tp={tp}   CUDA_VISIBLE_DEVICES={cvd!r}")

    # Load HF dataset
    print(f"[INFO] Loading HF dataset {args.hf_dataset} split={args.split} ...")
    ds = load_dataset(args.hf_dataset, split=args.split)
    total_raw = len(ds)
    print(f"[INFO] Raw dataset size: {total_raw}")

    if args.domain_filter:
        target = args.domain_filter.strip().lower()
        ds = ds.filter(
            lambda x: (x.get("domain") or "").strip().lower() == target,
            num_proc=max(1, os.cpu_count() // 4),
        )
        print(f"[INFO] After domain='{args.domain_filter}' filter: {len(ds)} rows")

    total = len(ds)
    end = total if args.limit is None else min(total, args.start + args.limit)
    ds = ds.select(range(args.start, end))
    print(f"[INFO] Using rows [{args.start}, {end}) → {len(ds)} prompts")

    conversations: List[List[Dict[str, str]]] = []
    metas: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        item = ds[i]
        conversations.append(_to_chat_messages(item, user_suffix=args.user_suffix))
        metas.append(_extract_meta(item))

    # Output paths
    tag = Path(args.model.rstrip("/")).name
    ds_tag = args.hf_dataset.replace("/", "__")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / ds_tag / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_t{args.temperature}_p{args.top_p}_k{args.top_k}_n{args.n}_results.jsonl"

    chat_template_kwargs: Dict[str, Any] = {}
    if args.enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = args.enable_thinking

    if not use_dp:
        llm = LLM(model=args.model, tensor_parallel_size=tp, max_model_len=args.max_model_len, dtype=args.dtype)
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            n=args.n,
        )
        print(f"[INFO] Generating {len(conversations)} prompts x n={args.n} (thinking={args.enable_thinking})")
        outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            **({"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else {}),
        ) or []
        with out_path.open("w", encoding="utf-8") as f:
            for row_idx, (meta, conv, output) in enumerate(zip(metas, conversations, outputs)):
                for sample_idx, comp in enumerate(output.outputs if output else []):
                    rec = {
                        "row_idx": row_idx + args.start,
                        "sample_idx": sample_idx,
                        "prompt": conv,
                        "response": getattr(comp, "text", "") or "",
                        **meta,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        # Split into dp chunks
        chunks = [[] for _ in range(dp)]
        indices = [[] for _ in range(dp)]
        chunk_metas = [[] for _ in range(dp)]
        for i, (conv, meta) in enumerate(zip(conversations, metas)):
            w = i % dp
            chunks[w].append(conv)
            indices[w].append(i + args.start)
            chunk_metas[w].append(meta)

        tmp_paths = [str(out_dir / f"_tmp_worker{w}.jsonl") for w in range(dp)]
        ctx = multiprocessing.get_context("spawn")
        procs = []
        for w, gid in enumerate(gpu_ids):
            proc = ctx.Process(
                target=_worker,
                args=(
                    gid, args.model, args.max_model_len, args.max_new_tokens,
                    args.temperature, args.top_p, args.top_k, args.min_p,
                    args.n, args.dtype,
                    chunks[w], indices[w], chunk_metas[w], tmp_paths[w],
                    chat_template_kwargs or None,
                ),
            )
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()

        # Concatenate tmp files → final out_path
        with out_path.open("w", encoding="utf-8") as fout:
            for tp_path in tmp_paths:
                if os.path.exists(tp_path):
                    with open(tp_path, "r", encoding="utf-8") as fin:
                        fout.write(fin.read())
                    os.remove(tp_path)

    print(f"\n[DONE] Wrote {out_path}")
    # Line count
    n_lines = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
    print(f"[DONE] Total records: {n_lines}")


if __name__ == "__main__":
    main()
