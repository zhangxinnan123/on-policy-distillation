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


def _worker(
    gpu_id: int,
    model: str,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    dtype: str,
    conversations: List[List[Dict[str, str]]],
    row_indices: List[int],
    items: List[Dict],
    tmp_path: str,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> None:
    """Single-GPU worker: loads model, generates its slice, writes to tmp_path."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm = LLM(model=model, tensor_parallel_size=1, max_model_len=max_model_len, dtype=dtype)
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p, n=n)

    print(f"[GPU {gpu_id}] generating {len(conversations)} prompts ...")
    outputs: List[Any] = llm.chat(
        messages=conversations, sampling_params=sampling_params,
        **({"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else {}),
    ) or []

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row_idx, item, output in zip(row_indices, items, outputs):
            for sample_idx, comp in enumerate(output.outputs if output else []):
                record = {
                    "row_idx": row_idx,
                    "sample_idx": sample_idx,
                    "data_source": item.get("data_source"),
                    "original_question": (item.get("extra_info") or {}).get("original_question"),
                    "ground_truth": (item.get("reward_model") or {}).get("ground_truth"),
                    "prompt": item.get("prompt"),
                    "response": getattr(comp, "text", "") or "",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[GPU {gpu_id}] done → {tmp_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate responses from a parquet dataset (no logprobs).")
    p.add_argument("--parquet", required=True, help="Path to parquet file.")
    p.add_argument("--model", required=True, help="HF model path or local dir for vLLM.")
    p.add_argument("--out_dir", default="./parquet_inference_out", help="Base output dir.")
    p.add_argument("--limit", type=int, default=None, help="Number of rows to run (default: all).")
    p.add_argument("--max_model_len", type=int, default=32768, help="vLLM max_model_len.")
    p.add_argument("--max_new_tokens", type=int, default=16384, help="Max new tokens.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.7)
    p.add_argument("--n", type=int, default=1, help="Number of generations per prompt.")
    p.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size per worker (default: 1).",
    )
    p.add_argument(
        "--dp",
        type=int,
        default=0,
        help="Data parallel workers (number of GPUs to use in parallel). "
             "0 = auto from CUDA_VISIBLE_DEVICES. Overrides --tp to 1 per worker.",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"], help="Model dtype (default: bfloat16).")
    p.add_argument(
        "--enable_thinking",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=None,
        help="Pass enable_thinking to chat_template_kwargs (true/false). Default: not set (model default).",
    )
    args = p.parse_args()

    # Resolve available GPU IDs
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    all_gpu_ids = [x.strip() for x in cvd.split(",") if x.strip()] if cvd else []

    use_dp = args.dp > 0 or (args.dp == 0 and len(all_gpu_ids) > 1)

    if use_dp:
        dp = args.dp if args.dp > 0 else len(all_gpu_ids)
        gpu_ids = all_gpu_ids[:dp] if all_gpu_ids else list(range(dp))
        tp = args.tp
        print(f"[INFO] Data-parallel mode: {dp} workers, tp={tp} each, GPUs={gpu_ids}")
    else:
        dp = 1
        tp = args.tp if args.tp > 0 else max(1, len(all_gpu_ids))
        gpu_ids = all_gpu_ids or ["0"]
        print(f"[INFO] Single-worker mode: tensor_parallel_size={tp}  CUDA_VISIBLE_DEVICES={cvd!r}")

    # Load dataset
    ds = load_dataset("parquet", data_files=args.parquet, split="train")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    limit = len(ds)
    items = [ds[i] for i in range(limit)]

    conversations: List[List[Dict[str, str]]] = []
    for item in items:
        messages = item.get("prompt")
        if not isinstance(messages, list):
            raise ValueError("Expected parquet column 'prompt' to be a list of messages.")
        conversations.append(messages)

    # Output paths
    tag = Path(args.model.rstrip("/")).name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / Path(args.parquet).stem / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_t{args.temperature}_p{args.top_p}_results.jsonl"

    chat_template_kwargs: Dict[str, Any] = {}
    if args.enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = args.enable_thinking

    if not use_dp:
        # Single worker path
        llm = LLM(model=args.model, tensor_parallel_size=tp, max_model_len=args.max_model_len, dtype=args.dtype)
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, n=args.n
        )
        print(f"[INFO] Generating {limit} prompts × n={args.n} (enable_thinking={args.enable_thinking}) ...")
        outputs: List[Any] = llm.chat(
            messages=conversations, sampling_params=sampling_params,
            **({"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else {}),
        ) or []

        with out_path.open("w", encoding="utf-8") as f:
            for row_idx, (item, output) in enumerate(zip(items, outputs)):
                for sample_idx, comp in enumerate(output.outputs if output else []):
                    record = {
                        "row_idx": row_idx,
                        "sample_idx": sample_idx,
                        "data_source": item.get("data_source"),
                        "original_question": (item.get("extra_info") or {}).get("original_question"),
                        "ground_truth": (item.get("reward_model") or {}).get("ground_truth"),
                        "prompt": item.get("prompt"),
                        "response": getattr(comp, "text", "") or "",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        # Data-parallel path: split dataset across workers
        chunks = [[] for _ in range(dp)]
        indices = [[] for _ in range(dp)]
        chunk_items = [[] for _ in range(dp)]
        for i, (conv, item) in enumerate(zip(conversations, items)):
            w = i % dp
            chunks[w].append(conv)
            indices[w].append(i)
            chunk_items[w].append(item)

        tmp_paths = [str(out_dir / f"_tmp_worker{w}.jsonl") for w in range(dp)]

        ctx = multiprocessing.get_context("spawn")
        procs = []
        for w, gid in enumerate(gpu_ids):
            proc = ctx.Process(
                target=_worker,
                args=(
                    gid, args.model, args.max_model_len, args.max_new_tokens,
                    args.temperature, args.top_p, args.n, args.dtype,
                    chunks[w], indices[w], chunk_items[w], tmp_paths[w],
                    chat_template_kwargs or None,
                ),
            )
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker exited with code {proc.exitcode}")

        # Merge tmp files in row_idx order
        all_records = []
        for tmp in tmp_paths:
            with open(tmp, encoding="utf-8") as f:
                for line in f:
                    all_records.append(json.loads(line))
            Path(tmp).unlink()

        all_records.sort(key=lambda r: (r["row_idx"], r["sample_idx"]))
        with out_path.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {limit * args.n} records → {out_path}")


if __name__ == "__main__":
    main()
