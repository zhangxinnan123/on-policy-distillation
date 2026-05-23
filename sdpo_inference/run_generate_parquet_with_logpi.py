from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from tqdm import tqdm
from vllm import LLM, SamplingParams


VLLM_MAX_ALLOWED_LOGPROBS = 20  # vLLM v1 engine currently enforces an upper bound (often 20)


def _extract_selected_token_logprobs(token_ids: Optional[List[int]], logprobs: Any) -> Optional[List[float]]:
    """Best-effort extraction of per-token logprob for the selected tokens."""
    if token_ids is None or logprobs is None:
        return None

    # Case: list[float] aligned with token_ids
    if isinstance(logprobs, list) and (len(logprobs) == len(token_ids)) and all(
        isinstance(x, (int, float, type(None))) for x in logprobs
    ):
        return [float(x) if x is not None else float("nan") for x in logprobs]

    # Case: list[dict[token_id -> (Logprob|float)]] aligned with token_ids
    if isinstance(logprobs, list) and (len(logprobs) == len(token_ids)):
        out: List[float] = []
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


def _extract_selected_token_ranks(token_ids: Optional[List[int]], logprobs: Any) -> Optional[List[Optional[int]]]:
    """Rank (1=highest logprob) of the selected token at each generation step, using returned top-k logprobs."""
    if token_ids is None or logprobs is None:
        return None
    if not (isinstance(logprobs, list) and len(logprobs) == len(token_ids)):
        return None

    ranks: List[Optional[int]] = []
    for tok, lp_entry in zip(token_ids, logprobs):
        if not isinstance(lp_entry, dict):
            ranks.append(None)
            continue
        val = lp_entry.get(tok)
        if val is None:
            ranks.append(None)  # selected token not in top-k
            continue
        selected_lp = float(val.logprob) if hasattr(val, "logprob") else float(val)
        num_greater = 0
        for cand_val in lp_entry.values():
            cand_lp = float(cand_val.logprob) if hasattr(cand_val, "logprob") else float(cand_val)
            if cand_lp > selected_lp:
                num_greater += 1
        ranks.append(1 + num_greater)
    return ranks


def _extract_first_completion(output: Any) -> Tuple[str, Optional[List[int]], Optional[List[float]], Optional[List[Optional[int]]], Dict[str, Any]]:
    """Extract text + token_ids + selected token logprobs + selected token ranks from vLLM output."""
    if output is None or not hasattr(output, "outputs") or not output.outputs:
        return "", None, None, None, {"error": "missing_output"}

    comp = output.outputs[0]
    text = getattr(comp, "text", "") or ""
    token_ids = getattr(comp, "token_ids", None)
    token_ids = list(token_ids) if token_ids is not None else None

    logprobs = getattr(comp, "logprobs", None)
    token_logprobs = _extract_selected_token_logprobs(token_ids, logprobs)
    token_ranks = _extract_selected_token_ranks(token_ids, logprobs)

    extra: Dict[str, Any] = {}
    cum_lp = getattr(comp, "cumulative_logprob", None)
    if cum_lp is not None:
        extra["cumulative_logprob"] = float(cum_lp)
    return text, token_ids, token_logprobs, token_ranks, extra


def main() -> None:
    p = argparse.ArgumentParser(description="Generate from a parquet dataset and save student token logp + rank.")
    p.add_argument("--parquet", required=True, help="Path to parquet file.")
    p.add_argument("--model", required=True, help="HF model path or local dir for vLLM.")
    p.add_argument("--out_dir", default="/home/li003968/data/lzy_eval_result/parquet_logpi", help="Base output dir.")
    p.add_argument("--limit", type=int, default=1, help="Number of rows to run (default: 1).")
    p.add_argument("--max_model_len", type=int, default=32768, help="vLLM max_model_len.")
    p.add_argument("--max_new_tokens", type=int, default=16384, help="Max new tokens.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--logprobs_k", type=int, default=20, help="Return top-k logprobs per position (for rank).")
    p.add_argument(
        "--tp",
        type=int,
        default=0,
        help="Tensor parallel size for vLLM. 0 = auto from CUDA_VISIBLE_DEVICES (default: 0).",
    )
    p.add_argument("--batch", action="store_true", help="Use batched vLLM chat().")
    p.add_argument("--n", type=int, default=1, help="Number of generations per prompt (each prompt is repeated n times).")
    args = p.parse_args()

    if args.logprobs_k > VLLM_MAX_ALLOWED_LOGPROBS:
        print(
            f"[WARN] requested --logprobs_k={args.logprobs_k} exceeds vLLM max allowed "
            f"({VLLM_MAX_ALLOWED_LOGPROBS}); clamping to {VLLM_MAX_ALLOWED_LOGPROBS}."
        )
        args.logprobs_k = VLLM_MAX_ALLOWED_LOGPROBS

    # Load parquet directly to avoid the pyarrow scanner bug that `datasets.load_dataset` trips on.
    # Only read columns we actually consume — skip nested list<struct> columns if present.
    pf = pq.ParquetFile(args.parquet)
    all_cols = pf.schema_arrow.names
    wanted = [c for c in ("prompt", "question", "ground_truth", "data_source", "extra_info", "reward_model", "generations_wo_think") if c in all_cols]
    ds: List[Dict[str, Any]] = []
    for batch in pf.iter_batches(columns=wanted, batch_size=500):
        ds.extend(batch.to_pylist())
        if len(ds) >= args.limit:
            break
    limit = min(args.limit, len(ds))
    ds = ds[:limit]

    tp = args.tp
    if tp <= 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            # e.g. "0,1,2,3" or "4,5"
            parts = [p.strip() for p in cvd.split(",") if p.strip() != ""]
            tp = max(1, len(parts))
        else:
            tp = 1
    print(f"[INFO] vLLM tensor_parallel_size={tp} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")

    llm = LLM(model=args.model, tensor_parallel_size=tp, max_model_len=args.max_model_len)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=args.logprobs_k,
    )

    # dataset prompt is either a messages list[dict{role,content}] in column 'prompt',
    # or a plain string in column 'question' (which we wrap into a single user turn).
    # Each prompt is repeated args.n times so we get n generations per row.
    conversations: List[List[Dict[str, str]]] = []
    for item in ds:
        messages = item.get("prompt")
        if not isinstance(messages, list):
            q = item.get("question")
            if not isinstance(q, str) or not q:
                raise ValueError("Parquet must have 'prompt' (messages list) or 'question' (string).")
            messages = [{"role": "user", "content": q}]
        for _ in range(args.n):
            conversations.append(messages)

    outputs: List[Any] = []
    if args.batch:
        outputs = llm.chat(messages=conversations, sampling_params=sampling_params) or []
    else:
        for conv in tqdm(conversations, desc="Generating", unit="row"):
            out = llm.chat(messages=conv, sampling_params=sampling_params)
            outputs.append(out[0] if out else None)

    tag = Path(args.model.rstrip("/")).name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / Path(args.parquet).stem / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_results.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for flat_idx in range(limit * args.n):
            row_idx = flat_idx // args.n
            sample_idx = flat_idx % args.n
            item = ds[row_idx]
            output = outputs[flat_idx] if flat_idx < len(outputs) else None
            text, token_ids, token_logprobs, token_ranks, extra = _extract_first_completion(output)
            original_question = (item.get("extra_info") or {}).get("original_question")
            if original_question is None:
                original_question = item.get("question")
            ground_truth = (item.get("reward_model") or {}).get("ground_truth")
            if ground_truth is None:
                ground_truth = item.get("ground_truth")
            # Pass through the full list of expert trajectories; the teacher script picks one.
            gens = item.get("generations_wo_think")
            generations_wo_think = list(gens) if isinstance(gens, list) else None
            # Save the messages list we actually used for inference (built from `prompt` or `question`).
            prompt_msgs = item.get("prompt")
            if not isinstance(prompt_msgs, list):
                q = item.get("question")
                prompt_msgs = [{"role": "user", "content": q}] if isinstance(q, str) and q else None
            record = {
                "row_idx": row_idx,
                "sample_idx": sample_idx,
                "data_source": item.get("data_source"),
                "original_question": original_question,
                "ground_truth": ground_truth,
                "generations_wo_think": generations_wo_think,
                # Split prompt / response explicitly.
                # Note: vLLM `token_ids/logprobs` here correspond to the generated response tokens only.
                "prompt": prompt_msgs,
                "response": text,
                "response_token_ids": token_ids,
                "response_token_logprobs": token_logprobs,
                "response_token_ranks": token_ranks,
                "logprobs_k": args.logprobs_k,
                "vllm_extra": extra,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

