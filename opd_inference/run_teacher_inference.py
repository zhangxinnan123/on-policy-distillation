from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


VLLM_MAX_ALLOWED_LOGPROBS = 20


def _auto_tp(tp: int) -> int:
    if tp > 0:
        return tp
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return 1
    return max(1, len([x for x in cvd.split(",") if x.strip()]))


def _read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
            if limit > 0 and len(out) >= limit:
                break
    return out


def _render(tokenizer: Any, prompt_messages: List[Dict[str, str]], response: str, add_gen_prompt: bool) -> str:
    msgs = prompt_messages if add_gen_prompt else (prompt_messages + [{"role": "assistant", "content": response}])
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen_prompt)
    # fallback: naive text
    if add_gen_prompt:
        return "\n".join([f"{m['role']}: {m.get('content','')}" for m in msgs] + ["assistant:"])
    return "\n".join([f"{m['role']}: {m.get('content','')}" for m in msgs])


def _encode(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _selected_logprobs(token_ids: List[int], topk_logprobs: Any) -> List[float]:
    out: List[float] = []
    if not isinstance(topk_logprobs, list):
        return [float("nan")] * len(token_ids)
    for tok, entry in zip(token_ids, topk_logprobs):
        if not isinstance(entry, dict):
            out.append(float("nan"))
            continue
        v = entry.get(tok)
        if v is None:
            out.append(float("nan"))
        else:
            out.append(float(v.logprob) if hasattr(v, "logprob") else float(v))
    return out


def _get_prompt_logprobs(req_out: Any) -> Any:
    # vLLM version differences: try output-level first, then output.outputs[0]
    v = getattr(req_out, "prompt_logprobs", None)
    if v is not None:
        return v
    if hasattr(req_out, "outputs") and req_out.outputs:
        return getattr(req_out.outputs[0], "prompt_logprobs", None)
    return None


def _score_one_model(
    rows: List[Dict[str, Any]],
    model: str,
    max_model_len: int,
    logprobs_k: int,
    tp: int,
    batch_size: int,
    out_fields_prefix: str,
) -> None:
    tp = _auto_tp(tp)
    # vLLM 的 prompt_logprobs 会在内部对 logits 做 float32 的 log_softmax。
    # 如果 chunked prefill 一次处理的 token 数太大（例如 16384），会出现巨大的临时张量，
    # 典型就是你看到的 “Tried to allocate ~9GiB”。
    #
    # 允许通过环境变量降低单次 prefill 的 token 数来避免显存尖峰：
    #   VLLM_MAX_NUM_BATCHED_TOKENS=2048
    llm_kwargs = dict(model=model, tensor_parallel_size=tp, max_model_len=max_model_len)
    mnbt = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "").strip()
    if mnbt:
        try:
            llm_kwargs["max_num_batched_tokens"] = int(mnbt)
        except ValueError:
            pass
    try:
        llm = LLM(**llm_kwargs)
    except TypeError:
        # older vLLM may not accept max_num_batched_tokens kwarg
        llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len)
    tok = llm.get_tokenizer()
    k = min(int(logprobs_k), VLLM_MAX_ALLOWED_LOGPROBS)
    # NOTE: some vLLM versions require max_tokens >= 1 (cannot be 0).
    # We only use prompt_logprobs (which scores the full prompt that includes the saved response),
    # and ignore the extra 1 generated token.
    sp = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, prompt_logprobs=k)

    prompts: List[str] = []
    metas: List[Tuple[int, List[int]]] = []  # (gen_start, gen_token_ids_from_student)
    for r in rows:
        prompt_msgs = r.get("prompt")
        resp = r.get("response", "")
        # Prefer using student's stored response tokenization to ensure alignment.
        student_resp_ids = r.get("response_token_ids")
        if not isinstance(prompt_msgs, list):
            prompts.append("")  # placeholder
            metas.append((0, []))
            r[f"{out_fields_prefix}_error"] = "missing_prompt"
            continue
        if not isinstance(student_resp_ids, list) or not all(isinstance(x, int) for x in student_resp_ids):
            prompts.append("")  # placeholder
            metas.append((0, []))
            r[f"{out_fields_prefix}_error"] = "missing_response_token_ids"
            continue
        prompt_only = _render(tok, prompt_msgs, resp, add_gen_prompt=True)
        p_ids = _encode(tok, prompt_only)
        gen_start = len(p_ids)
        gen_ids = student_resp_ids

        # Build full prompt text by decoding prompt token ids + student's response token ids.
        # This avoids chat-template whitespace/newline mismatches that change tokenization.
        full_ids = p_ids + gen_ids
        full_text = tok.decode(full_ids, skip_special_tokens=False)

        prompts.append(full_text)
        metas.append((gen_start, gen_ids))

    def batches() -> Iterable[Tuple[List[str], List[Tuple[int, List[int]]], List[int]]]:
        for i in range(0, len(prompts), max(1, int(batch_size))):
            j = min(len(prompts), i + max(1, int(batch_size)))
            yield prompts[i:j], metas[i:j], list(range(i, j))

    for p_batch, m_batch, idxs in tqdm(list(batches()), desc=f"Scoring({out_fields_prefix})", unit="batch"):
        outs = llm.generate(p_batch, sampling_params=sp) or []
        for out, (gen_start, gen_ids), ridx in zip(outs, m_batch, idxs):
            topk = _get_prompt_logprobs(out)
            topk_slice = topk[gen_start : gen_start + len(gen_ids)] if isinstance(topk, list) else None
            lp = _selected_logprobs(gen_ids, topk_slice)
            rows[ridx][f"{out_fields_prefix}_model"] = model
            rows[ridx][f"{out_fields_prefix}_response_token_ids"] = gen_ids
            rows[ridx][f"{out_fields_prefix}_response_token_logprobs"] = lp


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score saved prompt/response jsonl with teacher vLLM model (logprobs for response tokens). No extra generation."
    )
    ap.add_argument("--input_jsonl", required=True, help="Saved jsonl containing keys: prompt (messages list) + response (string).")
    ap.add_argument("--teacher_model", required=True, help="Teacher model to compute response logprobs (top-k prompt_logprobs).")
    ap.add_argument("--out_dir", default="/home/li003968/data/lzy_eval_result/scored_from_saved", help="Output base dir.")
    ap.add_argument("--limit", type=int, default=1, help="How many rows to process. (default: 1)")
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--logprobs_k", type=int, default=1)
    ap.add_argument("--tp_teacher", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    in_path = Path(args.input_jsonl).expanduser()
    rows = _read_jsonl(in_path, int(args.limit))


    _score_one_model(
        rows,
        model=str(args.teacher_model),
        max_model_len=int(args.max_model_len),
        logprobs_k=int(args.logprobs_k),
        tp=int(args.tp_teacher),
        batch_size=int(args.batch_size),
        out_fields_prefix="teacher_logp",
    )


    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    teacher_name = Path(args.teacher_model.rstrip("/")).name
    out_dir = Path(args.out_dir) / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_{teacher_name}_scored.jsonl"
    with out_path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

