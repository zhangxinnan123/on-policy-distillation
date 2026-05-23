from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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


EXPERT_GUIDANCE_TEMPLATE = (
    "\n\nYou may use the expert trajectory only as *private guidance* to check your own reasoning.\n"
    "Do NOT quote, copy, paraphrase, or explicitly reference any sentence from it.\n"
    "Expert trajectory:{expert}\n"
    "Now solve the problem with your own step-by-step reasoning:"
)


def _augment_with_expert(prompt_messages: List[Dict[str, str]], expert_trajectory: Optional[str]) -> List[Dict[str, str]]:
    """Append the expert-trajectory guidance block to the last user message. No-op if no expert."""
    if not expert_trajectory:
        return prompt_messages
    augmented = [dict(m) for m in prompt_messages]
    for i in range(len(augmented) - 1, -1, -1):
        if augmented[i].get("role") == "user":
            augmented[i]["content"] = (augmented[i].get("content", "") or "") + EXPERT_GUIDANCE_TEMPLATE.format(expert=expert_trajectory)
            return augmented
    augmented.append({"role": "user", "content": EXPERT_GUIDANCE_TEMPLATE.format(expert=expert_trajectory).lstrip("\n")})
    return augmented


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


def _find_think_nl_ids(tokenizer: Any) -> Tuple[Optional[int], Optional[int]]:
    """Return (think_token_id, newline_token_id); None for either if not resolvable to a single token."""
    think_id = nl_id = None
    try:
        ids = tokenizer.encode("<think>", add_special_tokens=False)
        if len(ids) == 1:
            think_id = ids[0]
    except Exception:
        pass
    try:
        ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(ids) == 1:
            nl_id = ids[0]
    except Exception:
        pass
    return think_id, nl_id


def _pad_think_newline(
    gen_ids: List[int], think_id: Optional[int], nl_id: Optional[int], tokenizer: Any
) -> Tuple[List[int], Optional[int], Optional[int]]:
    """Assuming <think> is gen_ids[0], ensure gen_ids[1] is \\n.

    If gen_ids[1] has a leading space, strip it (despace) so the teacher sees a well-formed
    <think>\\n<token> sequence. Returns (padded_gen_ids, insert_pos, despace_id).
    """
    if think_id is None or nl_id is None or not gen_ids:
        return gen_ids, None, None
    if gen_ids[0] != think_id:
        return gen_ids, None, None
    if len(gen_ids) > 1 and gen_ids[1] == nl_id:
        return gen_ids, None, None

    despace_id: Optional[int] = None
    if len(gen_ids) > 1:
        try:
            next_str = tokenizer.decode([gen_ids[1]], skip_special_tokens=False)
            if next_str.startswith(" "):
                ids = tokenizer.encode(next_str[1:], add_special_tokens=False)
                if len(ids) == 1:
                    despace_id = ids[0]
        except Exception:
            pass

    if despace_id is not None:
        padded = [gen_ids[0], nl_id, despace_id] + gen_ids[2:]
    else:
        padded = [gen_ids[0], nl_id] + gen_ids[1:]
    return padded, 1, despace_id


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
    expert_index: int = -1,
    pad_think: bool = True,
) -> None:
    tp = _auto_tp(tp)
    llm_kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        enforce_eager=True,  # required for Qwen3 (torch.compile tracing issue)
    )
    mnbt = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "").strip()
    if mnbt:
        try:
            llm_kwargs["max_num_batched_tokens"] = int(mnbt)
        except ValueError:
            pass
    try:
        llm = LLM(**llm_kwargs)
    except TypeError:
        llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len, enforce_eager=True)
    tok = llm.get_tokenizer()
    # tok = AutoTokenizer.from_pretrained(model)
    think_id, nl_id = _find_think_nl_ids(tok)
    k = min(int(logprobs_k), VLLM_MAX_ALLOWED_LOGPROBS)
    sp = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, prompt_logprobs=k)

    prompts: List[str] = []
    # (gen_start, orig_gen_ids, padded_gen_ids, insert_pos, despace_id)
    metas: List[Tuple[int, List[int], List[int], Optional[int], Optional[int]]] = []
    for r in rows:
        prompt_msgs = r.get("prompt")
        resp = r.get("response", "")
        student_resp_ids = r.get("response_token_ids")
        # Backfill: rebuild prompt from original_question if missing.
        if not isinstance(prompt_msgs, list):
            q = r.get("original_question")
            if isinstance(q, str) and q:
                prompt_msgs = [{"role": "user", "content": q}]
        if not isinstance(prompt_msgs, list):
            prompts.append("")
            metas.append((0, [], [], None, None))
            r[f"{out_fields_prefix}_error"] = "missing_prompt"
            continue
        if not isinstance(student_resp_ids, list) or not all(isinstance(x, int) for x in student_resp_ids):
            prompts.append("")
            metas.append((0, [], [], None, None))
            r[f"{out_fields_prefix}_error"] = "missing_response_token_ids"
            continue

        # Augment the teacher prompt with one expert trajectory if requested.
        expert = None
        if expert_index >= 0:
            gens = r.get("generations_wo_think")
            if isinstance(gens, list) and gens:
                idx = expert_index if expert_index < len(gens) else 0
                expert = gens[idx].strip()
        teacher_prompt_msgs = _augment_with_expert(prompt_msgs, expert)
        # Save the teacher's augmented prompt for visualization/debugging.
        r[f"{out_fields_prefix}_prompt"] = teacher_prompt_msgs

        prompt_only = _render(tok, teacher_prompt_msgs, resp, add_gen_prompt=True)
        p_ids = _encode(tok, prompt_only)
        gen_start = len(p_ids)
        gen_ids: List[int] = list(student_resp_ids)

        if pad_think:
            padded_gen_ids, insert_pos, despace_id = _pad_think_newline(gen_ids, think_id, nl_id, tok)
        else:
            padded_gen_ids, insert_pos, despace_id = list(gen_ids), None, None

        full_ids = p_ids + padded_gen_ids
        full_text = tok.decode(full_ids, skip_special_tokens=False)

        prompts.append(full_text)
        metas.append((gen_start, gen_ids, padded_gen_ids, insert_pos, despace_id))

    def batches() -> Iterable[Tuple[List[str], List[Tuple[int, List[int], List[int], Optional[int], Optional[int]]], List[int]]]:
        for i in range(0, len(prompts), max(1, int(batch_size))):
            j = min(len(prompts), i + max(1, int(batch_size)))
            yield prompts[i:j], metas[i:j], list(range(i, j))

    for p_batch, m_batch, idxs in tqdm(list(batches()), desc=f"Scoring({out_fields_prefix})", unit="batch"):
        # vLLM rejects empty decoder prompts — filter out error-flagged rows.
        valid = [(p, m, ri) for p, m, ri in zip(p_batch, m_batch, idxs) if p]
        if not valid:
            continue
        p_batch = [p for p, _, _ in valid]
        m_batch = [m for _, m, _ in valid]
        idxs = [ri for _, _, ri in valid]
        outs = llm.generate(p_batch, sampling_params=sp) or []
        for out, (gen_start, gen_ids, padded_gen_ids, insert_pos, despace_id), ridx in zip(outs, m_batch, idxs):
            topk = _get_prompt_logprobs(out)
            if isinstance(topk, list):
                topk_slice = topk[gen_start : gen_start + len(padded_gen_ids)]
                if insert_pos is not None:
                    topk_slice = topk_slice[:insert_pos] + topk_slice[insert_pos + 1 :]
            else:
                topk_slice = None
            lookup_ids = list(gen_ids)
            if insert_pos is not None and despace_id is not None:
                lookup_ids[insert_pos] = despace_id
            lp = _selected_logprobs(lookup_ids, topk_slice)
            rows[ridx][f"{out_fields_prefix}_model"] = model
            rows[ridx][f"{out_fields_prefix}_response_token_ids"] = gen_ids
            rows[ridx][f"{out_fields_prefix}_response_token_logprobs"] = lp
            # Full top-k distribution per response position: [[token_id, logprob], ...] sorted desc.
            if k > 1 and isinstance(topk_slice, list):
                topk_per_pos = []
                for entry in topk_slice:
                    if isinstance(entry, dict):
                        pairs = sorted(
                            [(tid, float(v.logprob) if hasattr(v, "logprob") else float(v))
                             for tid, v in entry.items()],
                            key=lambda x: x[1], reverse=True,
                        )
                        topk_per_pos.append([[tid, lp_val] for tid, lp_val in pairs])
                    else:
                        topk_per_pos.append([])
                rows[ridx][f"{out_fields_prefix}_topk_logprobs"] = topk_per_pos


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score saved prompt/response jsonl with teacher vLLM model (logprobs for response tokens). No extra generation."
    )
    ap.add_argument("--input_jsonl", default="sdpo_result/Qwen3-4B/deepmath_diff6to8_verified/Qwen3-4B/20260417-013031_results.jsonl", help="Saved jsonl containing keys: prompt (messages list) + response (string).")
    ap.add_argument("--teacher_model", default="Qwen/Qwen3-4B", help="Teacher model to compute response logprobs (top-k prompt_logprobs).")
    ap.add_argument("--out_dir", default="/home/li003968/data/lzy_eval_result/scored_from_saved", help="Output base dir.")
    ap.add_argument("--limit", type=int, default=1, help="How many rows to process. (default: 1)")
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--logprobs_k", type=int, default=1)
    ap.add_argument("--tp_teacher", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--expert_index", type=int, default=-1, help="Index into generations_wo_think to use as expert trajectory. -1 disables augmentation (default: -1).")
    ap.add_argument("--pad_think", action="store_true", help="Enable <think>\\n padding/despacing of the student response before teacher scoring (default: off).")
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
        expert_index=int(args.expert_index),
        pad_think=args.pad_think,
    )


    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    teacher_name = Path(args.teacher_model.rstrip("/")).name
    out_dir = Path(args.out_dir) / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_tag = f"_expert{args.expert_index}" if args.expert_index >= 0 else ""
    out_path = out_dir / f"{stamp}_{teacher_name}{expert_tag}_scored.jsonl"
    with out_path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

