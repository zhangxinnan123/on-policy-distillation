"""Generate student trajectories with vs. without a privileged 'expert trajectory'
in the prompt, then score both with math_boxed and report per-row + aggregate.

Mirrors the prompt-augmentation scheme from sdpo_inference/run_teacher_inference_v2.py:
  - Expert source per row: `generations_wo_think[expert_index]`
  - Injection: appended to the last user turn via EXPERT_GUIDANCE_TEMPLATE

Unlike v2, this script *generates* new responses (does not score existing tokens),
so we can answer: does the privileged hint actually produce better trajectories?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from vllm import LLM, SamplingParams


# Reuse v2's template verbatim so the comparison reflects that exact intervention.
EXPERT_GUIDANCE_TEMPLATE = (
    "\n\nYou may use the expert trajectory only as *private guidance* to check your own reasoning.\n"
    "Do NOT quote, copy, paraphrase, or explicitly reference any sentence from it.\n"
    "Expert trajectory:{expert}\n"
    "Now solve the problem with your own step-by-step reasoning:"
)


def _augment_with_expert(
    prompt_messages: List[Dict[str, str]], expert_trajectory: Optional[str]
) -> List[Dict[str, str]]:
    if not expert_trajectory:
        return prompt_messages
    augmented = [dict(m) for m in prompt_messages]
    for i in range(len(augmented) - 1, -1, -1):
        if augmented[i].get("role") == "user":
            augmented[i]["content"] = (augmented[i].get("content", "") or "") + EXPERT_GUIDANCE_TEMPLATE.format(
                expert=expert_trajectory
            )
            return augmented
    augmented.append(
        {"role": "user", "content": EXPERT_GUIDANCE_TEMPLATE.format(expert=expert_trajectory).lstrip("\n")}
    )
    return augmented


def _load_math_boxed():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "math_boxed", repo_root / "verl/utils/reward_score/math_boxed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _coerce_messages(item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    msgs = item.get("prompt")
    if isinstance(msgs, list) and msgs and all(isinstance(m, dict) for m in msgs):
        return msgs
    q = item.get("question") or (item.get("extra_info") or {}).get("original_question")
    if isinstance(q, str) and q:
        return [{"role": "user", "content": q}]
    return None


def _coerce_ground_truth(item: Dict[str, Any]) -> Optional[str]:
    rm = item.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return rm["ground_truth"]
    if item.get("ground_truth") is not None:
        return item["ground_truth"]
    return None


def _auto_tp(tp: int) -> int:
    if tp > 0:
        return tp
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return 1
    return max(1, len([x for x in cvd.split(",") if x.strip()]))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare student trajectories with vs. without privileged expert hint."
    )
    p.add_argument("--hf_dataset", default="XinnanZhang/deepmath-diff6to8-verified")
    p.add_argument("--hf_split", default="train")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--out_dir", default="./sdpo_result/compare_privileged")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--expert_index", type=int, default=0, help="Index into generations_wo_think.")
    p.add_argument("--max_model_len", type=int, default=18000)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--tp", type=int, default=0, help="vLLM tensor_parallel_size (0=auto).")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable torch.compile (required for Qwen3 in some setups).",
    )
    p.add_argument(
        "--no_math_verify",
        action="store_true",
        help="Skip math_verify symbolic fallback in scoring.",
    )
    args = p.parse_args()

    # ---------- Load dataset ----------
    print(f"[INFO] Loading {args.hf_dataset} split={args.hf_split} ...", flush=True)
    ds = load_dataset(args.hf_dataset, split=args.hf_split)
    if args.limit and args.limit < len(ds):
        ds = ds.select(range(args.limit))
    items: List[Dict[str, Any]] = [ds[i] for i in range(len(ds))]
    print(f"[INFO] {len(items)} rows loaded. Columns: {ds.column_names}", flush=True)

    # ---------- Build paired conversations ----------
    rows: List[Dict[str, Any]] = []
    convs_baseline: List[List[Dict[str, str]]] = []
    convs_with_expert: List[List[Dict[str, str]]] = []

    n_missing_expert = 0
    for i, item in enumerate(items):
        msgs = _coerce_messages(item)
        gt = _coerce_ground_truth(item)
        if msgs is None or gt is None:
            continue
        gens = item.get("generations_wo_think")
        expert: Optional[str] = None
        if isinstance(gens, list) and gens:
            idx = args.expert_index if 0 <= args.expert_index < len(gens) else 0
            expert = gens[idx]
        if not expert:
            n_missing_expert += 1
            continue  # only keep rows where we have an expert to inject

        rows.append(
            {
                "row_idx": i,
                "data_source": item.get("data_source"),
                "ground_truth": gt,
                "prompt": msgs,
                "expert_trajectory": expert,
            }
        )
        convs_baseline.append(msgs)
        convs_with_expert.append(_augment_with_expert(msgs, expert))

    if not rows:
        print("[ERROR] No usable rows (need prompt+ground_truth+generations_wo_think).", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Pairs to generate: {len(rows)}  (skipped without expert: {n_missing_expert})", flush=True)

    # ---------- vLLM setup ----------
    tp = _auto_tp(args.tp)
    print(f"[INFO] vLLM model={args.model} tp={tp} dtype={args.dtype}", flush=True)
    llm_kwargs: Dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, n=1
    )

    # Concatenate baseline + with_expert into one batch.
    all_convs = convs_baseline + convs_with_expert
    print(f"[INFO] Generating {len(all_convs)} conversations in one batch ...", flush=True)
    outs = llm.chat(messages=all_convs, sampling_params=sp) or []
    half = len(convs_baseline)
    outs_baseline = outs[:half]
    outs_expert = outs[half:]

    def _text(o: Any) -> str:
        if not o or not getattr(o, "outputs", None):
            return ""
        return getattr(o.outputs[0], "text", "") or ""

    # ---------- Score with math_boxed ----------
    math_boxed = _load_math_boxed()
    use_mv = not args.no_math_verify

    n_help = n_hurt = n_tie_correct = n_tie_wrong = 0
    correct_baseline = 0
    correct_expert = 0
    for row, ob, oe in zip(rows, outs_baseline, outs_expert):
        rb = _text(ob)
        re_ = _text(oe)
        sb = math_boxed.compute_score(rb, row["ground_truth"], use_math_verify=use_mv)
        se = math_boxed.compute_score(re_, row["ground_truth"], use_math_verify=use_mv)
        row["response_baseline"] = rb
        row["response_with_expert"] = re_
        row["acc_baseline"] = bool(sb["acc"])
        row["acc_with_expert"] = bool(se["acc"])
        row["pred_baseline"] = sb.get("pred", "")
        row["pred_with_expert"] = se.get("pred", "")
        correct_baseline += int(sb["acc"])
        correct_expert += int(se["acc"])
        if sb["acc"] and se["acc"]:
            n_tie_correct += 1
        elif (not sb["acc"]) and (not se["acc"]):
            n_tie_wrong += 1
        elif (not sb["acc"]) and se["acc"]:
            n_help += 1
        else:
            n_hurt += 1

    # ---------- Write output ----------
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = Path(args.model.rstrip("/")).name
    out_dir = Path(args.out_dir) / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_compare_expert{args.expert_index}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "model": args.model,
        "hf_dataset": args.hf_dataset,
        "n_pairs": len(rows),
        "acc_baseline": correct_baseline / len(rows),
        "acc_with_expert": correct_expert / len(rows),
        "help": n_help,
        "hurt": n_hurt,
        "tie_correct": n_tie_correct,
        "tie_wrong": n_tie_wrong,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "expert_index": args.expert_index,
        "output_jsonl": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== Privileged-info comparison ==========")
    print(f"  pairs:           {len(rows)}")
    print(f"  acc baseline:    {correct_baseline}/{len(rows)} = {summary['acc_baseline']:.1%}")
    print(f"  acc with expert: {correct_expert}/{len(rows)} = {summary['acc_with_expert']:.1%}")
    print(f"  help (0→1):      {n_help}")
    print(f"  hurt (1→0):      {n_hurt}")
    print(f"  tie correct:     {n_tie_correct}")
    print(f"  tie wrong:       {n_tie_wrong}")
    print(f"\n[OK] paired jsonl: {out_path}")
    print(f"[OK] summary:      {summary_path}")


if __name__ == "__main__":
    main()
