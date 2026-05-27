"""Preprocess XinnanZhang/deepmath-diff6to8-verified into verl GRPO parquet schema.

Output row schema (matching examples/data_preprocess/gsm8k.py):
    data_source : "math_dapo"      # routes to math_verify.compute_score in default_compute_score
    prompt      : [{"role": "user", "content": <question>}]
    ability     : "math"
    reward_model: {"style": "rule", "ground_truth": <ground_truth>}
    extra_info  : {"index": idx, "difficulty": ..., "original_question": ...}
    generations_wo_think : preserved verbatim for later SDPO use (the privileged expert trajectories).

Usage:
    python examples/data_preprocess/deepmath_diff6to8.py \
        --local_save_dir ~/data/deepmath_diff6to8
"""

import argparse
import os

from datasets import load_dataset


HF_DATASET = "XinnanZhang/deepmath-diff6to8-verified"
DATA_SOURCE = "math_dapo"  # routes to math_verify; same family as the AIME test parquet
# Appended to every training prompt so the model emits a \boxed{} answer that the
# reward function can extract (matches the AIME test parquet's expected format).
INSTRUCTION_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dataset", default=HF_DATASET)
    p.add_argument("--hf_split", default="train")
    p.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/deepmath_diff6to8"),
        help="Output directory for train.parquet.",
    )
    p.add_argument("--limit", type=int, default=0, help="If >0, only keep first N rows (debug).")
    args = p.parse_args()

    ds = load_dataset(args.hf_dataset, split=args.hf_split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[INFO] Loaded {len(ds)} rows from {args.hf_dataset}:{args.hf_split}")
    print(f"[INFO] Columns: {ds.column_names}")

    def _map(example, idx):
        question = example.get("question") or ""
        prompt_content = question + INSTRUCTION_SUFFIX
        gt = example.get("ground_truth")
        gens = example.get("generations_wo_think")
        difficulty = example.get("difficulty")
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(gt) if gt is not None else ""},
            "extra_info": {
                "index": idx,
                "difficulty": difficulty,
                "original_question": question,
            },
            # Keep the privileged expert trajectories alongside the row so SDPO can use them later.
            "generations_wo_think": list(gens) if isinstance(gens, list) else [],
        }

    ds = ds.map(_map, with_indices=True, remove_columns=ds.column_names)

    os.makedirs(args.local_save_dir, exist_ok=True)
    out_path = os.path.join(args.local_save_dir, "train.parquet")
    ds.to_parquet(out_path)
    print(f"[OK] Wrote {len(ds)} rows -> {out_path}")


if __name__ == "__main__":
    main()
