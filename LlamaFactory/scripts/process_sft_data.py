"""Process open-thoughts/OpenThoughts3-1.2M dataset.

Usage:
    # Count samples per domain (full download)
    python scripts/process_sft_data.py --count

    # Count unique questions and continuation counts in math domain
    python scripts/process_sft_data.py --count_unique

    # Preview a few math samples (streaming, no full download)
    python scripts/process_sft_data.py --preview

    # Sample 25K unique questions x 16 continuations and analyze lengths
    python scripts/process_sft_data.py --output_file data/openthoughts3_math_25k16.json

    # Custom unique/continuation counts
    python scripts/process_sft_data.py --output_file data/out.json --num_questions 10000 --num_continuations 8

    # Check completeness of an already-saved file
    python scripts/process_sft_data.py --check data/openthoughts3_math_25k16.json
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset


DATASET_NAME = "open-thoughts/OpenThoughts3-1.2M"
TARGET_DOMAIN = "math"


def count() -> None:
    """Load full dataset and print exact per-domain sample counts."""
    print(f"Loading {DATASET_NAME} (full download)...")
    ds = load_dataset(DATASET_NAME, split="train")
    counts = Counter(ds["domain"])
    total = len(ds)
    print(f"\n{'Domain':<20} {'Count':>10}  {'%':>6}")
    print("-" * 40)
    for domain, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{domain:<20} {n:>10,}  {100 * n / total:>5.1f}%")
    print("-" * 40)
    print(f"{'TOTAL':<20} {total:>10,}  100.0%")


def count_unique() -> None:
    """Count unique questions and their continuation counts in the math domain."""
    print(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train")
    math_ds = ds.filter(lambda x: x["domain"] == TARGET_DOMAIN, num_proc=8)
    print(f"Math samples: {len(math_ds):,}")

    # Group by first human turn
    question_to_samples: dict[str, list] = defaultdict(list)
    for sample in math_ds:
        for turn in sample["conversations"]:
            if turn["from"] == "human":
                question_to_samples[turn["value"].strip()].append(sample)
                break

    n_unique = len(question_to_samples)
    continuation_counts = Counter(len(v) for v in question_to_samples.values())
    total = len(math_ds)
    duplicates = total - n_unique

    print(f"\nTotal samples     : {total:,}")
    print(f"Unique questions  : {n_unique:,}")
    print(f"Duplicates        : {duplicates:,}  ({100 * duplicates / total:.1f}%)")

    print(f"\n{'Continuations':>15}  {'# Questions':>12}  {'% Questions':>12}")
    print("-" * 44)
    for k in sorted(continuation_counts):
        q_count = continuation_counts[k]
        print(f"{k:>15}  {q_count:>12,}  {100 * q_count / n_unique:>11.1f}%")

    eligible_16 = sum(v for k, v in continuation_counts.items() if k >= 16)
    print(f"\nQuestions with >= 16 continuations: {eligible_16:,}")


def preview(n: int = 3) -> None:
    """Stream a few math samples and print them."""
    print(f"Loading {DATASET_NAME} in streaming mode...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    seen = 0
    domain_counts: dict[str, int] = {}

    for sample in ds:
        domain = sample.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if domain == TARGET_DOMAIN and seen < n:
            print(f"\n{'=' * 60}")
            print(f"[Sample {seen + 1}]")
            print(f"  domain     : {sample['domain']}")
            print(f"  difficulty : {sample['difficulty']}")
            print(f"  source     : {sample['source']}")
            print(f"  conversations ({len(sample['conversations'])} turns):")
            for turn in sample["conversations"]:
                role = turn["from"]
                text = turn["value"]
                preview_text = text[:400] + ("..." if len(text) > 400 else "")
                print(f"\n    [{role}]\n    {preview_text}")
            seen += 1

        if seen >= n and sum(domain_counts.values()) >= 5000:
            break

    print(f"\n{'=' * 60}")
    print(f"Domain distribution (first {sum(domain_counts.values())} samples scanned):")
    for domain, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain:<20} {c}")


def _analyze_lengths(records: list[dict], tokenizer_name: str | None = None) -> None:
    """Find the longest sequence by character count, then tokenize only that one."""
    from transformers import AutoTokenizer

    if tokenizer_name is None:
        raise ValueError("--tokenizer is required for length analysis")

    # Find longest by character length (cheap, no tokenizer needed)
    ASSISTANT_ROLES = {"assistant", "gpt"}
    longest_rec = max(records, key=lambda r: sum(len(t["value"]) for t in r["conversations"]))
    convs = longest_rec["conversations"]
    q_text = next((t["value"] for t in convs if t["from"] == "human"), "")
    r_text = next((t["value"] for t in convs if t["from"] in ASSISTANT_ROLES), "")
    full_text = "".join(t["value"] for t in convs)

    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    q_tokens = len(tokenizer.encode(q_text, add_special_tokens=False))
    r_tokens = len(tokenizer.encode(r_text, add_special_tokens=False))
    full_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))

    print(f"\n=== Longest sequence ({tokenizer_name}) ===")
    print(f"  Question tokens      : {q_tokens:>10,}")
    print(f"  Response tokens      : {r_tokens:>10,}")
    print(f"  Full conv tokens     : {full_tokens:>10,}")


def check_file(input_file: str) -> None:
    """Stream through a saved JSON file and check response completeness."""
    import ijson

    print(f"Checking {input_file} ...")

    total = 0
    empty = 0
    no_think_close = 0
    ASSISTANT_ROLES = {"assistant", "gpt"}

    with open(input_file, "rb") as f:
        for rec in ijson.items(f, "item"):
            convs = rec["conversations"]
            assistant_text = next((t["value"] for t in convs if t["from"] in ASSISTANT_ROLES), "")
            stripped = assistant_text.strip()
            total += 1

            if not stripped:
                empty += 1
            if "</think>" not in stripped:
                no_think_close += 1

            if total % 50000 == 0:
                print(f"  ... {total:,} records scanned")

    print(f"\n{'Total records':<35}: {total:,}")
    print(f"{'Empty responses':<35}: {empty:,}")
    print(f"{'Missing </think>':<35}: {no_think_close:,}  ({100 * no_think_close / total:.1f}%)")
    print(f"{'Complete (have </think>)':<35}: {total - no_think_close:,}  ({100 * (total - no_think_close) / total:.1f}%)")



def _save_parquet(records: list[dict], output_file: str) -> None:
    """Write records to parquet using pa.list_ (not large_list) to avoid pyarrow scanner issues."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    turn_type = pa.struct([("from", pa.string()), ("value", pa.string())])
    schema = pa.schema([
        ("conversations", pa.list_(turn_type)),
        ("domain", pa.string()),
        ("difficulty", pa.string()),
        ("source", pa.string()),
    ])

    conversations = [[{"from": t["from"], "value": t["value"]} for t in r["conversations"]] for r in records]
    table = pa.table({
        "conversations": pa.array(conversations, type=pa.list_(turn_type)),
        "domain": pa.array([r["domain"] for r in records], type=pa.string()),
        "difficulty": pa.array([r["difficulty"] for r in records], type=pa.string()),
        "source": pa.array([r["source"] for r in records], type=pa.string()),
    }, schema=schema)

    pq.write_table(table, output_file)
    print(f"Saved {len(records):,} records to {output_file}")


def _push_to_hub(records: list[dict], repo_id: str) -> None:
    """Upload dataset as JSONL to HuggingFace Hub (avoids pyarrow parquet nested-type bug)."""
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        jsonl_path = f.name

    api.upload_file(
        path_or_fileobj=jsonl_path,
        path_in_repo="data/train-00000-of-00001.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    os.remove(jsonl_path)

    readme = """\
---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-00000-of-00001.jsonl
---
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")


def extract(
    output_file: str,
    num_questions: int = 25000,
    num_continuations: int = 16,
    seed: int = 42,
    tokenizer_name: str | None = None,
    push_to_hub: str | None = None,
    no_completeness_filter: bool = False,
) -> None:
    """Sample num_questions unique questions, each with num_continuations responses."""
    rng = random.Random(seed)

    print(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train")
    math_ds = ds.filter(lambda x: x["domain"] == TARGET_DOMAIN, num_proc=8)
    print(f"Math samples: {len(math_ds):,}")

    ASSISTANT_ROLES = {"assistant", "gpt"}

    if no_completeness_filter:
        print("Skipping completeness filter (--no_completeness_filter)")
    else:
        def is_complete(sample: dict) -> bool:
            text = next((t["value"] for t in sample["conversations"] if t["from"] in ASSISTANT_ROLES), "").strip()
            return bool(text) and "</think>" in text

        total_before = len(math_ds)
        math_ds = math_ds.filter(is_complete, num_proc=8)
        print(f"Complete samples (have </think>): {len(math_ds):,} / {total_before:,} ({100 * len(math_ds) / total_before:.1f}%)")

    # Group by first human turn
    HUMAN_ROLES = {"human", "user"}
    print("Grouping by question...")
    question_to_samples: dict[str, list] = defaultdict(list)
    for sample in math_ds:
        for turn in sample["conversations"]:
            if turn["from"] in HUMAN_ROLES:
                question_to_samples[turn["value"].strip()].append(sample)
                break

    # Keep only questions with enough continuations
    eligible = {q: samples for q, samples in question_to_samples.items() if len(samples) >= num_continuations}
    print(f"Questions with >= {num_continuations} continuations: {len(eligible):,}")

    if len(eligible) < num_questions:
        raise ValueError(
            f"Only {len(eligible):,} eligible questions, but {num_questions:,} requested. "
            f"Try reducing --num_questions or --num_continuations."
        )

    # Sample num_questions questions, then num_continuations per question
    sampled_questions = rng.sample(list(eligible.keys()), num_questions)

    import re
    _think_re = re.compile(r"^<think> +")

    def _normalize_think(text: str) -> str:
        """Convert '<think> word' to '<think>\nword'."""
        return _think_re.sub("<think>\n", text)

    records = []
    for q in sampled_questions:
        chosen = rng.sample(eligible[q], num_continuations)
        for sample in chosen:
            conversations = []
            for turn in sample["conversations"]:
                if turn["from"] in ASSISTANT_ROLES:
                    conversations.append({"from": turn["from"], "value": _normalize_think(turn["value"])})
                else:
                    conversations.append({"from": turn["from"], "value": turn["value"]})
            records.append({
                "conversations": conversations,
                "domain": sample["domain"],
                "difficulty": sample["difficulty"],
                "source": sample["source"],
            })

    total_records = len(records)
    print(f"\nSampled {num_questions:,} questions x {num_continuations} continuations = {total_records:,} records")

    if tokenizer_name:
        _analyze_lengths(records, tokenizer_name)

    if output_file.endswith(".parquet"):
        _save_parquet(records, output_file)
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {total_records:,} records to {output_file}")

    if push_to_hub:
        _push_to_hub(records, push_to_hub)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process OpenThoughts3-1.2M dataset")
    parser.add_argument("--count", action="store_true", help="Print exact per-domain sample counts (full download)")
    parser.add_argument("--count_unique", action="store_true", help="Count unique questions and continuation distribution")
    parser.add_argument("--preview", action="store_true", help="Stream and preview math samples without downloading")
    parser.add_argument("--preview_n", type=int, default=3, help="Number of math samples to preview (default: 3)")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save extracted data as JSON")
    parser.add_argument("--num_questions", type=int, default=25000, help="Number of unique questions to sample (default: 25000)")
    parser.add_argument("--num_continuations", type=int, default=16, help="Continuations per question (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-1.7B", help="Tokenizer for max-length analysis (default: Qwen/Qwen3-1.7B)")
    parser.add_argument("--check", type=str, default=None, metavar="FILE", help="Check completeness of a saved JSON file")
    parser.add_argument("--push_to_hub", type=str, default=None, metavar="REPO_ID", help="Push output parquet to HuggingFace Hub (e.g. MyUser/my-dataset)")
    parser.add_argument("--no_completeness_filter", action="store_true", help="Skip the </think> completeness filter")
    args = parser.parse_args()

    if args.count:
        count()
    elif args.count_unique:
        count_unique()
    elif args.preview:
        preview(n=args.preview_n)
    elif args.check:
        check_file(args.check)
    elif args.output_file:
        extract(
            args.output_file,
            num_questions=args.num_questions,
            num_continuations=args.num_continuations,
            seed=args.seed,
            tokenizer_name=args.tokenizer,
            push_to_hub=args.push_to_hub,
            no_completeness_filter=args.no_completeness_filter,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
