"""Expand a (DAPO parquet, full-generation JSONL) pair into a partials parquet.

Each source row → N output rows. Cutoffs are at ``k/N`` of the tokenized full
generation length, for ``k = 0, 1, ..., N-1``. The k=0 row is plain (empty
assistant continuation, default agent loop); k>0 rows carry the partial text
and the continuation agent name so verl's per-row dispatcher routes them to
``single_turn_partial_continue_agent``.

Inputs
  --gen_jsonl     JSONL with one record per source row, fields used:
                  ``row_idx`` (int, used as the join key) and ``response`` (str).
                  Produced by e.g. ``opd_inference/run_teacher_inference.py``.
  --base_parquet  DAPO-style parquet with columns
                  ``data_source, prompt, ability, reward_model, extra_info``.
                  Joined by row position == ``row_idx``.

Output
  --out_parquet   train.parquet equivalent with the original columns plus
                  ``partial`` (str), ``agent_name`` (str), ``partial_idx`` (int),
                  ``cutoff_tokens`` (int).

Optional test passthrough
  --test_in / --test_out  read a test parquet, add empty ``partial`` and
                  ``agent_name=single_turn_agent`` columns so schema matches
                  the train output, then write to ``--test_out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


PARTIAL_AGENT = "single_turn_partial_continue_agent"
DEFAULT_AGENT = "single_turn_agent"


def _load_gen_jsonl_by_row_idx(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row_idx = rec.get("row_idx")
            response = rec.get("response")
            if row_idx is None or not isinstance(response, str):
                continue
            if row_idx in out:
                # Multiple sample_idx values for the same row → keep the first.
                continue
            out[int(row_idx)] = response
    return out


def _cutoffs(token_count: int, n_partials: int) -> List[int]:
    return [int(round(k / n_partials * token_count)) for k in range(n_partials)]


def _expand_row(
    base_row: Dict[str, Any],
    response: str,
    tokenizer,
    n_partials: int,
) -> List[Dict[str, Any]]:
    token_ids = tokenizer.encode(response, add_special_tokens=False) if response else []
    cutoffs = _cutoffs(len(token_ids), n_partials)

    rows: List[Dict[str, Any]] = []
    for k, cut in enumerate(cutoffs):
        if cut == 0:
            partial = ""
            agent = DEFAULT_AGENT
        else:
            partial = tokenizer.decode(token_ids[:cut], skip_special_tokens=False)
            agent = PARTIAL_AGENT

        new_row = dict(base_row)
        new_row["partial"] = partial
        new_row["agent_name"] = agent
        new_row["partial_idx"] = k
        new_row["cutoff_tokens"] = cut
        rows.append(new_row)
    return rows


def _expand_train(args: argparse.Namespace) -> None:
    base_path = Path(args.base_parquet)
    gen_path = Path(args.gen_jsonl)
    out_path = Path(args.out_parquet)

    print(f"[load] base parquet: {base_path}")
    base_df = pd.read_parquet(base_path)
    print(f"[load]   {len(base_df)} rows, columns={list(base_df.columns)}")

    print(f"[load] gen jsonl: {gen_path}")
    gens = _load_gen_jsonl_by_row_idx(gen_path)
    print(f"[load]   {len(gens)} unique row_idx records")

    print(f"[load] tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    n = args.num_partials
    if n < 1:
        raise SystemExit(f"--num_partials must be >= 1 (got {n})")

    base_records = base_df.to_dict(orient="records")
    if args.limit > 0:
        base_records = base_records[: args.limit]
        print(f"[info] --limit applied: processing {len(base_records)} source rows")

    missing_rows = 0
    empty_responses = 0
    out_rows: List[Dict[str, Any]] = []
    for row_idx, base_row in enumerate(tqdm(base_records, desc="expanding")):
        response = gens.get(row_idx)
        if response is None:
            missing_rows += 1
            continue
        if not response.strip():
            empty_responses += 1
        out_rows.extend(_expand_row(base_row, response, tokenizer, n))

    print(f"[stat] source rows processed:  {len(base_records)}")
    print(f"[stat] source rows with no gen: {missing_rows}")
    print(f"[stat] empty responses:        {empty_responses}")
    print(f"[stat] output rows:            {len(out_rows)}  "
          f"(expected ~{(len(base_records) - missing_rows) * n})")

    out_df = pd.DataFrame(out_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[ok]   wrote {out_path}  ({len(out_df)} rows)")

    if len(out_df) > 0:
        sample = out_df.iloc[0]
        preview = sample["partial"]
        if isinstance(preview, str) and len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"[head] partial_idx=0 cutoff_tokens={sample['cutoff_tokens']} "
              f"agent={sample['agent_name']} partial={preview!r}")
        if len(out_df) > 1:
            sample = out_df.iloc[1]
            preview = sample["partial"]
            if isinstance(preview, str) and len(preview) > 120:
                preview = preview[:120] + "..."
            print(f"[head] partial_idx=1 cutoff_tokens={sample['cutoff_tokens']} "
                  f"agent={sample['agent_name']} partial={preview!r}")


def _passthrough_test(args: argparse.Namespace) -> None:
    test_in = Path(args.test_in)
    test_out = Path(args.test_out)
    print(f"[load] test parquet: {test_in}")
    df = pd.read_parquet(test_in)
    print(f"[load]   {len(df)} rows")

    df = df.copy()
    df["partial"] = ""
    df["agent_name"] = DEFAULT_AGENT
    df["partial_idx"] = 0
    df["cutoff_tokens"] = 0

    test_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(test_out, index=False)
    print(f"[ok]   wrote {test_out}  ({len(df)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen_jsonl", required=True,
                   help="JSONL with row_idx + response for each source row.")
    p.add_argument("--base_parquet", required=True,
                   help="DAPO-style parquet whose row positions match row_idx.")
    p.add_argument("--out_parquet", required=True,
                   help="Output train parquet path.")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base",
                   help="Student tokenizer (HF id or local path).")
    p.add_argument("--num_partials", type=int, default=4,
                   help="Number of partial rows per source row (cutoffs at k/N).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N source rows (debug).")
    p.add_argument("--test_in", default=None,
                   help="Optional: input test parquet for schema-aligned passthrough.")
    p.add_argument("--test_out", default=None,
                   help="Optional: output test parquet path. Required iff --test_in.")
    args = p.parse_args()

    if (args.test_in is None) != (args.test_out is None):
        raise SystemExit("--test_in and --test_out must be provided together")

    _expand_train(args)

    if args.test_in is not None:
        _passthrough_test(args)


if __name__ == "__main__":
    main()
