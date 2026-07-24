"""Merge extracted hints (JSONL) into the training parquet.

Input:
    - train.parquet           : produced by examples/data_preprocess/deepmath_diff6to8.py
                                columns: prompt, data_source, ability, reward_model,
                                         extra_info, generations_wo_think
    - train_hints.jsonl       : one line per row; each has {"index": int, "hints": [{"level_1":..., "level_2":..., "level_3":...}, ...]}

Output:
    - train_with_hints.parquet: original columns + level_1, level_2, level_3 (each is str)

Join key: `extra_info["index"]` on the parquet side, `"index"` on the jsonl side. Both
were assigned identically by the preprocessor + extractor (they iterated the HF split
in the same order), so this is a lossless 1:1 merge.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_hints(jsonl_path: Path) -> dict[int, dict]:
    """Return {index: {level_1, level_2, level_3}} — one hint per row."""
    hints = {}
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            idx = int(row["index"])
            h_list = row.get("hints", [])
            if not h_list:
                hints[idx] = {"level_1": "", "level_2": "", "level_3": ""}
                continue
            h = h_list[0]  # only 1 solution per row in the C-plan run
            hints[idx] = {
                "level_1": h.get("level_1", ""),
                "level_2": h.get("level_2", ""),
                "level_3": h.get("level_3", ""),
            }
    return hints


def main():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default=str(repo_root / "data/deepmath_diff6to8/train.parquet"))
    p.add_argument("--hints", default=str(repo_root / "data/deepmath_diff6to8/train_hints.jsonl"))
    p.add_argument("--output", default=str(repo_root / "data/deepmath_diff6to8/train_with_hints.parquet"))
    args = p.parse_args()

    parquet_path = Path(args.parquet)
    hints_path = Path(args.hints)
    out_path = Path(args.output)

    print(f"[INFO] parquet: {parquet_path}")
    print(f"[INFO] hints:   {hints_path}")

    df = pd.read_parquet(parquet_path)
    n = len(df)
    print(f"[INFO] loaded {n} parquet rows")

    hints = load_hints(hints_path)
    print(f"[INFO] loaded {len(hints)} hint rows")

    def get_hint_index(extra_info):
        # extra_info was written as {"index": idx, ...} by the preprocessor.
        return int(extra_info["index"])

    df["_join_idx"] = df["extra_info"].map(get_hint_index)

    missing = [idx for idx in df["_join_idx"] if idx not in hints]
    if missing:
        print(f"[WARN] {len(missing)} parquet rows have no matching hint (first 5: {missing[:5]})")
    extra = [idx for idx in hints if idx not in set(df["_join_idx"])]
    if extra:
        print(f"[WARN] {len(extra)} hint rows unmatched by parquet (first 5: {extra[:5]})")

    df["level_1"] = df["_join_idx"].map(lambda i: hints.get(i, {}).get("level_1", ""))
    df["level_2"] = df["_join_idx"].map(lambda i: hints.get(i, {}).get("level_2", ""))
    df["level_3"] = df["_join_idx"].map(lambda i: hints.get(i, {}).get("level_3", ""))
    df = df.drop(columns=["_join_idx"])

    empties = ((df["level_1"] == "") & (df["level_2"] == "") & (df["level_3"] == "")).sum()
    print(f"[INFO] rows with all 3 levels empty: {empties}")
    print(f"[INFO] final columns: {list(df.columns)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"[OK] wrote {out_path} ({out_path.stat().st_size / 1024**2:.1f} MB, {len(df)} rows)")

    print("\n=== sample row 0 ===")
    r = df.iloc[0]
    print(f"  extra_info.index: {r['extra_info']['index']}")
    print(f"  prompt (first 200 chars): {r['prompt'][0]['content'][:200]}...")
    print(f"  level_1: {r['level_1'][:150]}...")
    print(f"  level_2: {r['level_2'][:150]}...")
    print(f"  level_3: {r['level_3'][:150]}...")


if __name__ == "__main__":
    main()
