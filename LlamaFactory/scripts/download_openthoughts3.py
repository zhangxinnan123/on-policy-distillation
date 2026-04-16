"""Download XinnanZhang/openthoughts3-math-25k16 and convert to JSON.

Downloads parquet files directly via huggingface_hub, then reads with
pyarrow.parquet.read_table() to avoid the Dataset scanner nested-type bug.

Usage:
    python scripts/download_openthoughts3.py
    python scripts/download_openthoughts3.py --output_file data/openthoughts3_math_25k16.json
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="data/openthoughts3_math_25k16.json")
    parser.add_argument("--dataset", default="XinnanZhang/openthoughts3-math-25k16")
    args = parser.parse_args()

    print(f"Downloading parquet files for {args.dataset} ...")
    local_dir = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        ignore_patterns=["*.md", "*.txt"],
    )
    print(f"Downloaded to: {local_dir}")

    parquet_files = sorted(Path(local_dir).rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet file(s)")

    records = []
    for path in parquet_files:
        print(f"  Reading {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
        pf = pq.ParquetFile(str(path))
        for i in range(pf.metadata.num_row_groups):
            batch = pf.read_row_group(i, columns=["conversations"])
            for row in batch.column("conversations").to_pylist():
                records.append({"conversations": row})
        print(f"  -> {len(records)} records so far")

    print(f"Writing {len(records)} records to {args.output_file} ...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Done. Saved {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
