"""Convert generated SFT JSONL to LlamaFactory sharegpt format.

Input JSONL (one record per line):
  {"row_idx": ..., "sample_idx": ..., "prompt": [{"role":"user","content":"..."}],
   "response": "...", "source": "...", "domain": "..."}

Output JSON (LlamaFactory sharegpt):
  [{"conversations": [{"from": "human", "value": "..."},
                      {"from": "gpt",   "value": "..."}]}, ...]
"""
import argparse
import json
from pathlib import Path

ROLE_MAP = {"user": "human", "assistant": "gpt", "system": "system"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input JSONL from run_generate_hf.py")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Output JSON (sharegpt array) for LlamaFactory")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # Convert prompt (OpenAI chat) -> sharegpt
            conv = []
            for m in r["prompt"]:
                conv.append({
                    "from": ROLE_MAP.get(m["role"], m["role"]),
                    "value": m["content"],
                })
            # Append assistant turn = response
            conv.append({"from": "gpt", "value": r["response"]})
            records.append({"conversations": conv})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()
