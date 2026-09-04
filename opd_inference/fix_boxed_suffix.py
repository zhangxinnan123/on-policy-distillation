"""Fix the malformed suffix `\\boxed{.}` in generated SFT data.

Reads the JSONL, replaces `\\boxed{.}` -> `\\boxed{}.` in each prompt's last
user turn, writes to a new file (or in-place with backup).
"""
import argparse
import json
import shutil
from pathlib import Path

BAD = r"\boxed{.}"
GOOD = r"\boxed{}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default=None,
                    help="Output path. If omitted, writes in place with .bak backup.")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        # In-place with backup
        bak = in_path.with_suffix(in_path.suffix + ".bak")
        if not bak.exists():
            print(f"Backing up {in_path} -> {bak}")
            shutil.copy2(in_path, bak)
        out_path = in_path

    n_total = 0
    n_fixed = 0
    lines_out = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            r = json.loads(line)
            fixed = False
            for m in r.get("prompt", []):
                c = m.get("content", "")
                if BAD in c:
                    m["content"] = c.replace(BAD, GOOD)
                    fixed = True
            if fixed:
                n_fixed += 1
            lines_out.append(json.dumps(r, ensure_ascii=False))

    with out_path.open("w", encoding="utf-8") as f:
        for l in lines_out:
            f.write(l + "\n")

    print(f"Rows total: {n_total}")
    print(f"Rows with fix: {n_fixed}")
    print(f"Wrote → {out_path}")


if __name__ == "__main__":
    main()
