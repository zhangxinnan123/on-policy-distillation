"""Extract 3-level progressive hints from `generations_wo_think` via Bedrock (Claude Haiku/Sonnet).

Uses the tutoring-style prompt in `sdpo_experiment/data_process/hint_generation.jinja`.

Input parquet must have columns:
    prompt : list[{role, content}]
    generations_wo_think : list[str]

Output JSONL — one row per line, streamed as each row completes (resumable):
    {"index": 0, "question": "...", "hints": [{"level_1": "...", "level_2": "...", "level_3": "..."}]}

Resume: rerun the same command with the same --output path; already-completed indices
are skipped based on the "index" field in the existing JSONL.

Usage:
    # 10-row smoke test (default)
    python sdpo_inference/extract_hints_bedrock.py

    # full 56k, 1 hint per row (default), Haiku 4.5 (default)
    python sdpo_inference/extract_hints_bedrock.py \
        --input  data/deepmath_diff6to8/train.parquet \
        --output data/deepmath_diff6to8/train_hints.jsonl \
        --concurrency 32
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config
from jinja2 import Template

# Available on Bedrock (sfm-science account, us-west-2), strongest -> cheapest:
#   us.anthropic.claude-opus-4-8                     (~$15/M in, $75/M out)
#   us.anthropic.claude-sonnet-5                     (~$3/M in, $15/M out)
#   us.anthropic.claude-sonnet-4-20250514-v1:0
#   us.anthropic.claude-haiku-4-5-20251001-v1:0      (~$1/M in, $5/M out — default)
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "sdpo_experiment" / "data_process" / "hint_generation.jinja"

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
LEVEL_RE = re.compile(r'"level_([123])"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
LEVEL_LOOSE_RE = re.compile(
    r'"level_([123])"\s*:\s*"(.+?)"\s*(?:,\s*"level_[123]"|\}\s*```|\}\s*$)',
    re.DOTALL,
)


def load_template() -> Template:
    return Template(TEMPLATE_PATH.read_text(), keep_trailing_newline=True)


def strip_question(prompt_content: str) -> str:
    return prompt_content.split("\nPlease reason")[0].strip()


def render_user_message(template: Template, question: str, solution: str) -> str:
    return template.render(question=question, solution=solution)


def parse_hint_json(text: str) -> dict:
    """Extract {level_1, level_2, level_3}. Strict JSON first, then regex fallback."""
    m = JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else text.strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and any(k.startswith("level_") for k in obj):
            return {k: obj.get(k, "") for k in ("level_1", "level_2", "level_3")}
    except json.JSONDecodeError:
        pass
    out = {"level_1": "", "level_2": "", "level_3": ""}
    matched = False
    for rx in (LEVEL_LOOSE_RE, LEVEL_RE):
        for lvl, val in rx.findall(candidate):
            key = f"level_{lvl}"
            if not out[key]:
                out[key] = val.replace('\\"', '"').replace("\\n", "\n").strip()
                matched = True
        if all(out[k] for k in out):
            break
    if not matched:
        return {**out, "_raw": text}
    return out


def make_client(profile: str, region: str):
    kwargs = {"region_name": region}
    if profile:  # empty string -> fall back to default credential chain (works on cluster IAM role)
        kwargs["profile_name"] = profile
    session = boto3.Session(**kwargs)
    cfg = Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=120, connect_timeout=10)
    return session.client("bedrock-runtime", config=cfg)


def extract_text_from_response(payload: dict) -> str:
    blocks = payload.get("content", [])
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    if not texts:
        texts = [b["text"] for b in blocks if "text" in b]
    if not texts:
        raise ValueError(f"no text blocks in response; got types={[b.get('type') for b in blocks]}")
    return "\n".join(t for t in texts if t)


def extract_one(client, template: Template, question: str, solution: str, max_tokens: int = 4096) -> dict:
    """One Bedrock invoke; on thinking-only responses (max_tokens hit), retry once with 2x."""
    user_msg = render_user_message(template, question, solution)
    for attempt in range(2):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens * (attempt + 1),
            "messages": [{"role": "user", "content": user_msg}],
        }
        resp = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        try:
            text = extract_text_from_response(payload)
            return parse_hint_json(text)
        except ValueError:
            if payload.get("stop_reason") == "max_tokens" and attempt == 0:
                continue
            raise
    raise RuntimeError("unreachable")


def extract_row(client, template: Template, row_idx: int, question: str, solutions: list[str]):
    hints = []
    for sol in solutions:
        try:
            hints.append(extract_one(client, template, question, sol))
        except Exception as e:
            hints.append({"level_1": "", "level_2": "", "level_3": "", "_error": f"{type(e).__name__}: {str(e)[:200]}"})
    return row_idx, hints


def read_done_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def main():
    global MODEL_ID
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(repo_root / "data/deepmath_diff6to8/train_sample10.parquet"))
    p.add_argument("--output", default=str(repo_root / "data/deepmath_diff6to8/train_sample10_hints.jsonl"))
    p.add_argument("--profile", default="sfm-science-sfm-p5-cluster")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="If >0, process only first N rows.")
    p.add_argument("--n-solutions", type=int, default=1,
                   help="Hints per row: 1 = use only generations_wo_think[0]; 3 = use all three.")
    p.add_argument("--model-id", default=MODEL_ID)
    args = p.parse_args()
    MODEL_ID = args.model_id

    template = load_template()
    print(f"[INFO] template: {TEMPLATE_PATH}")
    print(f"[INFO] model:    {MODEL_ID}")

    df = pd.read_parquet(args.input)
    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
    n = len(df)
    print(f"[INFO] loaded {n} rows from {args.input}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = read_done_indices(out_path)
    remaining = [i for i in range(n) if i not in already_done]
    print(f"[INFO] already done: {len(already_done)} | remaining: {len(remaining)} | output: {out_path}")

    if not remaining:
        print("[OK] nothing to do")
        return

    client = make_client(args.profile, args.region)
    write_lock = threading.Lock()
    t0 = time.time()
    done_count = 0
    err_count = 0

    def submit(idx: int):
        row = df.iloc[idx]
        q = strip_question(row["prompt"][0]["content"])
        sols = list(row["generations_wo_think"])[: args.n_solutions]
        return extract_row(client, template, idx, q, sols)

    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(submit, idx): idx for idx in remaining}
        for fut in as_completed(futs):
            idx, hints = fut.result()
            q = strip_question(df.iloc[idx]["prompt"][0]["content"])
            row_out = {"index": int(idx), "question": q, "hints": hints}
            line = json.dumps(row_out, ensure_ascii=False)
            with write_lock:
                fout.write(line + "\n")
                fout.flush()
            done_count += 1
            if any("_error" in h or "_raw" in h for h in hints):
                err_count += 1
            if done_count % 50 == 0 or done_count == len(remaining):
                elapsed = time.time() - t0
                rate = done_count / max(elapsed, 1e-9)
                eta = (len(remaining) - done_count) / max(rate, 1e-9)
                print(f"[{done_count}/{len(remaining)}] {rate:.2f} rows/s | eta {eta/60:.1f} min | errors so far: {err_count}")

    print(f"[OK] wrote {out_path} | total done {done_count} | errors {err_count}")


if __name__ == "__main__":
    main()
