#!/usr/bin/env python3
"""Build the online-eval parquet in the opd-test-time-scaling (TTS) format.

    python3 jx_experiment/11_7/prepare_code_eval_tts.py \
        --tts-repo /projects/standard/mhong/zhan9359/work/opd-test-time-scaling \
        --out ~/data/code_opd_tts

Mirrors https://github.com/Geraldxm/opd-test-time-scaling evaluation/benchmarks:

  prompts   their frozen canonical jsonl (id, problem) rendered through their
            prompt files exactly as math_eval does (`{{problem}}` substituted
            into each <idx>-<role>.txt, file bytes kept verbatim):
              HumanEval+ / MBPP+  user: "Solve the following programming problem.
                                  Return only the complete Python solution,
                                  without Markdown fences. {{problem}}"
              LiveCodeBench v6    system: "You are an expert Python programmer..."
                                  user: problem incl. "### Format: ..." block
  extraction HumanEval+/MBPP+: FIRST complete fenced block, else the raw answer
            minus a leading "Assistant:"; imports/header lines before the prompt's
            first `def` are prepended when the solution does not start with them.
            LiveCodeBench: LAST complete fenced block, same fallback.
            (spec["extract"] / spec["prefix"], applied in code_reward.py)
  tests     HumanEval+/MBPP+: EvalPlus base + plus tests (evalplus/* on HF).
            LiveCodeBench: public + PRIVATE tests of test6.jsonl at their pinned
            revision, SHA-256 checked, judged by their vendored official
            lcb_runner.run_test (timeout 6s per test, all tests must pass).
            Private tests are base64+zlib+pickle in the official file; decoding
            them is what the official harness does, and the file is pinned by hash.

Writes <out>/test_tts.parquet, <out>/lcb_tests/<question_id>.json and
<out>/MANIFEST.json. Use subsample_eval.py --src test_tts.parquet for a subset.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pickle
import re
import zlib
from pathlib import Path

PROBLEM_MARKER = "{{problem}}"
MESSAGE_FILE = re.compile(r"(\d+)-(system|user|assistant)\.txt")


def render(prompt_dir: Path, problem: str) -> list[dict]:
    """math_eval.inference.load_prompt + render_prompt for input_mode=chat."""
    indexed = []
    for item in prompt_dir.iterdir():
        m = MESSAGE_FILE.fullmatch(item.name)
        if not m:
            raise ValueError(f"unexpected prompt file {item}")
        indexed.append((int(m.group(1)), m.group(2), item.read_bytes().decode("utf-8")))
    return [{"role": r, "content": c.replace(PROBLEM_MARKER, problem)} for _, r, c in sorted(indexed)]


def problem_import_prefix(problem_text: str) -> str:
    """Verbatim from their HumanEval+/MBPP+ raw_to_samples.py."""
    lines = problem_text.splitlines()
    first_function = next((i for i, line in enumerate(lines) if line.startswith("def ")), None)
    return "" if first_function is None else "\n".join(lines[:first_function]) + "\n\n"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_evalplus(bench_dir: Path, hf_id: str, data_source: str) -> list[dict]:
    from datasets import load_dataset

    meta = json.loads(next((bench_dir / "data/canonical").glob("*.metadata.json")).read_text())
    canon_path = next(p for p in (bench_dir / "data/canonical").glob("*.jsonl"))
    want = meta.get("canonical_sha256") or meta.get("canonical", {}).get("sha256")
    if want and sha256(canon_path) != want:
        raise ValueError(f"{canon_path}: SHA-256 mismatch with their metadata")
    canon = read_jsonl(canon_path)

    hf = {}
    for ex in load_dataset(hf_id, split="test"):
        tid = str(ex["task_id"])
        hf[tid if "/" in tid else f"Mbpp/{tid}"] = ex

    rows = []
    for row in canon:
        pid = str(row["id"])
        ex = hf[pid]
        problem = str(row["problem"])
        spec = {
            "type": "evalplus",
            "entry_point": ex.get("entry_point") if "def check(" in ex["test"] else None,
            "test_code": ex["test"],
            "extract": "first",
            "prefix": problem_import_prefix(problem),
        }
        if data_source == "mbppplus":
            m = re.search(r"^\s*def\s+([A-Za-z_]\w*)", ex.get("code") or "", re.MULTILINE)
            spec["expected_fn"] = m.group(1) if m else None
            if ex.get("test_imports"):
                imps = ex["test_imports"]
                spec["test_imports"] = list(imps) if not isinstance(imps, str) else [imps]
        rows.append(
            {
                "data_source": data_source,
                "prompt": render(bench_dir / "prompts/code-chat", problem),
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(spec)},
                "extra_info": {"split": "test", "index": 0, "task_id": pid},
            }
        )
    print(f"[eval] {data_source}: n={len(rows)}")
    return rows


def build_lcb(bench_dir: Path, out: Path) -> list[dict]:
    from huggingface_hub import hf_hub_download

    manifest = json.loads((bench_dir / "data/manifest.json").read_text())
    shard = "test6.jsonl"
    src = Path(hf_hub_download(manifest["dataset"], shard, repo_type="dataset", revision=manifest["revision"]))
    if sha256(src) != manifest["source_files"][shard]["sha256"]:
        raise ValueError(f"{shard}: SHA-256 mismatch with their pinned manifest")
    raw = {str(r["question_id"]): r for r in read_jsonl(src)}

    canon_path = bench_dir / "data/canonical/livecodebench_v6_only_chat.jsonl"
    if sha256(canon_path) != manifest["canonical_files"][canon_path.name]["sha256"]:
        raise ValueError(f"{canon_path}: SHA-256 mismatch")
    tests_dir = out / "lcb_tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in read_jsonl(canon_path):
        qid = str(row["id"])
        ex = raw[qid]
        public = json.loads(ex["public_test_cases"])
        try:
            private = json.loads(ex["private_test_cases"])
        except (TypeError, ValueError):
            private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(ex["private_test_cases"].encode("utf-8")))))
        tests = public + private
        fn_name = json.loads(ex["metadata"]).get("func_name", None)
        # Exactly lcb_runner CodeGenerationProblem.get_evaluation_sample().
        sample = {
            "input_output": json.dumps(
                {"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests], "fn_name": fn_name}
            )
        }
        tpath = tests_dir / f"{qid}.json"
        tpath.write_text(json.dumps(sample))
        spec = {"type": "lcb", "tests_path": str(tpath), "n_tests": len(tests), "extract": "last"}
        rows.append(
            {
                "data_source": "livecodebench",
                "prompt": render(bench_dir / "prompts/code-chat", str(row["problem"])),
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(spec)},
                "extra_info": {
                    "split": "test",
                    "index": 0,
                    "task_id": qid,
                    "testtype": "functional" if fn_name else "stdin",
                    "difficulty": str(ex.get("difficulty", "")),
                },
            }
        )
    n_tests = [json.loads(r["reward_model"]["ground_truth"])["n_tests"] for r in rows]
    print(f"[eval] livecodebench v6-only: n={len(rows)} tests/problem min={min(n_tests)} max={max(n_tests)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-repo", required=True, type=Path)
    ap.add_argument("--out", default=os.path.expanduser("~/data/code_opd_tts"), type=Path)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-prompt", type=int, default=2048)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    bench = args.tts_repo / "evaluation/benchmarks"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build_evalplus(bench / "HumanEval+", "evalplus/humanevalplus", "humanevalplus")
    rows += build_evalplus(bench / "MBPP+", "evalplus/mbppplus", "mbppplus")
    rows += build_lcb(bench / "LiveCodeBench", args.out)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lens = {}
    for r in rows:
        n = len(tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True, enable_thinking=False))
        lens.setdefault(r["data_source"], []).append(n)
    over = {k: sum(x > args.max_prompt for x in v) for k, v in lens.items()}
    if any(over.values()):
        raise SystemExit(f"prompts over --max-prompt {args.max_prompt}: {over}; raise MAX_PROMPT, do not drop (fixed denominator)")
    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i
    pd.DataFrame(rows).to_parquet(args.out / "test_tts.parquet")

    counts = {k: len(v) for k, v in lens.items()}
    (args.out / "MANIFEST.json").write_text(
        json.dumps(
            {
                "format": "opd-test-time-scaling evaluation/benchmarks (prompts, extraction, tests)",
                "tts_repo": str(args.tts_repo),
                "counts": counts,
                "max_prompt_tokens": {k: max(v) for k, v in lens.items()},
                "differences_from_tts_offline_eval": [
                    "decoding is verl's single val_kwargs setting for all three benchmarks "
                    "(TTS: HE+/MBPP+ temp 0.7 top-p 0.95; LCB temp 1.0 top-p 0.8)",
                    "n per prompt is val_kwargs.n (TTS: 512 HE+/MBPP+, 128 LCB)",
                    "EvalPlus tests run via the HF evalplus `test` harness inside this repo's "
                    "sandbox, not the EvalPlus evaluate() runner (no per-input dynamic time limits)",
                ],
            },
            indent=1,
        )
    )
    print(f"wrote {args.out / 'test_tts.parquet'} n={len(rows)} {counts}; max prompt tokens {({k: max(v) for k, v in lens.items()})}")


if __name__ == "__main__":
    main()
