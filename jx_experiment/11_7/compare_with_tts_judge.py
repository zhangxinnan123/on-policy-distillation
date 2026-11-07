#!/usr/bin/env python3
"""Re-judge verl's dumped validation generations with opd-test-time-scaling's
own judges, and compare per sample with the online reward (`acc`).

    python3 jx_experiment/11_7/compare_with_tts_judge.py \
        --dump logs/val_dump/valonly_tts50 \
        --eval-parquet ~/data/code_opd_tts/test_tts_50.parquet \
        --tts-repo /projects/standard/mhong/zhan9359/work/opd-test-time-scaling

Their side, unchanged code from their repo:
  * response -> code: HumanEval+/raw_to_samples.py `evalplus_sample` (chat mode,
    first block + header prefix), MBPP+/raw_to_samples.py, LiveCodeBench/
    raw_to_samples.py `code_text` (last block);
  * judges: vendor/evalplus/evaluate.py (base AND plus must pass) and
    LiveCodeBench/judge_chunk.py (lcb_runner codegen_metrics, timeout 6).
They normally run inside a root namespace sandbox; here they run as plain
subprocesses (no root on MSI), which is fine for a few hundred samples from a
public model but is NOT a security boundary.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--eval-parquet", required=True, type=Path)
    ap.add_argument("--tts-repo", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import pandas as pd

    bench = args.tts_repo / "evaluation/benchmarks"
    df = pd.read_parquet(args.eval_parquet)
    # Match a dumped `input` (chat-templated prompt text) to its eval row by the
    # last user message, which is unique per problem.
    rows = [(r.data_source, r.extra_info["task_id"], r.prompt[-1]["content"]) for r in df.itertuples()]

    dumps = sorted(args.dump.glob("*.jsonl"))
    if not dumps:
        raise SystemExit(f"no *.jsonl in {args.dump}")
    gens = read_jsonl(dumps[0])
    samples = []  # (data_source, task_id, sample_idx, response, our_acc)
    counter: Counter = Counter()
    for g in gens:
        hit = [(ds, tid) for ds, tid, user in rows if user.strip()[:400] in g["input"]]
        if len(hit) != 1:
            raise SystemExit(f"could not map a dumped input to exactly one problem ({len(hit)} matches)")
        ds, tid = hit[0]
        samples.append((ds, tid, counter[tid], g["output"], float(g.get("acc", g.get("score", 0.0)))))
        counter[tid] += 1
    print(f"{dumps[0].name}: {len(samples)} generations over {len(counter)} problems")

    theirs: dict[tuple[str, int], bool] = {}
    with tempfile.TemporaryDirectory(prefix="tts-judge-") as tmp:
        tmp = Path(tmp)
        # ---- HumanEval+ / MBPP+ -------------------------------------------
        for ds, bdir, ev_name, canon_name, env_key in [
            ("humanevalplus", "HumanEval+", "humaneval", "humaneval_plus.jsonl", "HUMANEVAL_OVERRIDE_PATH"),
            ("mbppplus", "MBPP+", "mbpp", "mbpp_plus.jsonl", "MBPP_OVERRIDE_PATH"),
        ]:
            sub = [s for s in samples if s[0] == ds]
            if not sub:
                continue
            conv = load_module(bench / bdir / "raw_to_samples.py", f"conv_{ev_name}")
            canon = {str(r["id"]): r for r in read_jsonl(bench / bdir / "data/canonical" / canon_name)}
            src_rows = read_jsonl(next((bench / bdir / "data/source").glob("*.jsonl")))
            want = {tid for _, tid, *_ in sub}
            src_path = tmp / f"{ev_name}_source.jsonl"
            src_path.write_text("".join(json.dumps(r) + "\n" for r in src_rows if str(r["task_id"]) in want))
            smp_path = tmp / f"{ev_name}_samples.jsonl"
            with smp_path.open("w") as fh:
                for _, tid, idx, out, _ in sub:
                    row = {"problem_id": tid, "input_mode": "chat", "final_text": out, "problem": canon[tid]["problem"]}
                    fh.write(json.dumps(conv.evalplus_sample(row)) + "\n")
            res_path = tmp / f"{ev_name}_results.json"
            env = {**os.environ, env_key: str(src_path), "EVALPLUS_CACHE_DIR": str(tmp / "evalplus-cache"),
                   "PYTHONPATH": str(bench / bdir / "vendor/evalplus")}
            subprocess.run(
                [sys.executable, str(bench / bdir / "vendor/evalplus/evalplus/evaluate.py"), "--dataset", ev_name,
                 "--samples", str(smp_path), "--output-file", str(res_path), "--parallel", str(args.workers)],
                env=env, cwd=tmp, check=True, stdout=subprocess.DEVNULL,
            )
            res = json.loads(res_path.read_text())["eval"]
            for tid, items in res.items():
                # Their evaluator keeps completion_id order == sample order.
                for item in sorted(items, key=lambda x: x.get("completion_id", 0)):
                    ok = item["base_status"] == "pass" and item["plus_status"] == "pass"
                    theirs[(tid, int(item.get("completion_id", 0)))] = ok

        # ---- LiveCodeBench --------------------------------------------------
        sub = [s for s in samples if s[0] == "livecodebench"]
        if sub:
            lcb = bench / "LiveCodeBench"
            conv = load_module(lcb / "raw_to_samples.py", "conv_lcb")
            from huggingface_hub import hf_hub_download

            man = json.loads((lcb / "data/manifest.json").read_text())
            raw = Path(hf_hub_download(man["dataset"], "test6.jsonl", repo_type="dataset", revision=man["revision"]))
            want = {tid for _, tid, *_ in sub}
            src_path = tmp / "lcb_source.jsonl"
            src_path.write_text("".join(x + "\n" for x in raw.read_text().splitlines() if x.strip() and json.loads(x)["question_id"] in want))
            by_q = defaultdict(list)
            for _, tid, idx, out, _ in sorted(sub, key=lambda s: (s[1], s[2])):
                by_q[tid].append(conv.code_text(out))
            cand_path = tmp / "lcb_candidates.json"
            cand_path.write_text(json.dumps([{"question_id": q, "code_list": c} for q, c in by_q.items()]))
            out_path = tmp / "lcb_graded.jsonl"
            subprocess.run(
                [sys.executable, str(lcb / "judge_chunk.py"), "--source", str(src_path), "--candidates", str(cand_path),
                 "--output", str(out_path), "--workers", str(args.workers), "--timeout", "6"],
                env={**os.environ, "PYTHONPATH": str(lcb / "vendor")}, cwd=tmp, check=True, stdout=subprocess.DEVNULL,
            )
            for r in read_jsonl(out_path):
                for i, ok in enumerate(r["correctness"]):
                    theirs[(r["question_id"], i)] = bool(ok)

    # ---- compare ------------------------------------------------------------
    per = defaultdict(lambda: Counter())
    mism = []
    for ds, tid, idx, out, ours in samples:
        t = theirs.get((tid, idx))
        if t is None:
            per[ds]["missing"] += 1
            continue
        o = ours >= 1.0
        per[ds]["n"] += 1
        per[ds]["ours"] += o
        per[ds]["theirs"] += t
        per[ds]["agree"] += o == t
        if o != t:
            mism.append((ds, tid, idx, o, t))
    print(f"\n{'benchmark':14s} {'n':>4s} {'ours acc':>9s} {'theirs acc':>10s} {'agree':>7s}")
    tot = Counter()
    for ds, c in per.items():
        tot.update(c)
        print(f"{ds:14s} {c['n']:4d} {c['ours']/c['n']:9.3f} {c['theirs']/c['n']:10.3f} {c['agree']/c['n']:7.3f}" + (f"  missing={c['missing']}" if c["missing"] else ""))
    print(f"{'ALL':14s} {tot['n']:4d} {tot['ours']/tot['n']:9.3f} {tot['theirs']/tot['n']:10.3f} {tot['agree']/tot['n']:7.3f}")
    print(f"\n{len(mism)} disagreements (ds, task, sample, ours, theirs):")
    for m in mism[:30]:
        print("  ", m)
    (args.dump / "tts_judge_comparison.json").write_text(json.dumps({"per_benchmark": {k: dict(v) for k, v in per.items()}, "disagreements": mism}, indent=1))


if __name__ == "__main__":
    main()
