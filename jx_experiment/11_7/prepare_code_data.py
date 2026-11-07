#!/usr/bin/env python3
"""Build the CodeR1-12k train parquet and the HumanEval+/MBPP+/LCB eval parquet.

Run on poe-skywalker (it downloads from HuggingFace, which the cluster reaches):

    conda activate verl
    cd /fsx/xinnanzh/on-policy-distillation
    python3 jx_experiment/11_7/prepare_code_data.py --inspect        # schemas only
    python3 jx_experiment/11_7/prepare_code_data.py --out ~/data/code_opd

Writes
    <out>/train.parquet   CodeR1-12k, prompts passed through verbatim
    <out>/test.parquet    humanevalplus + mbppplus + livecodebench
    <out>/MANIFEST.json   counts, drops, versions, and every caveat that applies

Every field mapping below was verified against the live datasets on 2026-11-07
(see MANIFEST.json "verified_on"). Three of them are NOT what the published
schema descriptions imply, and each would have produced a plausible-looking but
meaningless benchmark:

  * MBPP+ ships no `entry_point` and its harness defines no `check()`. It is a
    flat script that calls the function by name, and its `prompt` is bare natural
    language with NO signature. The signature has to come from `test_list[0]`
    or every sample fails for reasons unrelated to coding ability.
  * LiveCodeBench cannot be loaded with `load_dataset`: datasets>=5 dropped
    loading scripts and `code_generation_lite` is script-based. The raw jsonl
    shards are fetched directly instead.
  * CodeR1-12k is already in verl format and carries its OWN prompt template (a
    system message demanding <think>/<answer> tags). The eval prompts must reuse
    that exact template, or the benchmark measures format transfer rather than
    the distillation signal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

VERIFIED_ON = "2026-11-07"

# LCB release -> jsonl shards, mirroring livecodebench/code_generation_lite's own
# loading script. Reproduced here because that script no longer runs.
LCB_VERSIONS = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
    "v4_v5": ["test4.jsonl", "test5.jsonl"],
    # Single-shard subsets. `release_vN` is CUMULATIVE (release_v6 = test1..test6,
    # ~1055 problems); these are just the problems that shard N added, which is
    # the usual way to evaluate on a contamination-free window.
    #   v5_only  167 problems, contests 2024-09-22 .. 2025-01-04
    #   v6_only  175 problems, contests 2025-01-04 .. 2025-04-06
    "v5_only": ["test5.jsonl"],
    "v6_only": ["test6.jsonl"],
}


def _inspect_dump(name: str, n: int, example: dict) -> None:
    print(f"\n=== {name}  n={n} ===")
    for k, v in example.items():
        s = repr(v)
        print(f"  {k}: {type(v).__name__} = {s[:260]}{'...' if len(s) > 260 else ''}")


def _require(example: dict, keys: list[str], name: str) -> None:
    missing = [k for k in keys if k not in example]
    if missing:
        raise KeyError(
            f"{name}: expected field(s) {missing} not present.\n"
            f"  actual fields: {sorted(example.keys())}\n"
            f"The dataset changed since {VERIFIED_ON}. Re-run with --inspect and fix the "
            f"mapping; do not guess, a wrong field here scores nonsense silently."
        )


# ---------------------------------------------------------------------------
# Train: CodeR1-12k  (prompts passed through; this also DEFINES the eval format)
# ---------------------------------------------------------------------------
def build_train(
    dataset_id: str, split: str, prompt_mode: str, inspect: bool
) -> tuple[list[dict], list[dict]]:
    """Return (rows, prompt_prefix).

    ``prompt_prefix`` is whatever leading non-user messages the eval prompts must
    also carry, so train and eval cannot drift apart. What goes in it depends on
    prompt_mode:

      nonthink (default)
          CodeR1's system message is DROPPED. That message instructs the model to
          reason inside <think>...</think> and answer inside <answer>...</answer>,
          which is incompatible with running the Qwen3 template at
          enable_thinking=False -- the template emits an already-closed
          "<think>\\n\\n</think>" before the model writes a token. Dropping it
          leaves CodeR1's user message, which already opens with "Solve the
          programming task below in a Python markdown code block" and is exactly
          the instruction this harness's code extraction expects. This matches
          how every math arm in this project runs.

      coder1
          CodeR1's system message is kept verbatim. Only use this together with
          enable_thinking=True in the launch config.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    first = ds[0]
    if inspect:
        _inspect_dump(f"{dataset_id} [{split}]", len(ds), first)
        return [], []

    _require(first, ["prompt", "reward_model", "extra_info"], dataset_id)
    if not isinstance(first["prompt"], list):
        raise TypeError(f"{dataset_id}: expected a verl chat-list prompt, got {type(first['prompt'])}")

    if prompt_mode == "coder1":
        prompt_prefix = [dict(m) for m in first["prompt"] if m.get("role") != "user"]
        print(f"[train] {dataset_id}: n={len(ds)}; keeping CodeR1's own <think>/<answer> system prompt")
        for m in prompt_prefix:
            print(f"[train]   prefix role={m['role']!r}: {m['content'][:110]}...")
    else:
        prompt_prefix = []
        dropped = [m.get("role") for m in first["prompt"] if m.get("role") != "user"]
        print(
            f"[train] {dataset_id}: n={len(ds)}; nonthink mode -- dropping CodeR1 "
            f"non-user message(s) {dropped}, keeping the user turn verbatim"
        )

    rows = []
    for idx, ex in enumerate(ds):
        # CodeR1's ground truth comes in two shapes:
        #   {"functional": "<test program>"}   LeetCodeDataset, ~1/3
        #   {"inputs": [...], "outputs": [...]} TACO-verified, ~2/3
        # Both are translated into this harness's spec vocabulary so a held-out
        # CodeR1 split or task rewards can be turned on later without a rebuild.
        # Training itself runs with use_task_rewards=False and never reads this.
        spec: dict = {}
        try:
            gt = json.loads(ex["reward_model"]["ground_truth"])
            if "functional" in gt:
                spec = {"type": "evalplus", "entry_point": None, "test_code": gt["functional"]}
            elif "inputs" in gt and "outputs" in gt:
                spec = {"type": "stdio", "inputs": gt["inputs"], "outputs": gt["outputs"]}
        except (TypeError, ValueError, KeyError):
            spec = {}

        if prompt_mode == "coder1":
            prompt = [dict(m) for m in ex["prompt"]]
        else:
            prompt = [dict(m) for m in ex["prompt"] if m.get("role") == "user"]

        rows.append(
            {
                "data_source": "code_r1_12k",
                "prompt": prompt,
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(spec) if spec else ""},
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "origin": str((ex.get("extra_info") or {}).get("dataset", "")),
                },
            }
        )
    return rows, prompt_prefix


def _wrap(prefix: list[dict], body: str) -> list[dict]:
    return [*[dict(m) for m in prefix], {"role": "user", "content": body}]


# ---------------------------------------------------------------------------
# Eval: HumanEval+
# ---------------------------------------------------------------------------
def build_humaneval(dataset_id: str, prefix: list[dict], inspect: bool) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="test")
    first = ds[0]
    if inspect:
        _inspect_dump(dataset_id, len(ds), first)
        return []

    _require(first, ["prompt", "test", "entry_point", "task_id"], dataset_id)
    if "def check(" not in first["test"]:
        raise ValueError(
            f"{dataset_id}: harness no longer defines check(candidate). "
            "code_exec_runner._run_evalplus dispatches on that; re-verify before building."
        )

    rows = []
    for idx, ex in enumerate(ds):
        # The prompt is a signature plus docstring. Asking for a complete runnable
        # block (rather than a completion of the stub) matches how CodeR1 phrases
        # its task and how extract_code reads the answer back.
        body = (
            "Solve the programming task below in a Python markdown code block.\n"
            "Complete the function. Include the full function definition and any "
            "imports it needs.\n\n"
            f"```python\n{ex['prompt'].strip()}\n```"
        )
        spec = {
            "type": "evalplus",
            "entry_point": ex["entry_point"],
            "test_code": ex["test"],
        }
        rows.append(
            {
                "data_source": "humanevalplus",
                "prompt": _wrap(prefix, body),
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(spec)},
                "extra_info": {"split": "test", "index": idx, "task_id": str(ex["task_id"])},
            }
        )
    print(f"[eval] humanevalplus: n={len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# Eval: MBPP+
# ---------------------------------------------------------------------------
_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)", re.MULTILINE)


def build_mbpp(dataset_id: str, prefix: list[dict], inspect: bool) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="test")
    first = ds[0]
    if inspect:
        _inspect_dump(dataset_id, len(ds), first)
        return []

    _require(first, ["prompt", "test", "test_list", "code", "task_id"], dataset_id)
    if "def check(" in first["test"]:
        raise ValueError(
            f"{dataset_id}: harness now defines check(candidate), which it did not on "
            f"{VERIFIED_ON}. _run_evalplus would then need an entry_point; re-verify."
        )

    rows, n_nosig = [], 0
    for idx, ex in enumerate(ds):
        # MBPP+'s prompt is bare NL with no signature, and its harness calls the
        # function by name. Without a signature hint the model has no way to
        # guess the name, so every sample would fail on a NameError that has
        # nothing to do with its coding ability. test_list[0] is the standard
        # MBPP way to convey it.
        asserts = list(ex["test_list"] or [])
        if not asserts:
            n_nosig += 1
            continue
        body = (
            "Solve the programming task below in a Python markdown code block.\n\n"
            f"{ex['prompt'].strip()}\n\n"
            "Your function must satisfy this test, which also fixes its name and signature:\n"
            f"```python\n{asserts[0].strip()}\n```"
        )
        spec = {
            "type": "evalplus",
            # Flat-assert harness: no check() to call, so no entry point is
            # needed. Recorded anyway for debugging which function was expected.
            "entry_point": None,
            "test_code": ex["test"],
            "expected_fn": (_DEF_RE.search(ex["code"] or "") or [None, None])[1]
            if _DEF_RE.search(ex["code"] or "")
            else None,
        }
        if ex.get("test_imports"):
            imports = ex["test_imports"]
            spec["test_imports"] = imports if isinstance(imports, list) else [imports]
        rows.append(
            {
                "data_source": "mbppplus",
                "prompt": _wrap(prefix, body),
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(spec)},
                "extra_info": {"split": "test", "index": idx, "task_id": str(ex["task_id"])},
            }
        )
    print(f"[eval] mbppplus: n={len(rows)}" + (f" (skipped {n_nosig} with no test_list)" if n_nosig else ""))
    return rows


# ---------------------------------------------------------------------------
# Eval: LiveCodeBench
# ---------------------------------------------------------------------------
def _lcb_private(blob: str, allow_pickle: bool) -> list:
    """Decode private_test_cases, which is base64(zlib(pickle(json))).

    Unpickling an artifact downloaded from the internet executes arbitrary code,
    so this is gated. The public-tests-only default needs no such trust.
    """
    try:
        return json.loads(blob)
    except (TypeError, ValueError):
        pass
    if not allow_pickle:
        return []
    import base64
    import pickle
    import zlib

    return json.loads(pickle.loads(zlib.decompress(base64.b64decode(blob.encode("utf-8")))))


def build_lcb(
    repo: str,
    version: str,
    prefix: list[dict],
    tests_mode: str,
    max_cases: int,
    limit: int,
    inspect: bool,
) -> list[dict]:
    from huggingface_hub import hf_hub_download

    if version not in LCB_VERSIONS:
        raise ValueError(f"unknown LCB version {version!r}; known: {sorted(LCB_VERSIONS)}")

    records = []
    for shard in LCB_VERSIONS[version]:
        path = hf_hub_download(repo, shard, repo_type="dataset")
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if inspect:
        _inspect_dump(f"{repo} [{version}]", len(records), records[0])
        return []

    _require(records[0], ["question_content", "public_test_cases", "private_test_cases"], repo)

    rows: list[dict] = []
    n_no_tests = n_bad_functional = 0
    for idx, ex in enumerate(records):
        try:
            tests = list(json.loads(ex["public_test_cases"]))
        except (TypeError, ValueError):
            tests = []
        if tests_mode == "all":
            tests += list(_lcb_private(ex["private_test_cases"], allow_pickle=True))
        if not tests:
            n_no_tests += 1
            continue

        meta = {}
        if ex.get("metadata"):
            try:
                meta = json.loads(ex["metadata"])
            except (TypeError, ValueError):
                meta = {}
        fn_name = meta.get("func_name")
        testtype = str(tests[0].get("testtype") or "stdin").lower()

        # Cap cases per problem. Truncation can only make a solution look better,
        # never worse, so the result is an upper bound -- recorded in the manifest.
        if max_cases:
            tests = tests[:max_cases]

        if testtype == "functional" or fn_name:
            if not fn_name:
                n_bad_functional += 1
                continue
            try:
                # LCB's own convention: each arg is a JSON value on its own line.
                inputs = [[json.loads(ln) for ln in t["input"].split("\n") if ln.strip()] for t in tests]
                outputs = [json.loads(t["output"]) for t in tests]
            except (TypeError, ValueError):
                n_bad_functional += 1
                continue
            spec = {"type": "functional", "fn_name": fn_name, "inputs": inputs, "outputs": outputs}
        else:
            spec = {
                "type": "stdio",
                "inputs": [t["input"] for t in tests],
                "outputs": [t["output"] for t in tests],
            }

        body = "Solve the programming task below in a Python markdown code block.\n\n" + ex[
            "question_content"
        ].strip()
        starter = (ex.get("starter_code") or "").strip()
        if starter:
            body += f"\n\nUse this exact signature:\n```python\n{starter}\n```"

        rows.append(
            {
                "data_source": "livecodebench",
                "prompt": _wrap(prefix, body),
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(spec)},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "task_id": str(ex.get("question_id", idx)),
                    "testtype": spec["type"],
                    "difficulty": str(ex.get("difficulty", "")),
                    "platform": str(ex.get("platform", "")),
                },
            }
        )

    print(
        f"[eval] livecodebench[{version}]: raw={len(records)} usable={len(rows)} "
        f"no_tests={n_no_tests} unusable_functional={n_bad_functional} tests={tests_mode}"
    )
    if limit and len(rows) > limit:
        stride = len(rows) / limit
        rows = [rows[int(i * stride)] for i in range(limit)]
        print(f"[eval] livecodebench: strided down to {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
def filter_overlong(rows: list[dict], tok, max_prompt: int) -> tuple[list[dict], list]:
    """Drop prompts over max_prompt tokens and report exactly which.

    Done at build time rather than left to `data.filter_overlong_prompts` so the
    eval denominator is fixed in the manifest. Otherwise two runs that differ in
    nothing could silently score over different numbers of problems.
    """
    kept, dropped = [], []
    for r in rows:
        ids = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True)
        if len(ids) <= max_prompt:
            kept.append(r)
        else:
            dropped.append(
                {
                    "data_source": r["data_source"],
                    "task_id": r["extra_info"].get("task_id"),
                    "n_tokens": len(ids),
                }
            )
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/data/code_opd"))
    ap.add_argument("--train-dataset", default="ganler/code-r1-12k")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--humaneval-dataset", default="evalplus/humanevalplus")
    ap.add_argument("--mbpp-dataset", default="evalplus/mbppplus")
    ap.add_argument("--lcb-repo", default="livecodebench/code_generation_lite")
    ap.add_argument("--lcb-version", default="v6_only", choices=sorted(LCB_VERSIONS))
    ap.add_argument(
        "--lcb-tests",
        default="public",
        choices=["public", "all"],
        help="'all' additionally unpickles private_test_cases (arbitrary code execution)",
    )
    ap.add_argument("--lcb-max-cases", type=int, default=25, help="0 = no cap")
    ap.add_argument("--lcb-limit", type=int, default=0, help="0 = keep all")
    ap.add_argument("--skip-lcb", action="store_true")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument(
        "--prompt-mode",
        default="nonthink",
        choices=["nonthink", "coder1"],
        help="nonthink (default) drops CodeR1's <think>/<answer> system prompt; "
        "use with enable_thinking=False. coder1 keeps it; use with enable_thinking=True.",
    )
    ap.add_argument("--inspect", action="store_true", help="print real schemas and exit")
    args = ap.parse_args()

    if args.inspect:
        steps = [
            lambda: build_train(args.train_dataset, args.train_split, args.prompt_mode, True),
            lambda: build_humaneval(args.humaneval_dataset, [], True),
            lambda: build_mbpp(args.mbpp_dataset, [], True),
            lambda: build_lcb(args.lcb_repo, args.lcb_version, [], "public", 0, 0, True),
        ]
        for fn in steps:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - inspection must not stop at the first miss
                print(f"  !! {type(exc).__name__}: {exc}")
        print("\n--inspect done; nothing written.")
        return 0

    import pandas as pd
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, prefix = build_train(args.train_dataset, args.train_split, args.prompt_mode, False)

    test = build_humaneval(args.humaneval_dataset, prefix, False)
    test += build_mbpp(args.mbpp_dataset, prefix, False)
    if not args.skip_lcb:
        test += build_lcb(
            args.lcb_repo,
            args.lcb_version,
            prefix,
            args.lcb_tests,
            args.lcb_max_cases,
            args.lcb_limit,
            False,
        )

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train, train_dropped = filter_overlong(train, tok, args.max_prompt)
    test, test_dropped = filter_overlong(test, tok, args.max_prompt)

    for i, r in enumerate(train):
        r["extra_info"]["index"] = i
    for i, r in enumerate(test):
        r["extra_info"]["index"] = i

    pd.DataFrame(train).to_parquet(out / "train.parquet")
    pd.DataFrame(test).to_parquet(out / "test.parquet")

    counts: dict[str, int] = {}
    drops: dict[str, int] = {}
    for r in test:
        counts[r["data_source"]] = counts.get(r["data_source"], 0) + 1
    for d in test_dropped:
        drops[d["data_source"]] = drops.get(d["data_source"], 0) + 1

    caveats = [
        (
            "prompt_mode=nonthink: CodeR1's <think>/<answer> system prompt was stripped "
            "from train AND eval prompts. Launch with enable_thinking=False. Scores are "
            "not comparable to any run built with prompt_mode=coder1."
            if args.prompt_mode == "nonthink"
            else "prompt_mode=coder1: CodeR1's <think>/<answer> system prompt is present. "
            "Launch with enable_thinking=True or the model is told to fill a pre-closed "
            "think block."
        ),
        "Numbers from this harness are for comparing the four methods against each "
        "other. They are NOT the official EvalPlus / LiveCodeBench figures and must "
        "not be put in a table next to published ones.",
        "Overlong prompts were dropped at build time so the denominator is fixed "
        "across runs. See eval.dropped_by_source.",
        "acc is strict all-tests-pass per sample. case_pass_rate is logged alongside "
        "as a progress signal and is NOT a benchmark number.",
    ]
    if args.lcb_tests == "public":
        caveats.append(
            "LiveCodeBench used PUBLIC test cases only (no unpickling of private "
            "tests). Public tests are the examples shown in the problem statement, so "
            "LCB accuracy here is substantially HIGHER than the official figure."
        )
    if args.lcb_max_cases:
        caveats.append(
            f"LiveCodeBench capped at {args.lcb_max_cases} test cases per problem; "
            "truncation can only help a solution, so LCB accuracy is an upper bound."
        )

    manifest = {
        "verified_on": VERIFIED_ON,
        "prompt_mode": args.prompt_mode,
        "prompt_prefix": prefix,
        "requires_enable_thinking": args.prompt_mode == "coder1",
        "tokenizer": args.tokenizer,
        "max_prompt": args.max_prompt,
        "train": {
            "dataset": args.train_dataset,
            "n": len(train),
            "n_dropped_overlong": len(train_dropped),
        },
        "eval": {
            "n_total": len(test),
            "counts": counts,
            "n_dropped_overlong": len(test_dropped),
            "dropped_by_source": drops,
            "dropped": test_dropped[:200],
            "humaneval_dataset": args.humaneval_dataset,
            "mbpp_dataset": args.mbpp_dataset,
            "lcb_repo": None if args.skip_lcb else args.lcb_repo,
            "lcb_version": None if args.skip_lcb else args.lcb_version,
            "lcb_tests": None if args.skip_lcb else args.lcb_tests,
            "lcb_max_cases": args.lcb_max_cases,
            "lcb_limit": args.lcb_limit,
        },
        "caveats": caveats,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {out}/train.parquet  n={len(train)}")
    print(f"wrote {out}/test.parquet   n={len(test)}  {counts}")
    print(f"wrote {out}/MANIFEST.json")
    print(f"\nval cost per validation at n=8: {len(test) * 8} generations")
    if test_dropped:
        print(f"NOTE: {len(test_dropped)} eval prompts dropped as overlong: {drops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
