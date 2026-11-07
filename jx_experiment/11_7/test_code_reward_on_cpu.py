#!/usr/bin/env python3
"""Local tests for the code reward harness. No GPU, no cluster, no HF download.

    python3 jx_experiment/11_7/test_code_reward_on_cpu.py

Covers the three spec kinds and the failure modes that would otherwise show up
as a silently-zero benchmark on the wandb curve: a hang, a crash, a solution
that prints nothing, a solution wrapped in prose, and a solution that tries to
suppress its own verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from code_reward import compute_score, compute_score_batch, extract_code  # noqa: E402

FAILS: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
        FAILS.append(name)


def score(completion: str, spec: dict) -> float:
    return compute_score("t", completion, json.dumps(spec))["acc"]


def cases(completion: str, spec: dict) -> float:
    return compute_score("t", completion, json.dumps(spec))["case_pass_rate"]


# ---------------------------------------------------------------- extract_code
print("extract_code")
check("fenced", extract_code("blah\n```python\nx=1\n```\n").strip(), "x=1")
check(
    "last block wins",
    extract_code("```python\nwrong=1\n```\ntext\n```python\nright=2\n```").strip(),
    "right=2",
)
check("unterminated fence", extract_code("talk\n```python\nx=1\nx+=1").strip(), "x=1\nx+=1")
check("bare code", extract_code("def f():\n    return 1").strip(), "def f():\n    return 1")

# ------------------------------------------------------------------- evalplus
print("evalplus (HumanEval+ / MBPP+)")
EP_CHECK = {
    "type": "evalplus",
    "entry_point": "add",
    "test_code": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
}
check("pass", score("```python\ndef add(a,b):\n    return a+b\n```", EP_CHECK), 1.0)
check("wrong answer", score("```python\ndef add(a,b):\n    return a-b\n```", EP_CHECK), 0.0)
check("missing entry point", score("```python\ndef sub(a,b):\n    return a-b\n```", EP_CHECK), 0.0)
check("syntax error", score("```python\ndef add(a,b)\n    return a+b\n```", EP_CHECK), 0.0)
check("empty completion", score("I don't know.", EP_CHECK), 0.0)

EP_ASSERTS = {
    "type": "evalplus",
    "entry_point": "mul",
    "test_code": "assert mul(2, 3) == 6\nassert mul(0, 5) == 0\n",
}
check("flat asserts pass", score("```python\ndef mul(a,b):\n    return a*b\n```", EP_ASSERTS), 1.0)
check("flat asserts fail", score("```python\ndef mul(a,b):\n    return a+b\n```", EP_ASSERTS), 0.0)

EP_IMPORTS = {
    "type": "evalplus",
    "entry_point": "mean",
    "test_imports": ["import math"],
    "test_code": "assert math.isclose(mean([1,2,3]), 2.0)\n",
}
check(
    "test_imports honored",
    score("```python\ndef mean(xs):\n    return sum(xs)/len(xs)\n```", EP_IMPORTS),
    1.0,
)

# ----------------------------------------------------------------- functional
print("functional (LiveCodeBench call-based)")
FN = {
    "type": "functional",
    "fn_name": "twoSum",
    "inputs": [[[2, 7, 11, 15], 9], [[3, 2, 4], 6]],
    "outputs": [[0, 1], [1, 2]],
}
SOL_CLASS = """```python
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, v in enumerate(nums):
            if target - v in seen:
                return [seen[target - v], i]
            seen[v] = i
```"""
check("class Solution", score(SOL_CLASS, FN), 1.0)
SOL_BARE = """```python
def twoSum(nums, target):
    seen = {}
    for i, v in enumerate(nums):
        if target - v in seen:
            return [seen[target - v], i]
        seen[v] = i
```"""
check("bare function fallback", score(SOL_BARE, FN), 1.0)
check(
    "tuple vs list equivalent",
    score("```python\ndef twoSum(nums, target):\n    return (0,1) if target==9 else (1,2)\n```", FN),
    1.0,
)
PARTIAL = "```python\ndef twoSum(nums, target):\n    return [0, 1]\n```"
check("partial is not a pass", score(PARTIAL, FN), 0.0)
check("partial case_pass_rate", cases(PARTIAL, FN), 0.5)

FN_FLOAT = {"type": "functional", "fn_name": "f", "inputs": [[1]], "outputs": [0.3333333333]}
check(
    "float tolerance",
    score("```python\ndef f(n):\n    return n/3\n```", FN_FLOAT),
    1.0,
)

# ---------------------------------------------------------------------- stdio
print("stdio (LiveCodeBench stdin/stdout)")
IO = {"type": "stdio", "inputs": ["3\n1 2 3\n", "1\n5\n"], "outputs": ["6\n", "5\n"]}
SOL_IO = """```python
n = int(input())
print(sum(map(int, input().split())))
```"""
check("input() based", score(SOL_IO, IO), 1.0)
SOL_IO_STDIN = """```python
import sys
data = sys.stdin.read().split()
print(sum(map(int, data[1:])))
```"""
check("sys.stdin based", score(SOL_IO_STDIN, IO), 1.0)
check(
    "no state leak across cases",
    score("```python\nimport sys\nn=int(input())\nprint(sum(map(int,input().split())))\n```", IO),
    1.0,
)
check("prints nothing", score("```python\nx = 1\n```", IO), 0.0)
check(
    "trailing whitespace tolerated",
    score("```python\nn=int(input())\nprint(str(sum(map(int,input().split())))+' ')\n```", IO),
    1.0,
)
check(
    "sys.exit still scored",
    score(
        "```python\nimport sys\nn=int(input())\nprint(sum(map(int,input().split())))\nsys.exit(0)\n```",
        IO,
    ),
    1.0,
)

# ------------------------------------------------------- containment / hygiene
print("containment")
check("infinite loop is killed", score("```python\nwhile True:\n    pass\n```", IO), 0.0)
check(
    "exception inside solution",
    score("```python\nraise RuntimeError('boom')\n```", IO),
    0.0,
)
check(
    "closing stdout cannot hide the verdict",
    score("```python\nimport sys, os\nos.close(1)\n```", IO),
    0.0,
)
check(
    "os.remove is disabled",
    score("```python\nimport os\nos.remove('/tmp/nonexistent')\n```", IO),
    0.0,
)

print("per-sample interface (reward_loop naive)")
one = compute_score(
    data_source="humanevalplus",
    solution_str="```python\ndef add(a,b):\n    return a+b\n```",
    ground_truth=json.dumps(EP_CHECK),
    extra_info={"task_id": "x"},
)
check("kwargs call", sorted(one.keys()), ["acc", "case_pass_rate", "score"])
check("kwargs acc", one["acc"], 1.0)
check("no err column", "err" in one, False)

print("batch interface (legacy/offline)")
batch = compute_score_batch(
    ["humanevalplus", "livecodebench", "humanevalplus"],
    ["```python\ndef add(a,b):\n    return a+b\n```", SOL_IO, "nope"],
    [json.dumps(EP_CHECK), json.dumps(IO), json.dumps(EP_CHECK)],
)
check("batch length", len(batch), 3)
check("batch accs", [r["acc"] for r in batch], [1.0, 1.0, 0.0])
check("no string columns returned", any("err" in r for r in batch), False)
check("keys", sorted(batch[0].keys()), ["acc", "case_pass_rate", "score"])

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all passed")
