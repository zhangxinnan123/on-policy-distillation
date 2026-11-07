#!/usr/bin/env python3
"""Child process that runs ONE model-generated program against ONE test spec.

Invoked by ``code_reward.py`` as::

    python3 -I code_exec_runner.py        # payload on stdin, verdict on stdout

Reads a JSON payload ``{"code": str, "spec": {...}}`` from stdin and writes a
single JSON line ``{"passed": bool, "n_pass": int, "n_total": int, "error": str}``
to stdout.

Why a separate process rather than a thread or a ProcessPoolExecutor worker:
model-generated code segfaults, hangs, calls ``sys.exit``, closes stdout, spawns
threads and allocates without bound. Only process isolation plus an external
kill survives all of those, and only a fresh process guarantees the next sample
is not scored against state the previous one left behind. ``-I`` is required:
without it the interpreter puts this script's own directory and the CWD on
``sys.path``, so a model that emits ``json.py`` would hijack our imports.

Nothing here is trusted. This file never imports the generated code's module
namespace into its own, and the verdict is written to stdout *before* any
cleanup, so a crash during teardown still reports the measured result.
"""

from __future__ import annotations

import io
import json
import sys


# --------------------------------------------------------------------------
# Output comparison
# --------------------------------------------------------------------------
def _norm_stdout(s: str) -> list[str]:
    """Normalize program stdout to a list of stripped, non-empty-trailing lines."""
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _stdout_matches(got: str, want: str) -> bool:
    """Compare two stdout blobs the way code benchmarks conventionally do.

    Three passes, cheapest first: exact line match, then whitespace-insensitive
    token match, then numeric match with a tolerance. The numeric pass is what
    makes float-answer problems scoreable at all; the token pass covers the very
    common case of a trailing space or a different column separator.
    """
    g, w = _norm_stdout(got), _norm_stdout(want)
    if g == w:
        return True
    gt, wt = " ".join(g).split(), " ".join(w).split()
    if gt == wt:
        return True
    if len(gt) != len(wt):
        return False
    for a, b in zip(gt, wt):
        if a == b:
            continue
        try:
            if abs(float(a) - float(b)) > 1e-6 * max(1.0, abs(float(b))):
                return False
        except (ValueError, OverflowError):
            return False
    return True


def _values_match(got: object, want: object) -> bool:
    """Compare a returned Python value against an expected one.

    Tuples and lists are treated as equivalent because the two sides come from
    different serializations of the same problem (JSON gives us lists; solutions
    legitimately return tuples).
    """
    if got == want:
        return True
    if isinstance(got, (list, tuple)) and isinstance(want, (list, tuple)):
        if len(got) != len(want):
            return False
        return all(_values_match(a, b) for a, b in zip(got, want))
    if isinstance(got, float) or isinstance(want, float):
        try:
            return abs(float(got) - float(want)) <= 1e-6 * max(1.0, abs(float(want)))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


# --------------------------------------------------------------------------
# The three test-spec kinds
# --------------------------------------------------------------------------
def _run_evalplus(code: str, spec: dict) -> tuple[int, int]:
    """HumanEval+ / MBPP+ : exec the solution, then exec the test harness.

    EvalPlus ships two shapes of harness in the same field. Some records define
    ``check(candidate)`` and expect it to be called with the entry-point
    function; others are a flat sequence of ``assert`` statements that reference
    the entry point by name. Detect which by looking for the definition, because
    calling a non-existent ``check`` and exec-ing asserts twice are both silent
    wrong answers rather than errors.

    Scored all-or-nothing (1/1): the harness raises on the first failing assert,
    so partial credit is not observable without rewriting the harness.
    """
    ns: dict = {"__name__": "__solution__"}
    exec(compile(code, "<solution>", "exec"), ns)

    for imp in spec.get("test_imports") or []:
        exec(compile(imp, "<test_imports>", "exec"), ns)

    test_code = spec["test_code"]
    exec(compile(test_code, "<test>", "exec"), ns)

    if "def check(" in test_code:
        entry = spec.get("entry_point")
        if entry is None:
            raise ValueError("evalplus spec defines check() but carries no entry_point")
        if entry not in ns:
            raise NameError(f"solution does not define entry point {entry!r}")
        ns["check"](ns[entry])
    # else: the asserts already ran during the exec above.
    return 1, 1


def _run_functional(code: str, spec: dict) -> tuple[int, int]:
    """LiveCodeBench call-based: invoke a method on the generated class.

    LCB's call-based problems give a ``starter_code`` stub, conventionally a
    ``class Solution`` with one method. Prefer that exact shape; fall back to a
    bare module-level function of the same name, which models emit often enough
    that treating it as a failure would measure formatting rather than
    correctness.
    """
    fn_name = spec["fn_name"]
    inputs = spec["inputs"]
    outputs = spec["outputs"]

    ns: dict = {"__name__": "__solution__"}
    exec(compile(code, "<solution>", "exec"), ns)

    if "Solution" in ns and hasattr(ns["Solution"], fn_name):
        target = getattr(ns["Solution"](), fn_name)
    elif fn_name in ns and callable(ns[fn_name]):
        target = ns[fn_name]
    else:
        raise NameError(f"solution defines neither Solution.{fn_name} nor {fn_name}")

    n_pass = 0
    for args, want in zip(inputs, outputs):
        call_args = args if isinstance(args, list) else [args]
        if _values_match(target(*call_args), want):
            n_pass += 1
    return n_pass, len(outputs)


def _run_stdio(code: str, spec: dict) -> tuple[int, int]:
    """LiveCodeBench stdin/stdout: feed each input, compare each stdout.

    The solution is re-exec'd per test case. That is deliberate: these programs
    read stdin at import time and keep module-level state, so reusing the
    namespace makes case N's verdict depend on case N-1.
    """
    inputs = spec["inputs"]
    outputs = spec["outputs"]
    compiled = compile(code, "<solution>", "exec")

    real_stdin, real_stdout = sys.stdin, sys.stdout
    n_pass = 0
    try:
        for stdin_text, want in zip(inputs, outputs):
            buf = io.StringIO()
            sys.stdin = io.StringIO(stdin_text)
            sys.stdout = buf
            try:
                exec(compiled, {"__name__": "__main__"})
            except SystemExit:
                # A solution ending in sys.exit() has still produced its output.
                pass
            finally:
                sys.stdin, sys.stdout = real_stdin, real_stdout
            if _stdout_matches(buf.getvalue(), want):
                n_pass += 1
    finally:
        sys.stdin, sys.stdout = real_stdin, real_stdout
    return n_pass, len(outputs)


_KINDS = {
    "evalplus": _run_evalplus,
    "functional": _run_functional,
    "stdio": _run_stdio,
}


def _harden() -> None:
    """Remove the ways a generated program can damage the host or escape.

    Deliberately partial, and not a security boundary: the parent's timeout and
    the process boundary are what actually contain a hostile program. This only
    closes the accidents that are common in generated code and would otherwise
    corrupt the *run* -- deleting the dataset, forking until the node dies,
    or calling ``os._exit`` before we can report a verdict.
    """
    # NOTE: builtins.input is deliberately left alone. Competitive-programming
    # solutions -- the whole stdio half of LiveCodeBench -- read their input with
    # input(), and _run_stdio has already pointed sys.stdin at the test case, so
    # input() is the intended path rather than an escape.
    import os
    import shutil
    import subprocess

    try:
        import resource

        # 8 GiB of address space and 15s of CPU.
        #
        # These were 4 GiB / 8s and that was measurably too tight: at 8s the
        # CANONICAL solutions of HumanEval/139 (special_factorial) and MBPP+/599
        # were SIGKILLed by RLIMIT_CPU, i.e. the harness was scoring correct
        # reference code as wrong. EvalPlus's "plus" inputs are deliberate stress
        # cases and some genuinely need >8s of CPU. Verified by
        # debug_canonical_fails.py -- raise these only with that script re-run.
        #
        # Still well below what destabilizes a p4d node running 48 of these
        # concurrently alongside the trainer, and the parent's wall-clock kill
        # sits above the CPU limit so a busy-looping child is stopped by its own
        # rlimit (which still lets the parent read a verdict) rather than by the
        # outer kill (which does not).
        resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"] = "1"

    for mod, attr in (
        (os, "system"),
        (os, "fork"),
        (os, "forkpty"),
        (os, "kill"),
        (os, "remove"),
        (os, "unlink"),
        (os, "rmdir"),
        (os, "removedirs"),
        (os, "rename"),
        (os, "truncate"),
        (shutil, "rmtree"),
        (shutil, "move"),
        (subprocess, "Popen"),
        (subprocess, "run"),
        (subprocess, "call"),
        (subprocess, "check_output"),
    ):
        try:
            setattr(mod, attr, None)
        except Exception:
            pass


def main() -> int:
    payload = json.load(sys.stdin)
    code = payload["code"]
    spec = payload["spec"]
    kind = spec.get("type")

    verdict = {"passed": False, "n_pass": 0, "n_total": 0, "error": ""}
    runner = _KINDS.get(kind)
    if runner is None:
        verdict["error"] = f"unknown spec type {kind!r}"
    else:
        _harden()
        try:
            n_pass, n_total = runner(code, spec)
            verdict.update(
                passed=(n_total > 0 and n_pass == n_total),
                n_pass=n_pass,
                n_total=n_total,
            )
        except BaseException as exc:  # noqa: BLE001 - a crashing solution is a failing solution
            verdict["error"] = f"{type(exc).__name__}: {exc}"[:400]

    # Write to the real fd, not sys.stdout: a solution that reassigned or closed
    # sys.stdout must not be able to suppress or corrupt the verdict.
    import os

    os.write(1, (json.dumps(verdict) + "\n").encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
