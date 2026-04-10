# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from skyrl-gym/envs/math_boxed/utils.py
# Robust \\boxed{} answer extraction with multiple fallback strategies.

import re
from typing import Any, Dict, Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the \\boxed{} wrapper and return the inner content."""
    if "\\boxed " in s:
        return s[len("\\boxed "):]
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a math answer string for comparison."""
    if not final_answer:
        return ""
    final_answer = str(final_answer).strip()
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"\\dfrac", r"\\frac", final_answer)
    final_answer = re.sub(r"\^([a-zA-Z0-9])\b", r"^{\1}", final_answer)
    final_answer = re.sub(r"\\mathrm\{[^}]*\}", "", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    final_answer = re.sub(r"\\textbf\{([^}]*)\}", r"\1", final_answer)
    final_answer = re.sub(r"^\(([A-Za-z])\)$", r"\1", final_answer)
    final_answer = re.sub(r"\\left\(", "(", final_answer)
    final_answer = re.sub(r"\\right\)", ")", final_answer)
    final_answer = re.sub(r"\\left\[", "[", final_answer)
    final_answer = re.sub(r"\\right\]", "]", final_answer)
    final_answer = re.sub(r";", ",", final_answer)
    final_answer = re.sub(r",\s*", ",", final_answer)
    final_answer = re.sub(r"\s*([+\-*/=])\s*", r"\1", final_answer)
    final_answer = re.sub(r"\s+", "", final_answer)

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    if final_answer.replace(",", "").replace(".", "").isdigit():
        try:
            float_val = float(final_answer.replace(",", ""))
            final_answer = str(int(float_val)) if float_val.is_integer() else final_answer.replace(",", "")
        except ValueError:
            pass

    return final_answer.strip()


def is_equiv(str1: str, str2: str) -> bool:
    """Check if two math expressions are equivalent after normalization."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return normalize_final_answer(str1) == normalize_final_answer(str2)
    except Exception:
        return str1 == str2


def _extract_boxed(text: str, search_window: Optional[int] = None) -> Optional[str]:
    """Extract last \\boxed{} from text, optionally limiting to a tail window."""
    src = text[-search_window:] if search_window and len(text) > search_window else text
    boxed = last_boxed_only_string(src)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except Exception:
        return None


def compute_score(
    solution_str: str,
    ground_truth: str,
    search_window: int = 500,
    use_math_verify: bool = True,
) -> Dict[str, Any]:
    """Compute reward score by extracting \\boxed{} answer with multiple fallback strategies.

    Strategies tried in order:
    1. Full-text \\boxed{} extraction + string normalization
    2. Progressively larger tail windows (1000, 1500, full)
    3. Offsets from the end (skip last 100/200/300 chars)
    4. math_verify symbolic check (if use_math_verify=True)

    Returns dict with keys: score (1.0/0.0), acc (bool), pred (str), method (str).
    """
    # Strategy 1: standard window
    for window in [search_window, 1000, 1500, None]:
        pred = _extract_boxed(solution_str, window)
        if pred is not None and is_equiv(pred, ground_truth):
            return {"score": 1.0, "acc": True, "pred": pred, "method": f"boxed_window_{window}"}

    # Strategy 2: skip trailing chars (handles models that append extra text after \\boxed{})
    for offset in [100, 200, 300]:
        if len(solution_str) > offset:
            pred = _extract_boxed(solution_str[:-offset])
            if pred is not None and is_equiv(pred, ground_truth):
                return {"score": 1.0, "acc": True, "pred": pred, "method": f"boxed_offset_{offset}"}

    # Strategy 3: math_verify symbolic equivalence check
    if use_math_verify:
        try:
            from verl.utils.reward_score import math_verify as _math_verify
            score = _math_verify.compute_score(solution_str, ground_truth)
            if score > 0:
                pred = _extract_boxed(solution_str) or ""
                return {"score": 1.0, "acc": True, "pred": pred, "method": "math_verify"}
        except Exception:
            pass

    # All strategies failed — return best available pred for debugging
    pred = _extract_boxed(solution_str) or ""
    return {"score": 0.0, "acc": False, "pred": pred, "method": "none"}
