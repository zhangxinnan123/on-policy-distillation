# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""CPU-only smoke test for verl.trainer.distillation.loop_metrics.

Runs the LoopMetricsPool inline (no Ray) using the deterministic 1-char-per-token
tokenizer from opd_inference.loop_analysis, so it has no HF / Ray dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make opd_inference importable when running pytest from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from opd_inference.loop_analysis import _CharTokenizer  # type: ignore  # noqa: E402

from verl.trainer.distillation.loop_metrics import (  # noqa: E402
    LoopMetricsPool,
    compute_loop_metrics,
)


def _pad_rows(rows: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length token-id rows into (ids, mask) tensors."""
    max_len = max(len(r) for r in rows)
    bsz = len(rows)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, : len(r)] = 1
    return ids, mask


def test_compute_loop_metrics_inline_pool_two_rows():
    tok = _CharTokenizer()

    # Row 0: clear chunk-level loop ("Let me verify." x4 separated by \n\n).
    looped = (
        "Step 1: Parse the problem.\n\n"
        "Step 2: Compute candidate answer 42.\n\n"
        "Let me verify.\n\n"
        "Let me verify.\n\n"
        "Let me verify.\n\n"
        "Let me verify."
    )
    # Row 1: a normal response with no repetition.
    clean = (
        "Step 1: Parse the problem.\n\n"
        "Step 2: Compute candidate answer 42.\n\n"
        "The final answer is 42."
    )

    rows = [tok.encode(looped), tok.encode(clean)]
    ids_2d, mask_2d = _pad_rows(rows)

    pool = LoopMetricsPool(
        tokenizer_name_or_path="<unused>",  # inline mode uses inline_tokenizer
        num_workers=0,
        inline_tokenizer=tok,
    )

    metrics = compute_loop_metrics(
        token_ids_2d=ids_2d,
        response_mask_2d=mask_2d,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
        sample_fraction=1.0,
        rng_seed=0,
    )

    # 1 of 2 rows is a loop.
    assert metrics["distillation/loop_rate"] == pytest.approx(0.5)
    assert metrics["distillation/loop_n_total"] == 2.0
    assert metrics["distillation/loop_n_hits"] == 1.0
    # Loop starts somewhere inside the response, not at position 0 nor at the end.
    sr = metrics["distillation/loop_start_relative_mean"]
    assert 0.0 < sr < 1.0
    # The loop is a chunk-level loop, so the kind split should reflect that.
    assert metrics["distillation/loop_kind_chunk_frac"] == pytest.approx(1.0)
    assert metrics["distillation/loop_kind_token_frac"] == pytest.approx(0.0)
    # Pattern length and total loop tokens must be positive.
    assert metrics["distillation/loop_pattern_token_length_mean"] > 0
    assert metrics["distillation/loop_total_loop_tokens_mean"] > 0


def test_compute_loop_metrics_empty_batch_returns_empty_dict():
    tok = _CharTokenizer()
    pool = LoopMetricsPool("<unused>", num_workers=0, inline_tokenizer=tok)

    ids = torch.zeros((0, 0), dtype=torch.long)
    mask = torch.zeros((0, 0), dtype=torch.long)
    metrics = compute_loop_metrics(
        token_ids_2d=ids,
        response_mask_2d=mask,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
    )
    assert metrics == {}


def test_compute_loop_metrics_no_loops_present():
    tok = _CharTokenizer()
    clean = "Step 1: Parse.\n\nStep 2: Compute 42.\n\nFinal: 42."
    rows = [tok.encode(clean), tok.encode(clean + " Done.")]
    ids_2d, mask_2d = _pad_rows(rows)

    pool = LoopMetricsPool("<unused>", num_workers=0, inline_tokenizer=tok)
    metrics = compute_loop_metrics(
        token_ids_2d=ids_2d,
        response_mask_2d=mask_2d,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
    )
    assert metrics["distillation/loop_rate"] == 0.0
    assert metrics["distillation/loop_n_total"] == 2.0
    assert metrics["distillation/loop_n_hits"] == 0.0
    # Loop-only stats must not appear when there are no hits.
    assert "distillation/loop_start_relative_mean" not in metrics
    assert "distillation/loop_pattern_token_length_mean" not in metrics


def test_suffix_mode_matches_full_on_end_anchored_loop():
    """A loop at the response end is found by both modes, with the same
    start_token_idx and pattern_token_length."""
    tok = _CharTokenizer()
    looped = (
        "Step 1: parse.\n\n"
        "Step 2: compute candidate 42.\n\n"
        "Let me verify.\n\n"
        "Let me verify.\n\n"
        "Let me verify.\n\n"
        "Let me verify."
    )
    rows = [tok.encode(looped)]
    ids_2d, mask_2d = _pad_rows(rows)

    pool = LoopMetricsPool("<unused>", num_workers=0, inline_tokenizer=tok)

    m_full = compute_loop_metrics(
        token_ids_2d=ids_2d,
        response_mask_2d=mask_2d,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
        mode="full",
    )
    m_suffix = compute_loop_metrics(
        token_ids_2d=ids_2d,
        response_mask_2d=mask_2d,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
        mode="suffix",
    )

    assert m_full["distillation/loop_rate"] == 1.0
    assert m_suffix["distillation/loop_rate"] == 1.0
    assert m_full["distillation/loop_pattern_token_length_mean"] == pytest.approx(
        m_suffix["distillation/loop_pattern_token_length_mean"]
    )
    assert m_full["distillation/loop_total_loop_tokens_mean"] == pytest.approx(
        m_suffix["distillation/loop_total_loop_tokens_mean"]
    )


def test_suffix_mode_misses_mid_only_loop_that_full_catches():
    """A loop in the middle of the response that doesn't extend to the end
    must be caught by full mode but not by suffix mode."""
    tok = _CharTokenizer()
    # Mid-response loop ("Mid block.\n\n" x3), then a clean tail.
    mid_loop = (
        "Intro paragraph.\n\n"
        "Mid block A B C.\n\n"
        "Mid block A B C.\n\n"
        "Mid block A B C.\n\n"
        "Then we move on.\n\n"
        "And conclude with the final answer 42."
    )
    rows = [tok.encode(mid_loop)]
    ids_2d, mask_2d = _pad_rows(rows)

    pool = LoopMetricsPool("<unused>", num_workers=0, inline_tokenizer=tok)

    m_full = compute_loop_metrics(
        token_ids_2d=ids_2d,
        response_mask_2d=mask_2d,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
        mode="full",
    )
    m_suffix = compute_loop_metrics(
        token_ids_2d=ids_2d,
        response_mask_2d=mask_2d,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
        mode="suffix",
    )
    assert m_full["distillation/loop_rate"] == 1.0, "full mode should catch the mid-response loop"
    assert m_suffix["distillation/loop_rate"] == 0.0, (
        "suffix mode should not flag a loop that doesn't extend to the response end"
    )


def test_sub_sampling_respects_fraction_and_is_deterministic():
    tok = _CharTokenizer()
    rows = [tok.encode(f"Step {i}: x.\n\nFinal: 42.") for i in range(8)]
    ids_2d, mask_2d = _pad_rows(rows)

    pool = LoopMetricsPool("<unused>", num_workers=0, inline_tokenizer=tok)
    m1 = compute_loop_metrics(
        token_ids_2d=ids_2d,
        response_mask_2d=mask_2d,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
        sample_fraction=0.5,
        rng_seed=42,
    )
    m2 = compute_loop_metrics(
        token_ids_2d=ids_2d,
        response_mask_2d=mask_2d,
        pool=pool,
        min_repeats=2,
        min_pattern_tokens=5,
        sample_fraction=0.5,
        rng_seed=42,
    )
    assert m1["distillation/loop_n_total"] == 4.0  # 0.5 * 8
    assert m1 == m2
