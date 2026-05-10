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

"""Driver-side loop-detection metrics on student rollouts.

Wraps `opd_inference.loop_analysis.detect_loop` in a small Ray actor pool so
the per-rollout `tokenizer.decode` cost can be amortized across cores while
the trainer waits on reward / log-prob compute. Diagnostic only — produces
scalar metrics under the `distillation/loop_*` namespace, no effect on loss
or reward.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

__all__ = ["LoopMetricsPool", "compute_loop_metrics"]


def _import_detect_loop():
    """Import detect_loop from `opd_inference/`, which is a sibling of the verl
    package (not on the default sys.path of an installed verl). Adds the repo
    root to sys.path on first call and retries.
    """
    try:
        from opd_inference.loop_analysis import detect_loop  # type: ignore

        return detect_loop
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        if (repo_root / "opd_inference" / "loop_analysis.py").is_file():
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from opd_inference.loop_analysis import detect_loop  # type: ignore

            return detect_loop
        raise ImportError(
            "verl.trainer.distillation.loop_metrics requires "
            "`opd_inference/loop_analysis.py` at the repo root. Either disable "
            "distillation.loop_metrics, or add opd_inference/ to sys.path."
        )


def _detect_rows_inline(
    token_ids_per_row: Sequence[Sequence[int]],
    tokenizer: Any,
    *,
    min_repeats: int,
    min_pattern_tokens: int,
    mode: str,
    max_period_chunks: int,
) -> List[Dict[str, Any]]:
    detect_loop = _import_detect_loop()
    out: List[Dict[str, Any]] = []
    for ids in token_ids_per_row:
        ids_list = list(ids)
        text = tokenizer.decode(ids_list, skip_special_tokens=True)
        out.append(
            detect_loop(
                text,
                ids_list,
                tokenizer,
                min_repeats=min_repeats,
                min_pattern_tokens=min_pattern_tokens,
                mode=mode,
                max_period_chunks=max_period_chunks,
            )
        )
    return out


def _make_actor_class():
    """Defer the @ray.remote decoration until pool construction so that simply
    importing this module does not require a running Ray cluster.
    """
    import ray  # type: ignore

    @ray.remote
    class _LoopDetectActor:
        def __init__(self, tokenizer_name_or_path: str):
            from transformers import AutoTokenizer  # type: ignore

            self.tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        def detect(
            self,
            token_ids_per_row: List[List[int]],
            min_repeats: int,
            min_pattern_tokens: int,
            mode: str,
            max_period_chunks: int,
        ) -> List[Dict[str, Any]]:
            return _detect_rows_inline(
                token_ids_per_row,
                self.tok,
                min_repeats=min_repeats,
                min_pattern_tokens=min_pattern_tokens,
                mode=mode,
                max_period_chunks=max_period_chunks,
            )

    return _LoopDetectActor


class LoopMetricsPool:
    """Driver-side pool for running `detect_loop` in parallel.

    Args:
        tokenizer_name_or_path: passed to AutoTokenizer.from_pretrained inside
            each actor. For inline mode (num_workers <= 0) you can pass an
            already-loaded tokenizer via `inline_tokenizer` to avoid a second
            copy on the driver.
        num_workers: number of Ray actors to spin up. 0 runs inline on the
            calling thread (handy for tests, or when batches are small and the
            actor RPC overhead would dominate).
        inline_tokenizer: optional pre-loaded tokenizer used only when
            num_workers <= 0.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        num_workers: int,
        *,
        inline_tokenizer: Any = None,
    ):
        self.num_workers = max(0, int(num_workers))
        self._actors: List[Any] = []
        self._inline_tok: Any = None

        if self.num_workers <= 0:
            if inline_tokenizer is not None:
                self._inline_tok = inline_tokenizer
            else:
                from transformers import AutoTokenizer  # type: ignore

                self._inline_tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            return

        actor_cls = _make_actor_class()
        self._actors = [actor_cls.remote(tokenizer_name_or_path) for _ in range(self.num_workers)]

    def detect_batch(
        self,
        token_ids_per_row: Sequence[Sequence[int]],
        *,
        min_repeats: int,
        min_pattern_tokens: int,
        mode: str = "suffix",
        max_period_chunks: int = 64,
    ) -> List[Dict[str, Any]]:
        if not token_ids_per_row:
            return []

        rows: List[List[int]] = [list(t) for t in token_ids_per_row]

        if self.num_workers <= 0:
            return _detect_rows_inline(
                rows,
                self._inline_tok,
                min_repeats=min_repeats,
                min_pattern_tokens=min_pattern_tokens,
                mode=mode,
                max_period_chunks=max_period_chunks,
            )

        import ray  # type: ignore

        n = len(rows)
        # Round-robin shard so per-actor decode costs stay balanced even when
        # response lengths differ.
        shards: List[List[List[int]]] = [[] for _ in range(self.num_workers)]
        for i, r in enumerate(rows):
            shards[i % self.num_workers].append(r)

        futures = [
            self._actors[w].detect.remote(
                shards[w], min_repeats, min_pattern_tokens, mode, max_period_chunks
            )
            for w in range(self.num_workers)
            if shards[w]
        ]
        gathered = ray.get(futures)

        # Un-roll round-robin into input order. Each non-empty shard's results
        # appear in `gathered` in the same order it was submitted, so we can
        # walk per-shard cursors.
        active_shards = [w for w in range(self.num_workers) if shards[w]]
        shard_to_results: Dict[int, List[Dict[str, Any]]] = {
            w: gathered[idx] for idx, w in enumerate(active_shards)
        }
        cursors: Dict[int, int] = {w: 0 for w in active_shards}
        out: List[Dict[str, Any]] = []
        for i in range(n):
            sid = i % self.num_workers
            res = shard_to_results[sid][cursors[sid]]
            cursors[sid] += 1
            out.append(res)
        return out

    def shutdown(self) -> None:
        if not self._actors:
            return
        try:
            import ray  # type: ignore

            for a in self._actors:
                try:
                    ray.kill(a)
                except Exception:
                    pass
        finally:
            self._actors = []


def _strip_padding(
    token_ids_2d: torch.Tensor, response_mask_2d: torch.Tensor
) -> List[List[int]]:
    """Slice each row of `token_ids_2d` down to its valid response length using
    `response_mask_2d`. Both tensors must have shape (bsz, seqlen) and align
    column-wise (i.e. response_mask covers the response slice of token_ids).
    """
    if token_ids_2d.shape != response_mask_2d.shape:
        raise ValueError(
            f"shape mismatch: token_ids {tuple(token_ids_2d.shape)} vs "
            f"response_mask {tuple(response_mask_2d.shape)}"
        )
    lengths = response_mask_2d.sum(dim=-1).to(torch.long).tolist()
    rows: List[List[int]] = []
    ids_cpu = token_ids_2d.detach().to("cpu", dtype=torch.long)
    for i, L in enumerate(lengths):
        rows.append(ids_cpu[i, : int(L)].tolist())
    return rows


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Reduce per-row loop records into scalar metrics under the
    'distillation/loop_*' namespace.
    """
    if not records:
        return {}
    n = len(records)
    is_loop = np.fromiter((bool(r.get("is_loop")) for r in records), dtype=bool, count=n)
    loop_recs = [r for r in records if r.get("is_loop")]

    out: Dict[str, float] = {
        "distillation/loop_rate": float(is_loop.mean()),
        "distillation/loop_n_total": float(n),
        "distillation/loop_n_hits": float(is_loop.sum()),
    }
    if not loop_recs:
        return out

    def _arr(key: str) -> np.ndarray:
        vals = [r[key] for r in loop_recs if r.get(key) is not None]
        return np.asarray(vals, dtype=np.float64)

    sr = _arr("start_relative")
    if sr.size:
        out["distillation/loop_start_relative_mean"] = float(sr.mean())

    plen = _arr("pattern_token_length")
    if plen.size:
        out["distillation/loop_pattern_token_length_mean"] = float(plen.mean())

    tlt = _arr("total_loop_tokens")
    if tlt.size:
        out["distillation/loop_total_loop_tokens_mean"] = float(tlt.mean())

    reps = _arr("repeats")
    if reps.size:
        out["distillation/loop_repeats_mean"] = float(reps.mean())

    # Loop-kind split (chunk vs token suffix). Useful to know whether degenerate
    # token-level loops dominate or whether they're paragraph-level.
    n_chunk = sum(1 for r in loop_recs if r.get("loop_kind") == "chunk")
    n_token = sum(1 for r in loop_recs if r.get("loop_kind") == "token")
    out["distillation/loop_kind_chunk_frac"] = float(n_chunk / max(1, len(loop_recs)))
    out["distillation/loop_kind_token_frac"] = float(n_token / max(1, len(loop_recs)))

    return out


def compute_loop_metrics(
    token_ids_2d: torch.Tensor,
    response_mask_2d: torch.Tensor,
    pool: LoopMetricsPool,
    *,
    min_repeats: int,
    min_pattern_tokens: int,
    sample_fraction: float = 1.0,
    rng_seed: Optional[int] = 0,
    mode: str = "suffix",
    max_period_chunks: int = 64,
) -> Dict[str, float]:
    """Run loop detection over a rollout batch and return scalar metrics.

    Args:
        token_ids_2d: (bsz, seqlen) padded tensor of student response token IDs.
        response_mask_2d: (bsz, seqlen) tensor of valid-token indicators (1 for
            real response tokens, 0 for padding). Must align with `token_ids_2d`.
        pool: a LoopMetricsPool initialised with the matching tokenizer.
        min_repeats / min_pattern_tokens: detector thresholds.
        sample_fraction: fraction of rows to detect on, in (0, 1]. Sub-sampling
            lets you amortize cost when batches are large; the per-step
            estimate's variance grows accordingly.
        rng_seed: deterministic seed for sub-sampling. None for non-deterministic.

    Returns:
        Dict of scalar metrics. Empty dict if the input batch is empty.
    """
    rows = _strip_padding(token_ids_2d, response_mask_2d)
    n = len(rows)
    if n == 0:
        return {}

    if 0.0 < sample_fraction < 1.0:
        rng = np.random.default_rng(rng_seed)
        k = max(1, int(round(n * sample_fraction)))
        idx = rng.choice(n, size=k, replace=False)
        rows = [rows[int(i)] for i in idx]

    records = pool.detect_batch(
        rows,
        min_repeats=min_repeats,
        min_pattern_tokens=min_pattern_tokens,
        mode=mode,
        max_period_chunks=max_period_chunks,
    )
    return _aggregate(records)
