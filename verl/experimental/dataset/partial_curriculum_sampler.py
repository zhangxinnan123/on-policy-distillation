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
"""Static stage-based curriculum sampler for partial-continuation datasets.

The training parquet carries a ``partial_idx`` column where higher values mean
longer teacher prefixes (more constrained generation). This sampler emits rows
bucket-by-bucket in a configurable stage order, optionally capping each stage
to a fixed number of training steps.
"""
import random
from collections.abc import Sized

from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


class PartialCurriculumSampler(AbstractSampler):
    """Static curriculum over ``partial_idx`` buckets.

    Config (under ``data.sampler``):
        stages:           list[int] of ``partial_idx`` values in visit order.
                          Default ``[3, 2, 1, 0]`` (longest partial → vanilla).
        within_stage:     ``"preserve"`` (parquet order) or ``"shuffle"``.
        seed:             int, used when ``within_stage="shuffle"``.
        steps_per_stage:  optional int. If set, truncate each stage to
                          ``steps_per_stage * data.train_batch_size`` rows.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        df = data_source.dataframe  # datasets.Dataset (post filter)
        if "partial_idx" not in df.column_names:
            raise ValueError(
                "PartialCurriculumSampler requires a 'partial_idx' column in the dataset; "
                f"got columns={df.column_names}"
            )
        partial_idx = list(df["partial_idx"])

        sc = data_config.get("sampler", {}) or {}
        stages = list(sc.get("stages", [3, 2, 1, 0]))
        within = sc.get("within_stage", "preserve")
        seed = int(sc.get("seed", 0))
        cap = sc.get("steps_per_stage", None)
        bsz = int(data_config.train_batch_size)

        buckets: dict[int, list[int]] = {s: [] for s in stages}
        for i, p in enumerate(partial_idx):
            if p in buckets:
                buckets[p].append(i)

        if within == "shuffle":
            rng = random.Random(seed)
            for s in stages:
                rng.shuffle(buckets[s])
        elif within != "preserve":
            raise ValueError(f"within_stage must be 'preserve' or 'shuffle', got {within!r}")

        order: list[int] = []
        for s in stages:
            rows = buckets[s]
            if cap is not None:
                rows = rows[: int(cap) * bsz]
            order.extend(rows)

        if not order:
            raise ValueError(
                f"PartialCurriculumSampler produced an empty order. stages={stages}, "
                f"bucket sizes={{s: len(buckets[s]) for s in stages}}"
            )

        self._order = order
        self._stage_summary = {s: len(buckets[s]) for s in stages}

        print(
            f"[PartialCurriculumSampler] stages={stages} within_stage={within} "
            f"steps_per_stage={cap} bucket_sizes={self._stage_summary} total={len(order)}"
        )

    def __iter__(self):
        return iter(self._order)

    def __len__(self):
        return len(self._order)
