# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""ConsistencyRayPPOTrainer — RayPPOTrainer with before/after hooks around
`_update_actor` for the in-flight update-consistency measurement.

User must set actor_rollout_ref.actor.ppo_epochs=1 and ppo_mini_batch_size =
train_batch_size so each `_update_actor` dispatch is exactly one optimizer
step (the theory predicts per-optimizer-step Δπ).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

if TYPE_CHECKING:
    from .callback import UpdateConsistencyCallback

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class ConsistencyRayPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_consistency_callback: Optional["UpdateConsistencyCallback"] = None

    def set_update_consistency_callback(self, cb: "UpdateConsistencyCallback") -> None:
        self._update_consistency_callback = cb

    def _update_actor(self, batch: DataProto) -> DataProto:
        cb = self._update_consistency_callback
        if cb is not None:
            try:
                cb.before_update(self, batch)
            except Exception as e:
                logger.warning("[update_consistency] before_update raised: %r", e)

        actor_output = super()._update_actor(batch)

        if cb is not None:
            try:
                cb.after_update(self, batch)
            except Exception as e:
                logger.warning("[update_consistency] after_update raised: %r", e)

        return actor_output
