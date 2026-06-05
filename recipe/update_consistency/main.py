# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hydra entry for the update-consistency experiment.

Runs standard OPD training with `UpdateConsistencyCallback` attached. Every
`_update_actor` is bracketed by before/after `compute_log_prob` calls on the
current training batch.

Example:
  python -m recipe.update_consistency.main \\
      --config-name=measure_during_opd \\
      update_consistency.output_dir=data/measurements \\
      trainer.total_training_steps=10
"""

from __future__ import annotations

import os
import socket

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner, run_ppo


class ConsistencyTaskRunner(TaskRunner):
    """TaskRunner that swaps RayPPOTrainer → ConsistencyRayPPOTrainer and
    attaches the measurement callback before calling trainer.fit().

    Mirrors `TaskRunner.run` (verl/trainer/main_ppo.py:300) faithfully — only
    the trainer construction site is overridden.
    """

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.trainer.main_ppo import (
            create_rl_dataset,
            create_rl_sampler,
            need_critic,
            need_reference_policy,
            validate_config,
        )
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        from .callback import UpdateConsistencyCallback
        from .trainer import ConsistencyRayPPOTrainer

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            is_train=True, max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False, max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = ConsistencyRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()

        uc_cfg = config.update_consistency
        # Pull lr from the actor's optimizer config so the callback can write
        # predicted_absolute_magnitude = lr · |A| · shape into the parquet.
        try:
            lr_for_callback = float(config.actor_rollout_ref.actor.optim.lr)
        except Exception:
            lr_for_callback = None
        cb = UpdateConsistencyCallback(
            output_dir=uc_cfg.output_dir,
            measure_every=int(uc_cfg.measure_every),
            single_token_mode=bool(uc_cfg.get("single_token_mode", False)),
            seed=int(uc_cfg.get("seed", 0)),
            noise_check=bool(uc_cfg.get("noise_check", False)),
            lr=lr_for_callback,
        )
        trainer.set_update_consistency_callback(cb)
        trainer.fit()


@hydra.main(config_path="config", config_name="measure_during_opd", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ConsistencyTaskRunner))


if __name__ == "__main__":
    main()
