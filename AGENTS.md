# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is verl

verl (Volcano Engine Reinforcement Learning) is a flexible, distributed RL training library for LLMs, built on Ray for single-controller orchestration. It is the open-source implementation of the HybridFlow paper. The library supports FSDP/Megatron-LM training backends with vLLM/SGLang/HF rollout backends, and covers PPO, GRPO, and many other RL algorithms.

---

## Agent Instructions (Mandatory)

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

### Contribution Policy

**Duplicate-work checks** — Before proposing a PR, run:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- Do not open one-off PRs for tiny edits (single typo, isolated style change). Cleanups only when bundled with substantive work.
- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- PR descriptions for AI-assisted work **must** include why it's not a duplicate, test commands run and results, and a clear statement that AI assistance was used.

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

### Commit messages

```text
Your commit message here

Co-authored-by: Claude
Signed-off-by: Your Name <your.email@example.com>
```

### Domain-specific guides

Do not modify code in these areas without first reading the linked guide:

- **Editing these instructions**: [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)

---

## Environment Setup

```bash
# Use uv for environment management
uv venv --python 3.12
source .venv/bin/activate

# Python-only development (no GPU deps)
pip install -e .[test,vllm]
# or
pip install -e .[test,sglang]

# Pre-commit setup
uv pip install pre-commit hydra-core
pre-commit install
```

## Commands

### Linting

```bash
# Run on staged changes
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

The `autogen-trainer-cfg` hook regenerates `verl/trainer/config/_generated_*.yaml` from config dataclasses (via `scripts/generate_trainer_config.sh`). Run it manually after adding or renaming fields in any config dataclass under `verl/workers/config/` or `verl/trainer/config/`.

The `check-docs-time-info` hook enforces that any modified `.md` file under `docs/` contains a `Last updated:` line. Update that field whenever you edit documentation.

### Tests

```bash
# Run ALL CPU tests (no GPU required) — files ending in *_on_cpu.py
pytest tests/ -k "_on_cpu"

# Run a single CPU test file
pytest tests/test_protocol_on_cpu.py

# Run a specific GPU test
pytest tests/workers/test_some_worker.py

# Tests are organized under tests/ mirroring verl/ namespaces:
#   tests/trainer/   → verl/trainer/
#   tests/workers/   → verl/workers/
#   tests/models/    → verl/models/
# Special subdirs:
#   tests/special_distributed/  — multi-GPU unit tests
#   tests/special_e2e/          — end-to-end training/generation tests
#   tests/special_sanity/       — quick sanity checks
#   tests/special_standalone/   — standalone environment tests
```

### Running a training job

The main entrypoint is `verl/trainer/main_ppo.py`, launched via `python -m`:

```bash
python -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    [hydra overrides...]
```

Example: on-policy distillation on GSM8K:

```bash
# Set DATA_PATH before running
bash examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh
```

### Building docs

```bash
pip install -e .[test]
cd docs
pip install -r requirements-docs.txt
make clean && make html
python -m http.server -d _build/html/
```

---

## Architecture Overview

### Programming model

verl uses a **single-controller** design: a Ray driver process (`RayPPOTrainer` in `verl/trainer/ppo/ray_trainer.py`) orchestrates all computation. Workers are Ray actors grouped into `RayWorkerGroup`s; the driver dispatches batched calls to them and handles the RL loop logic.

### Key abstractions

| Component | Location | Description |
|---|---|---|
| `DataProto` / `TensorDict` | `verl/protocol.py` | Core data container passed between all components |
| `RayWorkerGroup` | `verl/single_controller/ray/` | Wraps Ray actors; supports `dispatch` to call methods across all workers |
| `ResourcePoolManager` | `verl/single_controller/ray/` | Maps roles to GPU resource pools; enables colocated or split placement |
| `Worker` (base) | `verl/single_controller/base/` | Base class for all distributed workers |

### Workers

Workers live in `verl/workers/` and are instantiated as Ray actors:

- **`fsdp_workers.py`** / **`megatron_workers.py`**: Implement actor, critic, reference, and rollout methods for FSDP and Megatron backends respectively.
- **`engine_workers.py`**: Engine-based worker combining actor+rollout in one hybrid engine.
- **`workers/actor/`**, **`workers/critic/`**, **`workers/rollout/`**: Role-specific logic.
- **`workers/rollout/`**: Supports `vllm_rollout/`, `sglang_rollout/`, `hf_rollout.py`, `trtllm_rollout/`.
- **`workers/reward_manager/`**: Reward computation (model-based and function-based).
- **`workers/config/`**: Dataclass configs for each worker role (`ActorConfig`, `RolloutConfig`, `DistillationConfig`, etc.).

### Trainer loop

`verl/trainer/ppo/ray_trainer.py` (`RayPPOTrainer`) drives the synchronous RL loop:

1. Rollout: actor generates sequences via vLLM/SGLang.
2. Teacher (if distillation): `TeacherModelManager` (`verl/experimental/teacher_loop/`) queries top-k log-probs from a separate teacher inference server.
3. Reward: compute task rewards and/or reference KL penalty.
4. Advantage estimation: GAE, GRPO, RLOO, etc. (configured via `algorithm.adv_estimator`).
5. Actor/critic update: PPO/GRPO gradient updates.

Core algorithm implementations: `verl/trainer/ppo/core_algos.py`.

### Configuration system

All configs use **Hydra + OmegaConf**. Root config: `verl/trainer/config/ppo_trainer.yaml`. It uses `defaults:` lists to compose sub-configs from `verl/trainer/config/` subdirectories (`actor/`, `rollout/`, `critic/`, `distillation/`, etc.). Configs are converted to dataclasses via `omega_conf_to_dataclass`.

Key config groups:
- `actor_rollout_ref.*` — actor, rollout, reference model settings
- `critic.*` — critic model settings
- `distillation.*` — on-policy distillation settings (see below)
- `algorithm.*` — RL algorithm hyperparameters
- `trainer.*` — training loop settings (epochs, save/test freq, logging)
- `data.*` — dataset paths and batching

### On-policy distillation

This fork adds on-policy knowledge distillation. There are two separate distillation paths:

**Path 1 — FSDP/Megatron general path** (standard, use this for most cases):
Enabled via `distillation.enabled=True` in `main_ppo.py`. Key components:

- **`verl/trainer/distillation/losses.py`**: Loss registry. New loss modes are added via `@register_distillation_loss(DistillationLossSettings(...))`.
- **`verl/trainer/distillation/fsdp/`** and **`verl/trainer/distillation/megatron/`**: Backend-specific forward KL top-k kernel implementations.
- **`verl/experimental/teacher_loop/teacher_manager.py`** (`TeacherModelManager`): Manages a pool of async vLLM/SGLang teacher inference servers. Returns top-k log-probs + token indices per sequence.
- **`verl/workers/config/distillation.py`** (`DistillationConfig`, `DistillationLossConfig`, `DistillationTeacherModelConfig`): All distillation configuration dataclasses.

**Path 2 — Megatron async recipe** (higher throughput, overlaps rollout/teacher/update phases):
Uses `recipe/gkd/` (submodule; init with `git submodule update --init --recursive recipe`) and `python3 -m recipe.gkd.main_gkd`. See `docs/advance/async-on-policy-distill.md` for the scheduler design (`one_step_off` / `two_step_off`). Requires a running teacher ZeroMQ server (`recipe/gkd/teacher/start_server.sh`).

The distillation config (`verl/trainer/config/distillation/distillation.yaml`) controls:
- `loss_mode`: choose between two families:
  - `forward_kl_topk` — forward KL using the full top-k teacher distribution; requires `topk > 1` teacher logprobs per token
  - `k1`, `k3`, `kl`, `abs`, `mse`, `k2`, `low_var_kl` — single-sample KL estimators; only need 1 teacher logprob (set `topk: 1`)
- `topk`: number of top-k teacher logits (only meaningful for `forward_kl_topk`)
- `use_task_rewards`: combine distillation loss with RL reward
- `use_policy_gradient`: treat negative distillation loss as advantage signal instead of direct supervised loss
- `teacher_model.*`: teacher inference server configuration
  - `reapply_chat_template` (default `False`): re-apply the teacher's chat template to the raw prompt before concatenating the student's response ids. Use when student and teacher have different prompt formats (e.g. student is a base model, teacher is an instruct model). Requires `teacher_model.model_path` to be set. Student and teacher must share the same tokenizer vocabulary.
  - `enable_thinking` (default `False`): passed to `apply_chat_template(enable_thinking=...)` when `reapply_chat_template=True`.
  - `substitute_eos_token` (default `False`): replace the student's EOS token with the teacher's `<|im_end|>` token at the end of each response before querying the teacher. Use when the student (e.g. a base model using `<|endoftext|>`) and teacher (e.g. an instruct model using `<|im_end|>`) share the same tokenizer vocabulary but have different EOS tokens. Enable in bash script with `distillation.teacher_model.substitute_eos_token=True`.

Example script: `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`

### Experimental modules

`verl/experimental/` contains features under active development:
- `teacher_loop/`: Teacher model management for distillation
- `one_step_off_policy/`: Async one-step-off-policy scheduler
- `fully_async_policy/`: Fully async decoupled trainer/rollout
- `transfer_queue/`: Async data transfer between workers
- `agent_loop/`: Agentic multi-turn loop infrastructure
- `reward_loop/`: Modular reward function pipeline
- `dataset/`: Custom dataset samplers and curriculum learning

### `recipe/` submodule

Advanced algorithm recipes (DAPO, GSPO, PRIME, SPPO, etc.) live in a separate `verl-recipe` repo, linked as a git submodule at `recipe/`. Initialize with:

```bash
git submodule update --init --recursive recipe
```
