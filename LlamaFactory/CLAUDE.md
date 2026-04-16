# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is LLaMA Factory

LLaMA Factory is a unified fine-tuning framework for 100+ LLMs. It supports pre-training, SFT, reward modeling, PPO, DPO, KTO, ORPO, and other RL-based methods. It is installed as the `llamafactory` package under `src/llamafactory/`, exposed via two CLI aliases: `llamafactory-cli` and `lmf`.

---

## Commands

### Installation

```bash
pip install -e ".[dev]"
```

### Linting and formatting

```bash
make style        # auto-fix: ruff check --fix + ruff format
make quality      # check only (no fix)
make commit       # pre-commit install + run --all-files
```

### Tests

```bash
# Run all tests (CPU + GPU)
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ tests_v1/

# Run a single test file
WANDB_DISABLED=true pytest -vv tests/data/test_template.py

# Skip slow tests (default behavior)
pytest tests/           # slow tests are skipped unless RUN_SLOW=1
RUN_SLOW=1 pytest tests/  # include slow tests

# Multi-GPU tests require the require_distributed marker and the right number of visible devices
```

Custom markers defined in `tests/conftest.py`:
- `@pytest.mark.slow` — skipped unless `RUN_SLOW=1`
- `@pytest.mark.runs_on(["cuda"])` — skipped if device type doesn't match
- `@pytest.mark.require_distributed(N)` — skipped if fewer than N GPUs visible

### Training (CLI)

```bash
# Single-GPU or CPU
llamafactory-cli train examples/train_full/qwen3_full_sft.yaml

# Multi-GPU (auto-detected, uses torchrun internally)
llamafactory-cli train examples/train_full/qwen3_full_sft.yaml

# Other subcommands
llamafactory-cli api      # OpenAI-compatible API server
llamafactory-cli chat     # CLI chat interface
llamafactory-cli webchat  # Gradio chat demo
llamafactory-cli webui    # LlamaBoard full UI
llamafactory-cli export   # merge LoRA adapters and export
llamafactory-cli env      # print environment info
```

Arguments can be passed as YAML files or inline overrides. Distributed env vars: `NNODES`, `NODE_RANK`, `NPROC_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`. Set `FORCE_TORCHRUN=1` to always use torchrun. Set `USE_V1=1` to use the experimental v1 trainer stack.

### Build

```bash
make build   # builds a wheel via uv build / python -m build
```

---

## Architecture

### Package layout (`src/llamafactory/`)

| Module | Purpose |
|---|---|
| `cli.py` + `launcher.py` | Entrypoint; dispatches subcommands; auto-launches torchrun for multi-GPU |
| `hparams/` | All argument dataclasses (`ModelArguments`, `DataArguments`, `TrainingArguments`, `FinetuningArguments`, `GeneratingArguments`) and `get_train_args` / `get_infer_args` parsers |
| `model/` | Model loading (`loader.py`), LoRA adapter attachment (`adapter.py`), model patching (`patcher.py`), and per-feature utilities in `model_utils/` (attention, quantization, RoPE, etc.) |
| `data/` | Dataset loading (`loader.py`), chat-template application (`template.py`), tokenization/formatting (`processor/`), data collation (`collator.py`), and multi-modal input handling (`mm_plugin.py`) |
| `train/` | One subpackage per stage: `pt/`, `sft/`, `dpo/`, `kto/`, `ppo/`, `rm/`; shared utilities in `trainer_utils.py` and `callbacks.py`; top-level `tuner.py` selects the right stage |
| `api/` | FastAPI OpenAI-compatible server |
| `chat/` | CLI and streaming chat model |
| `webui/` | Gradio-based LlamaBoard UI |
| `extras/` | Logging, constants, misc helpers, package detection (`packages.py`) |
| `v1/` | Experimental next-generation trainer stack (activated with `USE_V1=1`) |

### Training flow

1. `launcher.py` parses `sys.argv`; if multi-GPU, re-launches via `torchrun`.
2. `train/tuner.py:run_exp` → `get_train_args` parses all args → loads tokenizer (`model/loader.py`) → loads dataset (`data/loader.py`) → dispatches to stage-specific `run_*` function (e.g., `train/sft/workflow.py:run_sft`).
3. Each stage builds a HuggingFace `Trainer` subclass, adds callbacks, and calls `.train()`.

### Configuration

Training configs are YAML files passed as positional arguments. All top-level keys map directly to argument dataclass fields (see `hparams/`). Key fields:

- `stage`: `pt` | `sft` | `rm` | `ppo` | `dpo` | `kto` | `orpo`
- `finetuning_type`: `full` | `freeze` | `lora`
- `model_name_or_path`: HuggingFace model id or local path
- `dataset`: comma-separated names from `data/dataset_info.json`
- `template`: chat template name (e.g., `qwen3`, `llama3`, `mistral`)
- `deepspeed`: path to a DeepSpeed config JSON (examples in `examples/deepspeed/`)

### Dataset registration

Custom datasets are registered in `data/dataset_info.json`. Each entry maps a dataset name to a file path and optional column remapping. Supported formats: `alpaca` (default) and `sharegpt`. New datasets should be added here before referencing them in a training config.

### Adding a new model

1. Add the model's chat template to `data/template.py` as a new `Template` subclass.
2. If the model has a non-standard architecture, add patches in `model/model_utils/` and register them in `model/patcher.py`.
3. Register any custom projector/module names in `model/model_utils/visual.py` (for VLMs).

### Multi-modal support

Each model family's visual preprocessor is wrapped by a `BasePlugin` subclass in `data/mm_plugin.py`. Templates reference these via `mm_plugin`. To add a new VLM, implement a plugin and hook it into the relevant template.

---

## Code style

Follows Google Python Style Guide. `ruff` enforces formatting (line length 119, double quotes, Google-style docstrings). Run `make style` before committing. License header is required on all source files — `make license` checks compliance.
