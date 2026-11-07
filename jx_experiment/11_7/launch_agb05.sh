#!/usr/bin/env bash
# Run one 11_7 coding arm directly on agb05 (8x A100-40GB, 16 CPUs, no Slurm).
#
#   ssh agb05 "cd $REPO && setsid nohup bash jx_experiment/11_7/launch_agb05.sh \
#       jx_experiment/11_7/run_1p7b_base_code_opd_p4.sh > logs/<name>.log 2>&1 &"
#
# Differences from submit_code_p4.sbatch (p4d, 96 vCPU) -- environment only, the
# training config in _common_code_p4.sh is untouched:
#   * env: the `seed` conda env (vllm 0.11.0 / torch 2.8 / transformers 4.57),
#     with this repo FIRST on PYTHONPATH -- `seed` has an editable install of a
#     different verl (jev-agentic-multi-turn) that Ray workers would otherwise
#     import -- and flash-attn 2.8.3 from a side dir, since `seed` lacks it.
#   * 16 physical CPUs: Ray is told 64 logical CPUs so the ~30 one-CPU actors
#     (reward/agent/teacher workers) can all be scheduled, and code-execution
#     concurrency is 1 per reward worker (8 sandboxes total) instead of 6.
set -xeuo pipefail

RUN="${1:?usage: launch_agb05.sh <run_*.sh> [extra hydra overrides]}"
shift

REPO=/projects/standard/mhong/zhan9359/work/on-policy-distillation/verl
ENV=/projects/standard/mhong/zhan9359/.conda/envs/seed
EXTRA=/projects/standard/mhong/zhan9359/pyextra/opd_seed
CONDA_BASE=/common/software/install/migrated/anaconda/python3-2020.07-mamba

export PATH="$ENV/bin:$CONDA_BASE/bin:$PATH"   # config.sh calls `conda info --base`
export PYTHONPATH="$REPO:$EXTRA${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/projects/standard/mhong/zhan9359/.cache/
# Models live directly under .cache/models--*, not .cache/hub/.
export HF_HUB_CACHE=/projects/standard/mhong/zhan9359/.cache
# NOT offline: transformers 4.57's _patch_mistral_regex calls the Hub API
# (model_info) for every non-local model id and raises OfflineModeIsEnabled.
# agb05 reaches huggingface.co; weights still come from the cache above.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export WANDB_INIT_TIMEOUT=600
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES RAY_ADDRESS

export OPD_CODE_REWARD_CONCURRENCY="${OPD_CODE_REWARD_CONCURRENCY:-1}"   # 1 for a 16-CPU allocation
export OPD_CODE_REWARD_TIMEOUT="${OPD_CODE_REWARD_TIMEOUT:-25}"

cd "$REPO"
test -f "$RUN"
python -c "import verl, flash_attn, vllm; assert verl.__file__.startswith('$REPO'), verl.__file__; print('verl', verl.__file__, 'vllm', vllm.__version__, 'flash_attn', flash_attn.__version__)"

ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray/ray_current_cluster
pkill -u "$(id -u)" -x "VLLM::EngineCor" || true
pkill -u "$(id -u)" -f "code_exec_runner.py" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

bash "$RUN" ray_kwargs.ray_init.num_cpus=64 "$@"
