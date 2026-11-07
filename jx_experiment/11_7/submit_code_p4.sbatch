#!/bin/bash
# Run one 11_7 coding arm on p4 (poe-skywalker), single node. Pass the arm via RUN=.
#
#   RUN=jx_experiment/11_7/run_1p7b_base_code_opdt4_p4.sh \
#     sbatch --export=ALL,RUN=$RUN --job-name=opdt4_1p7b jx_experiment/11_7/submit_code_p4.sbatch
#
# Smoke test first (3 steps, validation on, so the executing reward path is
# exercised end to end before committing 8 long jobs):
#   RUN=... TOTAL_STEPS=3 TEST_FREQ=2 \
#     sbatch --export=ALL,RUN=$RUN,TOTAL_STEPS=3,TEST_FREQ=2 --job-name=smoke ...
#
# Derived from 9_16/submit_p4_single.sbatch; every workaround below is carried
# over from a failure that script records, not added speculatively.
#SBATCH --job-name=opd_code
#SBATCH --partition=ml-p4d-24xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/xinnanzh/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%j.err

set -xeu

source ~/miniconda3/etc/profile.d/conda.sh
# `verl` (transformers 4.56.1), not `opd`: this cluster's `opd` has transformers
# 5.x, and vLLM 0.11.0 reads tokenizer.all_special_tokens_extended, which v5
# removed.
conda activate verl

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
# Students, teacher and tokenizer are all local by now, but the eval parquet was
# built offline and nothing in the training loop should reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online
export WANDB_ENTITY=rl_agent
# Online mode makes wandb init a hard dependency; a job once died after 45 min on
# the 90s default before running a single step.
export WANDB_INIT_TIMEOUT=600

# Code-execution reward settings (jx_experiment/11_7/code_reward.py).
#
# Concurrency is PER PROCESS. reward.num_workers=8 RewardLoopWorker actors each
# import code_reward.py, so total concurrent child interpreters is 6 x 8 = 48
# against 96 vCPUs -- half the node, leaving the rest for the Ray driver and the
# resident vLLM engines. This is the knob that sets how long a validation takes:
# ~5.7k generations must each be executed.
export OPD_CODE_REWARD_CONCURRENCY="${OPD_CODE_REWARD_CONCURRENCY:-6}"
export OPD_CODE_REWARD_TIMEOUT="${OPD_CODE_REWARD_TIMEOUT:-25}"

RUN="${RUN:?set RUN=<path to run script>}"

unset RAY_ADDRESS
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray/ray_current_cluster

cd /fsx/xinnanzh/on-policy-distillation
test -f "$RUN"
test -f "$HOME/data/code_opd/test.parquet"

############################ Reap orphaned vLLM engines ############################
# A node can still hold VLLM::EngineCore processes from a run days earlier (8 such
# orphans, each pinning 1414MiB of CUDA context, killed job 7630). `ray stop
# --force` does not touch them: they are not Ray workers and they outlive the
# Slurm job that spawned them.
#
# Safe unconditionally here -- this runs before this job starts any vLLM, so every
# match is a leftover, and -u restricts it to our own processes. Match on comm,
# NOT cmdline: the kernel truncates comm to 15 chars ("VLLM::EngineCor"), and
# `pgrep -f` would match this very command line, which is the self-match trap
# that once killed a live ssh session.
n=$(pgrep -u "$(id -u)" -cx "VLLM::EngineCor" || true)
if [ "${n:-0}" -gt 0 ]; then
    echo "$(hostname): reaping $n orphaned VLLM::EngineCore"
    pkill -u "$(id -u)" -x "VLLM::EngineCor" || true
    sleep 5
    pkill -9 -u "$(id -u)" -x "VLLM::EngineCor" || true
else
    echo "$(hostname): no orphaned vLLM engines"
fi

############################ Reap orphaned reward sandboxes ############################
# Same class of leftover, different process: code_exec_runner.py children from a
# previous validation that blocked on I/O and outlived their parent. Each holds a
# Python interpreter. Matched on the script name via -f, which is safe for this
# pattern because this sbatch's own command line does not contain it.
pkill -u "$(id -u)" -f "code_exec_runner.py" 2>/dev/null || true

bash "$RUN"
