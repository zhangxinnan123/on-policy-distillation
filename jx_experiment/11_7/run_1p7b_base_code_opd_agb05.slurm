#!/bin/bash
# OPD (all-PG k1) on the coding task: Qwen3-1.7B-Base student <- Qwen3-8B teacher,
# on agb05 (MSI, partition mhong, 8x A100-40GB). The job IS the run, so the
# allocation cannot expire underneath training (run 1 was SIGTERMed at step 189
# when a separate keep-alive allocation ended).
#
#   sbatch jx_experiment/11_7/run_1p7b_base_code_opd_agb05.slurm
#
# Checkpoints every 50 steps (1 kept) + resume_mode=auto: resubmitting this file
# continues from the last checkpoint. Every validation generation is dumped to
# VAL_DUMP_DIR so harness changes can be re-scored offline.
#SBATCH --job-name=code_opd_1p7b
#SBATCH --partition=mhong
#SBATCH --account=mhong
#SBATCH --nodelist=agb05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:a100:8
#SBATCH --time=1-12:00:00
#SBATCH --output=/projects/standard/mhong/zhan9359/work/on-policy-distillation/verl/logs/%x_%j.log

set -xeuo pipefail
REPO=/projects/standard/mhong/zhan9359/work/on-policy-distillation/verl
cd "$REPO"

export EXP_SUFFIX="${EXP_SUFFIX:-_v2_val100}"
# Validate on a fixed stratified 100-prompt subset (HE+ 23 / MBPP+ 53 / LCB 24),
# 800 generations per validation instead of 5736. Built by subsample_eval.py.
export CODE_TEST_FILE=/projects/standard/mhong/zhan9359/data/code_opd/test_100.parquet
export SAVE_FREQ=50
export RESUME_MODE=auto
export VAL_DUMP_DIR="$REPO/logs/val_dump/code_opd_1p7b${EXP_SUFFIX}"
# 64 allocated CPUs: 4 sandboxes per reward worker x 8 workers = 32.
export OPD_CODE_REWARD_CONCURRENCY=4

echo "Host: $(hostname)  JOB: $SLURM_JOB_ID  GPUs: $CUDA_VISIBLE_DEVICES"
bash jx_experiment/11_7/launch_agb05.sh jx_experiment/11_7/run_1p7b_base_code_opd_p4.sh
