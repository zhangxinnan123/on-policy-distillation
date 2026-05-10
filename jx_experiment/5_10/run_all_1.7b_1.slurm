#!/bin/bash
#SBATCH --job-name=opd_all_1.7b_0510
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --output='/home/jasonjx/on-policy-distillation/logs/slurm_outputs/%x_%j.out'
#SBATCH --account=mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest

set -uo pipefail

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

cd /home/jasonjx/on-policy-distillation

# Job information
echo "=========================================="
echo "On-Policy Distillation Job (5_10 All 1.7B queue)"
echo "=========================================="
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Total GPUs: 4 (1 node x 4 GPUs)"
echo "=========================================="

HERE="/home/jasonjx/on-policy-distillation"

SCRIPTS=(
    "jx_experiment/5_10/run_1.7b_base_8b_opd_nonthink_partial.slurm"
    "jx_experiment/5_10/run_1.7b_base_8b_opd_nonthink_partial_nocurriculum.slurm"
    "jx_experiment/5_10/run_1.7b_base_8b_opd_nonthink.slurm"
    "jx_experiment/5_10/run_1.7b_base_8b_opd_nonthink_subeos.slurm"
    "jx_experiment/5_10/run_1.7b_SFT_8b_opd_nonthink.slurm"
)

declare -a RESULTS=()
for s in "${SCRIPTS[@]}"; do
    echo "=================================================================="
    echo "[$(date -Is)] STARTING: $s"
    echo "=================================================================="
    if bash "$HERE/$s"; then
        RESULTS+=("OK    $s")
    else
        rc=$?
        RESULTS+=("FAIL($rc) $s")
        echo "[$(date -Is)] $s exited with $rc — continuing to next."
    fi
done

echo "=================================================================="
echo "Summary (1.7b queue):"
printf '  %s\n' "${RESULTS[@]}"
echo "=================================================================="
