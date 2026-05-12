#!/bin/bash
#SBATCH --job-name=opd_1.7b_queue_2_0511
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

HERE="/home/jasonjx/on-policy-distillation"

SCRIPTS=(
    "jx_experiment/5_11/run_1.7b_base_8b_opd_nonthink_opdt_4.slurm"
    "jx_experiment/5_11/run_1.7b_base_8b_opd_nonthink_opdt_5.slurm"
    "jx_experiment/5_11/run_1.7b_base_8b_opd_nonthink_opdt_6.slurm"
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
echo "Summary (1.7b queue 2):"
printf '  %s\n' "${RESULTS[@]}"
echo "=================================================================="
