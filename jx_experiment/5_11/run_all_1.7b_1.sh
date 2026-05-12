#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

SCRIPTS=(
    "jx_experiment/5_11/run_1.7b_base_8b_opd_nonthink_opdt_1.sh"
    "jx_experiment/5_11/run_1.7b_base_8b_opd_nonthink_opdt_2.sh"
    "jx_experiment/5_11/run_1.7b_base_8b_opd_nonthink_opdt_3.sh"
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
