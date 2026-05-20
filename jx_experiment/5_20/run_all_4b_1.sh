#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

SCRIPTS=(
    "run_4b_base_8b_opd_nonthink_opdt_1.sh"
    "run_4b_base_8b_opd_nonthink_opdt_2.sh"
    "run_4b_base_8b_opd_nonthink_opdt_3.sh"
    "run_4b_base_8b_opd_nonthink_opdt_4.sh"
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
echo "Summary (4b queue):"
printf '  %s\n' "${RESULTS[@]}"
echo "=================================================================="
