#!/usr/bin/env bash
# End-to-end pipeline for the in-flight update-consistency experiment.
#
# Usage:
#   bash recipe/update_consistency/run.sh
#
# Stages:
#   1. train_and_measure → data/measurements/measurements.parquet
#   2. analyze           → figures/update_consistency/{*.pdf,*.png,*.csv}

set -euo pipefail

DATA_DIR="${UPDATE_CONSISTENCY_DATA_DIR:-data/update_consistency}"
MEASURE_DIR="${DATA_DIR}/measurements"
FIG_DIR="${DATA_DIR}/figures"

mkdir -p "${MEASURE_DIR}" "${FIG_DIR}"

MEASURE_EVERY=1
TOTAL_STEPS=10

# 1. Train + measure
python3 -m recipe.update_consistency.main \
    --config-name=measure_during_opd \
    update_consistency.output_dir="${MEASURE_DIR}" \
    update_consistency.measure_every="${MEASURE_EVERY}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    "$@"

# 2. Analyze
# python3 -m recipe.update_consistency.analyze \
#     --measurements "${MEASURE_DIR}/measurements.parquet" \
#     --output-dir "${FIG_DIR}"

echo "Done. See ${FIG_DIR}/ for the figure and ${MEASURE_DIR}/ for raw data."
