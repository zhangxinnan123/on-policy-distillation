#!/bin/bash

# Script to regenerate toward_teacher_grid plots with token visualization
# Usage: bash regenerate_with_tokens.sh <measurements_parquet> <model_name>
# Example: bash regenerate_with_tokens.sh data/measurements/measurements.parquet Qwen/Qwen3-1.7B

if [ $# -lt 2 ]; then
    echo "Usage: $0 <measurements_parquet> <model_name>"
    echo "Example: $0 data/measurements/measurements.parquet Qwen/Qwen3-1.7B"
    exit 1
fi

MEASUREMENTS="$1"
MODEL="$2"
OUTPUT_DIR="recipe/update_consistency/figures"

if [ ! -f "$MEASUREMENTS" ]; then
    echo "Error: Measurements file not found: $MEASUREMENTS"
    exit 1
fi

echo "Regenerating toward_teacher_grid plots with token visualization..."
echo "Measurements: $MEASUREMENTS"
echo "Model/Tokenizer: $MODEL"
echo "Output directory: $OUTPUT_DIR"

# Generate the grid plot with tokens
python -m recipe.update_consistency.analyze_toward_teacher \
    --measurements "$MEASUREMENTS" \
    --output-dir "$OUTPUT_DIR" \
    --per-token \
    --tokenizer "$MODEL" \
    --suffix "_b1"

# Also generate with linear scale
python -m recipe.update_consistency.analyze_toward_teacher \
    --measurements "$MEASUREMENTS" \
    --output-dir "$OUTPUT_DIR" \
    --per-token \
    --linear \
    --tokenizer "$MODEL" \
    --suffix "_b1"

# Generate with delta-pi floor filtering
python -m recipe.update_consistency.analyze_toward_teacher \
    --measurements "$MEASUREMENTS" \
    --output-dir "$OUTPUT_DIR" \
    --per-token \
    --delta-pi-floor 1e-3 \
    --tokenizer "$MODEL" \
    --suffix "_b1"

echo "Done! Check the output directory for new plots with '_tokens' suffix"