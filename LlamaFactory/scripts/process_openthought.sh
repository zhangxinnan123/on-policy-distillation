python scripts/process_sft_data.py \
    --output_file data/openthoughts3_math_50k8_complete.parquet \
    --num_questions 50000 \
    --num_continuations 8 \
    --tokenizer Qwen/Qwen3-1.7B \
    --push_to_hub XinnanZhang/openthoughts3-math-50k8 \
    --no_completeness_filter

