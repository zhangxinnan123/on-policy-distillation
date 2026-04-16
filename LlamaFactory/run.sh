
cd LlamaFactory
export WANDB_API_KEY="c4b67c713ad88ef65b62908bcaa8b5c5cb72d1a9"
export WANDB_ENTITY="${WANDB_ENTITY:-rl_agent}"

OUTPUT_DIR="saves/OpenThinker3-1.7B-Base-SFT"
if ls "$OUTPUT_DIR"/checkpoint-* >/dev/null 2>&1; then
    echo "Found existing checkpoint in $OUTPUT_DIR, resuming..."
    RESUME_FLAG="resume_from_checkpoint=true"
else
    echo "No checkpoint found, starting fresh..."
    RESUME_FLAG="resume_from_checkpoint=false"
fi

ALLOW_EXTRA_ARGS=1 llamafactory-cli train examples/train_full/openthinker3.yaml "$RESUME_FLAG"
