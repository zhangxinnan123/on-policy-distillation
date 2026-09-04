#!/usr/bin/env bash
# Eval-only run of Qwen3-8B (thinking mode) as reference/upper-bound on our test set.
#
# Env vars:
#   VAL_MAX_RESPONSE_LENGTH  (default 16384)  # override to 30719 for 32k run
#   STUDENT_MODEL            (default Qwen/Qwen3-8B)
#   TEST_PARQUET             (default test.parquet — aime24/25/26 + amc23;
#                             set to test_extended_no_amc.parquet for
#                             aime24/25/26 + hmmt_feb_2025)

set -xeuo pipefail

source "$(dirname "$0")/../config.sh"

ROLLOUT_NAME="vllm"

FAMILY="Qwen"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-8B}"

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=4096
VAL_MAX_RESPONSE_LENGTH="${VAL_MAX_RESPONSE_LENGTH:-16384}"
MAX_NUM_TOKENS=$(( MAX_PROMPT + VAL_MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=128
TRAIN_PROMPT_BSZ=128

STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=8
SP=1

ROLLOUT_N=1

# Qwen3 thinking-mode recommended sampling
val_top_p=0.95
val_top_k=20

TAG=$(basename "${STUDENT_MODEL}")
# Only suffix the run name for non-default test sets, so existing output paths
# (which all used test.parquet) stay exactly where they were.
if [[ -n "${TEST_PARQUET:-}" && "${TEST_PARQUET}" != "test.parquet" ]]; then
    TESTSET_SUFFIX="_$(basename "${TEST_PARQUET%.parquet}")"
else
    TESTSET_SUFFIX=""
fi
EXP_NAME="eval/${TAG}/val_max${VAL_MAX_RESPONSE_LENGTH}${TESTSET_SUFFIX}"

ENFORCE_EAGER=True

############################ Paths ############################
DATA_PATH="${HOME}/data/dapo_17k_aime2426-suffix"

DAPO_TRAIN_PATH=$DATA_PATH/train.parquet
DAPO_TEST_PATH=$DATA_PATH/${TEST_PARQUET:-test.parquet}

TRAIN_FILES="['$DAPO_TRAIN_PATH']"
TEST_FILES="['$DAPO_TEST_PATH']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    +data.apply_chat_template_kwargs.enable_thinking=True
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=False
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=8
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=${VAL_MAX_RESPONSE_LENGTH}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold="0.5_2.0"
    algorithm.rollout_correction.bypass_mode=False
)

VAL_DUMP_DIR="/fsx/xinnanzh/eval_out/${TAG}/val_max${VAL_MAX_RESPONSE_LENGTH}${TESTSET_SUFFIX}"
mkdir -p "$VAL_DUMP_DIR"

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=1
    trainer.total_epochs=0
    trainer.val_before_train=True
    trainer.val_only=True
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=0
    trainer.log_val_generations=10
    trainer.total_training_steps=0
    trainer.validation_data_dir="$VAL_DUMP_DIR"
)

export VLLM_USE_V1=1
export WANDB_API_KEY=$WANDB_API_KEY

echo "================== EVAL Qwen3-8B (thinking mode) =================="
echo "MODEL: $STUDENT_MODEL"
echo "VAL_MAX_RESPONSE_LENGTH: $VAL_MAX_RESPONSE_LENGTH"
echo "TEST: $DAPO_TEST_PATH"
echo "sampling: T=0.6 top_p=0.95 top_k=20 n=8"
echo "===================================================="

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
