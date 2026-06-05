#!/usr/bin/env bash
# Update-consistency — BATCH MODE (all tokens contribute) — B=32, lr=1e-3.
# Larger-batch test of the per-token theory degradation.
#
# Reference points already in:
#   B=1 single-token (50 steps): sampled u sign agreement → 100% at |A|>0.1
#   B=8 batched      (10 steps): sampled u sign agreement → 50% (chance)
#
# B=32 prediction:
#   With 4x more tokens contributing simultaneously through the shared LM head,
#   each token's "own" gradient is 1/(32*~50) ≈ 1/1600 of the total batch
#   gradient. NTK off-diagonal contamination dominates even more strongly.
#   Sign agreement should remain ≈ 50% (chance). Magnitude bias (×3500
#   underestimate) should be similar to B=8 since backbone NTK contribution
#   is per-position, not per-batch.
#
# Diff vs run_update_consistency_batch_b8_lr1e3_10steps_1.7b_8b.sh:
#   - TRAIN_PROMPT_BSZ: 8 → 32
#   - MAX_NUM_SEQS: 8 → 32 (rollout needs to handle 32 prompts)

set -xeuo pipefail

source "$(dirname "$0")/../config.sh"

ROLLOUT_NAME="vllm"
FAMILY="Qwen"
STUDENT_MODEL=Qwen/Qwen3-1.7B-Base
TEACHER_MODEL=Qwen/Qwen3-1.7B

# Pinned for the theory test.
DISTILLATION_LOSS_MODE="k1_topk_overlap"
USE_POLICY_GRADIENT=True
USE_FUSED_KERNELS=False

DISTILLATION_TOPK=20
TOTAL_TRAINING_STEPS=10
MEASURE_EVERY=1
OPTIMIZER=SGD
LR=1e-3

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=64
VAL_MAX_RESPONSE_LENGTH=64
MAX_NUM_TOKENS=$(( MAX_PROMPT + VAL_MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=32
TRAIN_PROMPT_BSZ=32

STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=4
TEACHER_RESOURCE_POOL=False
TEACHER_WORLD_SIZE=4
SP=1
ROLLOUT_N=1
train_batch_size=$(( TRAIN_PROMPT_BSZ ))

EXP_NAME="update_consistency/student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}_pg/opt-${OPTIMIZER}_topk${DISTILLATION_TOPK}_b${train_batch_size}_lr${LR}_reslen${MAX_RESPONSE_LENGTH}_steps${TOTAL_TRAINING_STEPS}_every${MEASURE_EVERY}"

ENFORCE_EAGER=True

############################ Paths ############################
DATA_PATH="${HOME}/data/dapo_17k_aime2426-suffix"
DAPO_TRAIN_PATH=$DATA_PATH/train.parquet
DAPO_TEST_PATH=$DATA_PATH/test.parquet
TRAIN_FILES="['$DAPO_TRAIN_PATH']"
TEST_FILES="['$DAPO_TEST_PATH']"

UC_OUTPUT_DIR="${HOME}/data/update_consistency/$(basename "$0" .sh)"
mkdir -p "${UC_OUTPUT_DIR}"

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.8

############################ Parameter Groups ############################

UPDATE_CONSISTENCY=(
    update_consistency.output_dir="${UC_OUTPUT_DIR}"
    update_consistency.measure_every=$MEASURE_EVERY
    update_consistency.single_token_mode=false
    update_consistency.seed=0
)

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.num_workers=8
    distillation.teacher_model.enable_resource_pool=$TEACHER_RESOURCE_POOL
    distillation.teacher_model.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_model.inference.gpu_memory_utilization=0.5
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=$MAX_NUM_SEQS
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=$DISTILLATION_TOPK
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.loop_metrics.enabled=False
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.optim.optimizer=$OPTIMIZER
    actor_rollout_ref.actor.optim.weight_decay=0.0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
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
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
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

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=1
    trainer.log_val_generations=0
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS
)

############################ Launch ############################

export VLLM_USE_V1=1
export WANDB_API_KEY=$WANDB_API_KEY

echo "============== Update Consistency (BATCH MODE, B=32, 10 steps) =============="
echo "STUDENT_MODEL:         $STUDENT_MODEL"
echo "TEACHER_MODEL:         $TEACHER_MODEL"
echo "OPTIMIZER:             $OPTIMIZER"
echo "LR:                    $LR"
echo "single_token_mode:     FALSE — all response tokens contribute to gradient"
echo "TRAIN_PROMPT_BSZ:      $TRAIN_PROMPT_BSZ  (4x B=8)"
echo "MAX_RESPONSE_LENGTH:   $MAX_RESPONSE_LENGTH"
echo "→ ~$((TRAIN_PROMPT_BSZ * MAX_RESPONSE_LENGTH)) tokens/step"
echo "TOTAL_TRAINING_STEPS:  $TOTAL_TRAINING_STEPS"
echo "MEASURE_EVERY:         $MEASURE_EVERY"
echo "UC_OUTPUT_DIR:         $UC_OUTPUT_DIR"
echo "DATA_PATH:             $DATA_PATH"
echo "=============================================================================="

python3 -m recipe.update_consistency.main \
    --config-name=measure_during_opd \
    "${UPDATE_CONSISTENCY[@]}" \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
