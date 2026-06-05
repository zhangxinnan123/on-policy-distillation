#!/usr/bin/env bash
# Update-consistency — B=1 single-token, lr=1e-5, 50 steps (bf16-floor CONTROL).
# Same single-token setup as the lr=1e-3 anchor, but lr dropped 100x to 1e-5
# to test whether a smaller (more perturbative) step improves agreement with
# the tabular theory.
#
# EXPECTED (from the lr=1e-3 run): median |Δπ(c)| was ~1.2e-5 at lr=1e-3, so at
# lr=1e-5 the typical signal is ~1e-7 — well BELOW the bf16 LSB (~6e-5 in prob
# space). Sign agreement should collapse toward ~50% for BOTH u and c because
# Δlogit gets quantized to 0. This run DOCUMENTS the bf16 floor; it is not a
# clean perturbative-regime test (that needs fp32 logprob measurement).
#
# Diff vs run_update_consistency_singletoken_b1_lr1e3_50steps_1.7b_8b.sh:
#   only LR bumped from 1e-3 to 1e-5.

set -xeuo pipefail

source "$(dirname "$0")/../config.sh"

ROLLOUT_NAME="vllm"
FAMILY="Qwen"
STUDENT_MODEL=Qwen/Qwen3-1.7B-Base
TEACHER_MODEL=Qwen/Qwen3-1.7B

# Pinned for the theory test.
DISTILLATION_LOSS_MODE="k1_topk_overlap"   # k1 loss + topK diagnostics
USE_POLICY_GRADIENT=True                    # REINFORCE single-sample RKL
USE_FUSED_KERNELS=False

DISTILLATION_TOPK=20   # K candidates per response position; candidate set = K + sampled u
TOTAL_TRAINING_STEPS=50  # long run for robust per-token theory test statistics
MEASURE_EVERY=1
OPTIMIZER=SGD
LR=1e-5   # back to original; EXPECT bf16 quantization floor to dominate (sign → chance)

# ULTIMATE single-token: B=1 sequence, dp=1 GPU, R=64 → exactly 1 token contributes.
MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=64
VAL_MAX_RESPONSE_LENGTH=64
MAX_NUM_TOKENS=$(( MAX_PROMPT + VAL_MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=2
TRAIN_PROMPT_BSZ=1

STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=1
TEACHER_RESOURCE_POOL=False
TEACHER_WORLD_SIZE=1
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

# Measurement output (parquet + figures).
UC_OUTPUT_DIR="${HOME}/data/update_consistency/$(basename "$0" .sh)"
mkdir -p "${UC_OUTPUT_DIR}"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.8

############################ Parameter Groups ############################

UPDATE_CONSISTENCY=(
    update_consistency.output_dir="${UC_OUTPUT_DIR}"
    update_consistency.measure_every=$MEASURE_EVERY
    update_consistency.single_token_mode=true
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
    # Theory predicts per-optimizer-step Δπ — pin one optimizer step per
    # trainer iteration. ppo_epochs=1 + mini_batch=train_batch_size + dynamic
    # bsz means update_actor dispatch == one optimizer step.
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    # 1.7B fits on a single H100/A100, no need to offload.
    actor_rollout_ref.actor.fsdp_config.param_offload=False
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
    # With B=1, agent_loop must use exactly 1 worker (else prompts.chunk()
    # fails because batch_size % num_workers != 0).
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
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

echo "====== Update Consistency (ULTIMATE B=1 dp=1 single-token) ======"
echo "STUDENT_MODEL:         $STUDENT_MODEL"
echo "TEACHER_MODEL:         $TEACHER_MODEL"
echo "OPTIMIZER:             $OPTIMIZER"
echo "LR:                    $LR"
echo "STUDENT_WORLD_SIZE:    $STUDENT_WORLD_SIZE  (single GPU)"
echo "TRAIN_PROMPT_BSZ:      $TRAIN_PROMPT_BSZ  (single sequence)"
echo "MAX_RESPONSE_LENGTH:   $MAX_RESPONSE_LENGTH"
echo "single_token_mode:     TRUE — exactly 1 token contributes to gradient"
echo "TOTAL_TRAINING_STEPS:  $TOTAL_TRAINING_STEPS"
echo "MEASURE_EVERY:         $MEASURE_EVERY"
echo "UC_OUTPUT_DIR:         $UC_OUTPUT_DIR"
echo "DATA_PATH:             $DATA_PATH"
echo "==================================================================="

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
