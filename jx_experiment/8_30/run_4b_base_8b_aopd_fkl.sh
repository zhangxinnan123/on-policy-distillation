#!/usr/bin/env bash
set -xeuo pipefail

############################ AOPD (Advantage-guided OPD) ############################
# Routing (verl/trainer/distillation/hybrid_masks.py:_mask_aopd):
#   A = log π_T(u) − log π_S(u)
#   A >= 0  → PG (k1)         student under-confident on sampled token
#   A <  0  → FKL (sup)       student over-confident — steer via teacher top-k
#
# Motivation: our stats show A<0 + high FKL positions are the main "learn-worse"
# regime under REINFORCE-RKL. Redirecting them to FKL avoids the misdirected
# single-sample gradient.

source "$(dirname "$0")/../config.sh"

ROLLOUT_NAME="vllm"

FAMILY="Qwen"
STUDENT_MODEL=Qwen/Qwen3-4B-Base
TEACHER_MODEL=Qwen/Qwen3-8B

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1_pg_fkl_topk"   # use_hybrid=True — mask actually routes
USE_FUSED_KERNELS=False

HYBRID_MASK_STRATEGY="aopd"
HYBRID_MASK_AOPD_THRESHOLD=0.0

PG_LOSS_COEF=1.0
SUPERVISED_LOSS_COEF=1.0

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=4096
VAL_MAX_RESPONSE_LENGTH=8192
MAX_NUM_TOKENS=$(( MAX_PROMPT + VAL_MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=128
TRAIN_PROMPT_BSZ=128

STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=8
TEACHER_RESOURCE_POOL=False
TEACHER_WORLD_SIZE=8
# RERUN with the real hybrid loss. The original 8_9 run used k1_topk_overlap,
# which is registered use_hybrid=False (losses.py:1138) and never calls
# _build_hybrid_masks — the mask ran only as a diagnostic via
# _compute_pg_diag_for_non_hybrid, so the loss was plain all-token k1 PG and this
# was literally the same experiment as run_*_baseline_all_pg.sh. Switching to
# k1_pg_fkl_topk (use_hybrid=True) makes the mask actually route.
#
# SP=2 + gpu_memory_utilization=0.4 — the exact combination the 8_30 v4 runs
# (15734/15735/15736) completed 200 steps on.
#
# Why not SP=1, which is what 8_9 originally used: SP=1 now fails 8/8 in vLLM
# WorkerProc init (Engine core initialization failed, zero OOM lines — training
# never starts), at both gpu_mem 0.4 (jobs 15759-15762) and 0.5 (jobs
# 15766-15769). Since both gpu_mem values fail, the KV-cache-starvation theory is
# ruled out. SP=2 is 3/3 successful. 8_9 did run SP=1 fine, so this is
# environment drift (vllm/transformers were changed since), not a script bug.
#
# Ulysses SP is mathematically exact, so results stay comparable to the 8_9
# all-PG baseline. It also halves activation memory, which the FKL arm needs
# because it backprops through the student top-k logprobs — something
# k1_topk_overlap never did.
SP=2

ROLLOUT_N=1
LR=1e-6
train_batch_size=$(( TRAIN_PROMPT_BSZ ))

EXP_NAME="fsdp/student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}/mask-${HYBRID_MASK_STRATEGY}_thr${HYBRID_MASK_AOPD_THRESHOLD}_b${train_batch_size}_n${ROLLOUT_N}_lr${LR}_reslen${MAX_RESPONSE_LENGTH}_fklhybrid"

ENFORCE_EAGER=True

############################ Paths ############################
DATA_PATH="${HOME}/data/dapo_17k_aime2426-suffix"

DAPO_TRAIN_PATH=$DATA_PATH/train.parquet
DAPO_TEST_PATH=$DATA_PATH/test.parquet

TRAIN_FILES="['$DAPO_TRAIN_PATH']"
TEST_FILES="['$DAPO_TEST_PATH']"

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.8

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
    distillation.teacher_model.inference.gpu_memory_utilization=0.3
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=$MAX_NUM_SEQS
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=16
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
    distillation.distillation_loss.hybrid_mask_strategy=$HYBRID_MASK_STRATEGY
    distillation.distillation_loss.pg_loss_coef=$PG_LOSS_COEF
    distillation.distillation_loss.supervised_loss_coef=$SUPERVISED_LOSS_COEF
    +distillation.distillation_loss.hybrid_mask_kwargs.threshold=$HYBRID_MASK_AOPD_THRESHOLD
    distillation.loop_metrics.enabled=False
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=$LR
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
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
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=20
    trainer.total_epochs=2
    trainer.val_before_train=True
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=1
    trainer.log_val_generations=5
    trainer.total_training_steps=200
)

export VLLM_USE_V1=1
export WANDB_API_KEY=$WANDB_API_KEY

echo "================== AOPD threshold=$HYBRID_MASK_AOPD_THRESHOLD =================="
echo "DISTILLATION_LOSS_MODE: $DISTILLATION_LOSS_MODE"
echo "HYBRID_MASK_STRATEGY: $HYBRID_MASK_STRATEGY"
echo "HYBRID_MASK_AOPD_THRESHOLD: $HYBRID_MASK_AOPD_THRESHOLD (A cutoff)"
echo "  A >= thr  → PG (k1)      (student under-confident)"
echo "  A <  thr  → FKL (sup)    (student over-confident)"
echo "PG_LOSS_COEF: $PG_LOSS_COEF"
echo "SUPERVISED_LOSS_COEF: $SUPERVISED_LOSS_COEF"
echo "======================================================================="

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
