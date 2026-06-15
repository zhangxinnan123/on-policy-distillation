#!/usr/bin/env bash
# SDPO+GRPO on deepmath, FROZEN-TEACHER variant.
#
# Same outcome-only GRPO baseline + hint-rollout solvable mask as
# run_qwen_sdpo_grpo.sh, but the SDPO sign-flip uses a FROZEN teacher for
# the hinted logprob:
#     ref_log_prob = log pi_teacher(y|x, e)   (Role.RefPolicy, fixed weights)
#     delta        = ref_log_prob - log pi_student(y|x)
# Teacher checkpoint is read from actor_rollout_ref.ref.model.path
# (defaults to the student's model.path if unset). The trainer spins up a
# real Role.RefPolicy worker because need_reference_policy() now triggers
# on `sdpo.modify_ref_prompt.enabled=True AND use_current_actor=False`.
#
# Requires the sdpo_grpo wiring in verl/trainer/ppo/sdpo_ray_trainer.py.
set -xeuo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}

############################ Quick Config ############################

ROLLOUT_NAME="vllm"

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}
# Frozen teacher checkpoint. Defaults to a snapshot of the student so this
# script is self-contained; override to a stronger teacher when available.
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-1.7B}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_sdpo}
EXP_NAME=${EXP_NAME:-qwen_deepmath_sdpo_grpo_frozen_teacher}

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=8192
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=256

TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")

SP=1
ENFORCE_EAGER=True

############################ Paths ############################
TRAIN_PARQUET=${TRAIN_PARQUET:-/mnt/data1/zhan9359/data/deepmath_diff6to8/train.parquet}
TEST_PARQUET=${TEST_PARQUET:-/mnt/data1/zhan9359/data/dapo_17k_aime2426-suffix/test.parquet}

TRAIN_FILES="['${TRAIN_PARQUET}']"
TEST_FILES="['${TEST_PARQUET}']"

# Sampling
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
GRPO_N=${GRPO_N:-8}            # main rollouts per prompt (for advantage)
# Hint rollouts per prompt. They are concatenated into the main batch and
# scored alongside, so each uid has GRPO_N main rows + N_HINT hint rows in
# the merged batch (see merge_into_group flag in the SDPO block below).
N_HINT=${N_HINT:-2}

############################ SDPO knobs ############################
# Expert text source: dataset (looked up first in extra_info[<key>], then as a
# top-level non_tensor column with the same name) or rollout (model's own response).
# For hint_rollout, source MUST be "dataset" (no responses pre-rollout).
#
# deepmath_diff6to8 preprocessing stores expert trajectories in the TOP-LEVEL
# column `generations_wo_think` (list-of-strings per row), NOT in extra_info.
# `EXPERT_INDEX` selects which list element to use as the expert (default 0).
EXPERT_SOURCE=${EXPERT_SOURCE:-dataset}
EXPERT_KEY=${EXPERT_KEY:-generations_wo_think}
EXPERT_INDEX=${EXPERT_INDEX:-0}

# Sign-flip token reweighting (compute_sdpo_grpo_advantage), regime B only:
#   delta    = log pi_teacher(y|x,e) - log pi_student(y|x)
#   w_t      = exp(sign(A) * delta)                  # per-token
#   lambda_t = lambda_pos if A > 0 else lambda_neg
#   scale    = (1 - lambda_t) + lambda_t * clip(w_t, 1-eps, 1+eps)
#   advantage = grpo_adv * scale * response_mask
#
# Defaults are 0.0 == OFF, so the first run exercises only the merge_into_group
# regime dispatch (A: GRPO on main; B: GRPO on main; C: GRPO with group=n+n_hint).
# Turn KL on later via env: e.g. SDPO_SIGN_FLIP_LAMBDA_POS=0.3 bash ... .
SDPO_SIGN_FLIP_LAMBDA_POS=${SDPO_SIGN_FLIP_LAMBDA_POS:-0.0}
SDPO_SIGN_FLIP_LAMBDA_NEG=${SDPO_SIGN_FLIP_LAMBDA_NEG:-0.0}
SDPO_SIGN_FLIP_EPSILON=${SDPO_SIGN_FLIP_EPSILON:-0.2}

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
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
    # KL is recombined inside sdpo_grpo advantage; do NOT add it to the loss too.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
)

REF=(
    +actor_rollout_ref.ref.model.path="${TEACHER_MODEL}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS
    actor_rollout_ref.rollout.n=$GRPO_N
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=8
)

ALGORITHM=(
    algorithm.adv_estimator=sdpo_grpo
    # MUST be False: sdpo_grpo asserts this and recombines KL inside the estimator.
    algorithm.use_kl_in_reward=False
)

# SDPO subtree is not part of the bundled schema; prefix with `+` to add new keys.
SDPO=(
    +sdpo.modify_ref_prompt.enabled=True
    # FROZEN teacher: query Role.RefPolicy (loaded from
    # actor_rollout_ref.ref.model.path) with the hinted prompt, NOT the actor.
    +sdpo.modify_ref_prompt.use_current_actor=False
    +sdpo.modify_ref_prompt.source=${EXPERT_SOURCE}
    +sdpo.modify_ref_prompt.dataset_key=${EXPERT_KEY}
    +sdpo.modify_ref_prompt.expert_index=${EXPERT_INDEX}

    +sdpo.hint_rollout.enabled=True
    +sdpo.hint_rollout.n_hint=${N_HINT}
    # Concatenate hint rollouts onto the main training batch and let
    # compute_sdpo_grpo_advantage dispatch per uid by regime:
    #   A: no hint succeeds          → vanilla GRPO on main only (group=n)
    #   B: hint+main both succeed    → SDPO sign-flip on main only (group=n)
    #   C: stuck student (all main fail, hint succeeds) → GRPO with group=n+n_hint
    +sdpo.hint_rollout.merge_into_group=True

    +sdpo.sdpo_grpo.sign_flip_lambda_pos=${SDPO_SIGN_FLIP_LAMBDA_POS}
    +sdpo.sdpo_grpo.sign_flip_lambda_neg=${SDPO_SIGN_FLIP_LAMBDA_NEG}
    +sdpo.sdpo_grpo.sign_flip_epsilon=${SDPO_SIGN_FLIP_EPSILON}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=40
    trainer.total_epochs=5
    trainer.total_training_steps=200
    trainer.val_before_train=False
    # SDPO modify_ref_prompt / use_current_actor only run in the LEGACY worker
    # branch of _compute_ref_log_prob. Must NOT be "disable".
    trainer.use_legacy_worker_impl=auto
    trainer.resume_mode=disable
    trainer.log_val_generations=5
)

############################ Launch ############################

export VLLM_USE_V1=1
export WANDB_ENTITY=${WANDB_ENTITY:-rl_agent}

python3 -m verl.trainer.main_sdpo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${SDPO[@]}" \
    "${MODEL[@]}" \
    "${STUDENT[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"
