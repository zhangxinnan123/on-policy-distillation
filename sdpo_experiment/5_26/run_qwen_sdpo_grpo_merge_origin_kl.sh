#!/usr/bin/env bash
# SDPO+GRPO on deepmath, MERGE variant + origin-KL constraint:
#   reward      = outcome - β_init · KL(π_cur(y|x) ‖ π_init(y|x))
#                                            (origin KL via use_kl_in_reward=True)
#   advantage   = compute_sdpo_grpo_advantage with merge_into_group=True
#                 per-uid regime dispatch on combined (main+hint) batch:
#                   A: no hint succeeds          -> vanilla GRPO on main only
#                   B: hint+main both succeed    -> SDPO sign-flip on main only
#                   C: stuck student (all main fail, hint succeeds)
#                                                -> GRPO with group = n + n_hint
#                                                   (both halves get gradient)
#
# Difference from run_qwen_sdpo_grpo_signflip_origin_kl2.sh: that script keeps
# hint rollouts OUT of the group (legacy path, hint only labels solvability).
# This script lets hint rows enter the group so regime C can salvage gradient
# on prompts where the student is stuck but the hint solves.
#
# Two refs are computed per step and stashed in distinct batch keys:
#   sdpo_ref_log_prob = π_cur(y|x, e)   (current actor, hinted)   -> sign-flip
#   ref_log_prob      = π_init(y|x)     (frozen, un-hinted)       -> origin KL in reward
#
# Knobs:
#   ORIGIN_KL_COEF  -> verl kl_ctrl.kl_coef (β_init, default 0.001)
#   ORIGIN_KL_TYPE  -> kl_penalty type for origin KL (default low_var_kl)
#   INIT_REF_MODEL  -> frozen anchor path (default = STUDENT_MODEL)
#   SDPO_SIGN_FLIP_LAMBDA_POS/NEG/EPSILON -> sign-flip (hint) knobs
#   N_HINT          -> number of hint rollouts per prompt (default 1)
#
# Requires the sdpo_grpo wiring in verl/trainer/ppo/sdpo_ray_trainer.py.
set -xeuo pipefail
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

############################ Quick Config ############################

ROLLOUT_NAME="vllm"

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_sdpo}
# EXP_NAME is auto-derived AFTER knobs are read (see below); override via
# `EXP_NAME=... bash ...` if you want a custom name.
DATE_TAG=${DATE_TAG:-5_26}

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=8192
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=256

TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=4
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=4

SP=1
ENFORCE_EAGER=True

############################ Paths ############################
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-${HOME}/data/deepmath_diff6to8}
TEST_DATA_DIR=${TEST_DATA_DIR:-${HOME}/data/dapo_17k_aime2426-suffix}
TRAIN_PARQUET=${TRAIN_PARQUET:-${TRAIN_DATA_DIR}/train.parquet}
TEST_PARQUET=${TEST_PARQUET:-${TEST_DATA_DIR}/test.parquet}

TRAIN_FILES="['${TRAIN_PARQUET}']"
TEST_FILES="['${TEST_PARQUET}']"

# Sampling
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
GRPO_N=${GRPO_N:-8}            # main rollouts per prompt (for advantage)
N_HINT=${N_HINT:-1}            # hint rollouts per prompt (merged into batch, regime dispatch in compute_sdpo_grpo_advantage)

############################ SDPO knobs (hint via sign-flip) ############################
EXPERT_SOURCE=${EXPERT_SOURCE:-dataset}
EXPERT_KEY=${EXPERT_KEY:-generations_wo_think}
EXPERT_INDEX=${EXPERT_INDEX:-0}

# Sign-flip token reweighting (compute_sdpo_grpo_advantage), regime B:
#   delta    = log pi_theta(y|x,e) - log pi_theta(y|x)   (current actor)
#   w_t      = exp(sign(A) * delta)
#   lambda_t = lambda_pos if A > 0 else lambda_neg
#   scale    = (1 - lambda_t) + lambda_t * clip(w_t, 1-eps, 1+eps)
#   advantage = grpo_adv * scale * response_mask
# Defaults to ON (lambda_pos=0.5, asymmetric) since this script's whole point
# is to combine sign-flip hint with origin-KL drift control.
SDPO_SIGN_FLIP_LAMBDA_POS=${SDPO_SIGN_FLIP_LAMBDA_POS:-0.3}
SDPO_SIGN_FLIP_LAMBDA_NEG=${SDPO_SIGN_FLIP_LAMBDA_NEG:-0.3}
SDPO_SIGN_FLIP_EPSILON=${SDPO_SIGN_FLIP_EPSILON:-0.2}

############################ Origin-KL-in-reward knobs ############################
# Standard verl apply_kl_penalty path:
#   token_level_rewards = token_level_scores - β_init * KL(π_cur || π_init) * mask
# Reads `ref_log_prob` (frozen un-hinted) from the batch, written by the second
# ref pass in _compute_ref_log_prob (gated on use_kl_in_reward=True OR use_kl_loss=True).
ORIGIN_KL_COEF=${ORIGIN_KL_COEF:-0.001}
ORIGIN_KL_TYPE=${ORIGIN_KL_TYPE:-low_var_kl}
INIT_REF_MODEL=${INIT_REF_MODEL:-${STUDENT_MODEL}}

# Derive explicit EXP_NAME from knobs so each wandb run is self-describing.
# `merge` tag distinguishes from the no-merge sibling script.
EXP_NAME=${EXP_NAME:-qwen_deepmath_sdpo_merge_n${N_HINT}_lpos${SDPO_SIGN_FLIP_LAMBDA_POS}_lneg${SDPO_SIGN_FLIP_LAMBDA_NEG}_kl${ORIGIN_KL_COEF}_${DATE_TAG}}

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
    # Origin KL goes to REWARD via use_kl_in_reward=True, not to loss.
    # Sign-flip handles the hint side, also at the advantage level.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
)

REF=(
    +actor_rollout_ref.ref.model.path="${INIT_REF_MODEL}"
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
    # Origin KL goes through apply_kl_penalty -> subtract β·KL(cur||init)
    # from token_level_rewards. ref_log_prob (un-hinted, frozen) is written
    # by _compute_ref_log_prob's second pass; sign-flip uses sdpo_ref_log_prob
    # which is a different key, so the two mechanisms don't double-count.
    algorithm.use_kl_in_reward=True
    algorithm.kl_penalty=${ORIGIN_KL_TYPE}
    algorithm.kl_ctrl.kl_coef=${ORIGIN_KL_COEF}
)

# SDPO subtree is not part of the bundled schema; prefix with `+` to add new keys.
SDPO=(
    +sdpo.modify_ref_prompt.enabled=True
    # Current actor as the SDPO ref. This drives the second-pass logic in
    # _compute_ref_log_prob: when use_current_actor=True AND
    # (use_kl_loss OR use_kl_in_reward) AND RefPolicy is a real worker
    # (not aliased to actor), an extra un-hinted forward through RefPolicy
    # writes init student logprob into ref_log_prob for origin KL.
    +sdpo.modify_ref_prompt.use_current_actor=True
    +sdpo.modify_ref_prompt.source=${EXPERT_SOURCE}
    +sdpo.modify_ref_prompt.dataset_key=${EXPERT_KEY}
    +sdpo.modify_ref_prompt.expert_index=${EXPERT_INDEX}

    +sdpo.hint_rollout.enabled=True
    +sdpo.hint_rollout.n_hint=${N_HINT}
    # Merge hint rows into the main batch (size = n + n_hint per uid) and let
    # compute_sdpo_grpo_advantage dispatch per uid by regime:
    #   A: no hint succeeds          -> vanilla GRPO on main only (group=n)
    #   B: hint+main both succeed    -> SDPO sign-flip on main only (group=n)
    #   C: all main fail, hint OK    -> GRPO with group=n+n_hint (both halves
    #                                   get gradient; hint rescues stuck student)
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
