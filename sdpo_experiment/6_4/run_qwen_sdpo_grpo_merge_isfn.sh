#!/usr/bin/env bash
# SDPO+GRPO on deepmath, MERGE + IS CORRECTION with f(w) squashing. No origin KL.
# IS weight w=π(y|x)/π(y|x,e) on regime-C hint rows is squashed by
# f(w)=w/(w+IS_FN_CONST) and applied as rollout_is_weights (biased but tunable
# rescue: lifts tiny weights, saturates large at 1). IS_FN_CONST=0 falls back to
# the strict-unbiased implicit-ratio mode.
#
# Hint rollouts are merged into the GRPO group (merge_into_group=True) with
# per-uid regime A/B/C. Regime C (student stuck, hint solves) feeds the hint
# response into the group as a positive example — but the hint was sampled from
# the HINTED behavior policy π_θ(·|x,e), not the deployment policy π_θ(·|x).
#
# To make regime C an UNBIASED policy gradient for the deployment objective,
# this script enables strict IS correction (sdpo.sdpo_grpo.is_correction):
#   - Before the actor update, hint rows' prompt is swapped back to UN-HINTED [x]
#     so the score function is ∇log π_θ(y_hint|x) (not ∇log π_θ(y_hint|x,e)).
#   - old_log_probs for hint rows stays the hinted behavior log π_θ(y_hint|x,e)
#     (computed before the swap); with actor.use_rollout_log_probs=True the PPO
#     ratio becomes π_θ(y|x)/π_θ(y|x,e) = the IS weight, bounded by PPO clip.
#   The gradient is then  A · [π(y|x)/π(y|x,e)] · ∇log π_θ(y|x)  — unbiased.
#
#   reward      = outcome   (no KL term; origin KL removed)
#   advantage   = compute_sdpo_grpo_advantage, merge_into_group=True, regime A/B/C
#   IS weight   = π_θ(y_hint|x)/π_θ(y_hint|x,e), via PPO ratio, PPO-clipped
#
# NOTE: stuck prompts (the rescue target) get small IS weights by construction,
# so the clip range (actor.clip_ratio) controls the rescue/variance tradeoff.
#
# Requires the sdpo_grpo + is_correction wiring in
# verl/trainer/ppo/sdpo_ray_trainer.py (_swap_hint_rows_to_unhinted).
set -xeuo pipefail
source "$(dirname "$0")/../config.sh"
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

############################ Quick Config ############################
ROLLOUT_NAME="vllm"

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_sdpo}
# EXP_NAME is auto-derived AFTER knobs are read (see below).
DATE_TAG=${DATE_TAG:-6_4}

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=8192
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=256

TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=4
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=8

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
GRPO_N=${GRPO_N:-8}            # main rollouts per prompt
N_HINT=${N_HINT:-1}           # hint rollouts per prompt (merged into group)

############################ SDPO knobs ############################
EXPERT_SOURCE=${EXPERT_SOURCE:-dataset}
EXPERT_KEY=${EXPERT_KEY:-generations_wo_think}
EXPERT_INDEX=${EXPERT_INDEX:-0}

# Sign-flip on regime B (main+hint both succeed). Independent of the IS
# correction, which only acts on regime C hint rows. Defaults OFF here so the
# run isolates the merge + IS mechanism; turn on via env to combine.
SDPO_SIGN_FLIP_LAMBDA_POS=${SDPO_SIGN_FLIP_LAMBDA_POS:-0.0}
SDPO_SIGN_FLIP_LAMBDA_NEG=${SDPO_SIGN_FLIP_LAMBDA_NEG:-0.0}
SDPO_SIGN_FLIP_EPSILON=${SDPO_SIGN_FLIP_EPSILON:-0.2}

############################ IS correction with squashing function f(w) ############################
# Explicit IS mode: the hint-row IS weight w = π(y|x)/π(y|x,e) is computed in the
# trainer and squashed by f(w) = w/(w + IS_FN_CONST), then applied as
# rollout_is_weights. f lifts tiny weights (stuck prompts → stronger rescue) and
# saturates large ones at 1. IS_FN_CONST is the half-saturation point (f=0.5 at
# w=IS_FN_CONST). Set IS_FN_CONST=0 to fall back to the strict-unbiased implicit
# mode (IS weight = PPO ratio, needs use_rollout_log_probs=True).
IS_CORRECTION=${IS_CORRECTION:-True}
IS_FN_CONST=${IS_FN_CONST:-0.1}

# Explicit f(w) mode needs a NEUTRAL PPO ratio so the coefficient is exactly f(w):
# use_rollout_log_probs=False (on-policy detach → ratio≡1). Implicit mode (fn=0)
# needs True. Auto-select based on IS_FN_CONST.
if (( $(awk -v c="$IS_FN_CONST" 'BEGIN{print (c>0)?1:0}') )); then
    USE_ROLLOUT_LP=False
else
    USE_ROLLOUT_LP=${IS_CORRECTION}
fi

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
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    # REQUIRED for IS correction: use stored old_log_probs (hinted behavior for
    # hint rows) instead of the on-policy detach, so the PPO ratio = IS weight.
    +actor_rollout_ref.actor.use_rollout_log_probs=${USE_ROLLOUT_LP}
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
    # No origin KL in reward: pure outcome rewards. _compute_ref_log_prob only
    # writes sdpo_ref_log_prob (current actor, hinted); no RefPolicy worker.
    algorithm.use_kl_in_reward=False
)

# SDPO subtree is not part of the bundled schema; prefix with `+` to add new keys.
SDPO=(
    +sdpo.modify_ref_prompt.enabled=True
    +sdpo.modify_ref_prompt.use_current_actor=True
    +sdpo.modify_ref_prompt.source=${EXPERT_SOURCE}
    +sdpo.modify_ref_prompt.dataset_key=${EXPERT_KEY}
    +sdpo.modify_ref_prompt.expert_index=${EXPERT_INDEX}

    +sdpo.hint_rollout.enabled=True
    +sdpo.hint_rollout.n_hint=${N_HINT}
    # Merge hint rows into the GRPO group (size = n + n_hint per uid):
    #   A: no hint succeeds          -> vanilla GRPO on main only (group=n)
    #   B: hint+main both succeed    -> GRPO(main) + sign-flip (if lambda>0)
    #   C: all main fail, hint OK    -> GRPO with group=n+n_hint; hint rows get
    #                                   gradient, IS-corrected (see below)
    +sdpo.hint_rollout.merge_into_group=True

    +sdpo.sdpo_grpo.sign_flip_lambda_pos=${SDPO_SIGN_FLIP_LAMBDA_POS}
    +sdpo.sdpo_grpo.sign_flip_lambda_neg=${SDPO_SIGN_FLIP_LAMBDA_NEG}
    +sdpo.sdpo_grpo.sign_flip_epsilon=${SDPO_SIGN_FLIP_EPSILON}

    # Strict-unbiased IS correction for merged hint rows (regime C).
    +sdpo.sdpo_grpo.is_correction.enabled=${IS_CORRECTION}
    # f(w) = w/(w + fn_const) squashing on the IS weight (0 => strict implicit).
    +sdpo.sdpo_grpo.is_correction.fn_const=${IS_FN_CONST}
)

# Derive EXP_NAME from knobs (must precede TRAINER, which references it).
EXP_NAME=${EXP_NAME:-qwen_deepmath_sdpo_merge_isfn${IS_FN_CONST}_n${N_HINT}_lpos${SDPO_SIGN_FLIP_LAMBDA_POS}_noklbase_${DATE_TAG}}

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
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"
