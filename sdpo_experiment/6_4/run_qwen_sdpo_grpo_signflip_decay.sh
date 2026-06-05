#!/usr/bin/env bash
# SDPO+GRPO on deepmath, SELF-DISTILLED-KL DECAY only. No origin KL, NO merge.
#
# Stripped-down ablation:
#   - NO origin KL against base/init policy (reward = pure outcome)
#   - NO merge: hint rollouts stay OUT of the GRPO group (merge_into_group=False);
#     compute_sdpo_grpo_advantage takes the legacy path (is_hint_row=None):
#       vanilla GRPO over n main rows per uid, sign-flip applied ONLY on prompts
#       whose hint_solvable=True. No regime A/B/C, no n+n_hint expansion.
#   - Sign-flip strength decays over training (the only hint mechanism left).
#
#   reward      = outcome   (no KL term)
#   advantage   = GRPO(main) with sign-flip on hint_solvable prompts
#   self-distilled KL delta = log π_θ(y|x,e) - log π_θ(y|x)   (current actor)
#   lambda(t)   = lambda_base * mult(t)
#   mult(t)     = cosine|linear annealed 1.0 -> SIGN_FLIP_FINAL_SCALE
#                 over SIGN_FLIP_DECAY_TOTAL_STEPS (0 => total_training_steps)
#
# Hint rollouts still run (n_hint per prompt) purely to label hint_solvable.
#
# Single ref key:
#   sdpo_ref_log_prob = π_cur(y|x, e)   (current actor, hinted)  -> sign-flip
#   (ref_log_prob is NOT computed — use_kl_in_reward=False, no RefPolicy worker)
#
# Knobs:
#   SDPO_SIGN_FLIP_LAMBDA_POS/NEG/EPSILON -> sign-flip (hint) base strength
#   SIGN_FLIP_DECAY / _TYPE / _FINAL_SCALE / _TOTAL_STEPS / WARMUP -> decay schedule
#   N_HINT          -> number of hint rollouts per prompt (default 1)
#
# Requires the sdpo_grpo wiring in verl/trainer/ppo/sdpo_ray_trainer.py.
set -xeuo pipefail
source "$(dirname "$0")/../config.sh"
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

############################ Quick Config ############################

ROLLOUT_NAME="vllm"

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_sdpo}
# EXP_NAME is auto-derived AFTER knobs are read (see below); override via
# `EXP_NAME=... bash ...` if you want a custom name.
DATE_TAG=${DATE_TAG:-6_4}

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

############################ Self-distilled-KL decay schedule ############################
# Anneal sign_flip_lambda_{pos,neg} toward SIGN_FLIP_FINAL_SCALE over training:
#   lambda(t) = lambda_base * mult(t)
#   mult(t)   = cosine|linear from 1.0 -> SIGN_FLIP_FINAL_SCALE over total_steps
# Implemented in sign_flip_decay_multiplier() in sdpo_ray_trainer.py; effective
# lambdas are logged as sdpo/sign_flip/lambda_{pos,neg}_eff and decay_mult.
SIGN_FLIP_DECAY=${SIGN_FLIP_DECAY:-True}
SIGN_FLIP_DECAY_TYPE=${SIGN_FLIP_DECAY_TYPE:-linear}          # cosine | linear
SIGN_FLIP_FINAL_SCALE=${SIGN_FLIP_FINAL_SCALE:-0.0}          # 0.0 == fully decay to off
SIGN_FLIP_DECAY_TOTAL_STEPS=${SIGN_FLIP_DECAY_TOTAL_STEPS:-50} # decay fully to 0 by step 50, then stays off
SIGN_FLIP_WARMUP_STEPS=${SIGN_FLIP_WARMUP_STEPS:-0}          # hold lambda at base for first N steps

# NOTE: origin KL against the base/init policy is intentionally REMOVED in this
# script (no ORIGIN_KL_COEF / INIT_REF_MODEL / RefPolicy worker).

# Derive explicit EXP_NAME from knobs so each wandb run is self-describing.
# `nomerge` + `decay` tags distinguish from siblings; `noklbase` marks no origin KL.
if [ "${SIGN_FLIP_DECAY}" = "True" ]; then
    DECAY_TAG="_decay${SIGN_FLIP_DECAY_TYPE}2${SIGN_FLIP_FINAL_SCALE}"
else
    DECAY_TAG="_nodecay"
fi
EXP_NAME=${EXP_NAME:-qwen_deepmath_sdpo_nomerge_n${N_HINT}_lpos${SDPO_SIGN_FLIP_LAMBDA_POS}_lneg${SDPO_SIGN_FLIP_LAMBDA_NEG}${DECAY_TAG}_noklbase_${DATE_TAG}}

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
    # No KL anywhere: origin KL removed, sign-flip is the only hint mechanism.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
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
    # No origin KL in reward: pure outcome rewards into the GRPO baseline.
    # _compute_ref_log_prob therefore only writes sdpo_ref_log_prob (hinted,
    # current actor); no frozen RefPolicy second pass.
    algorithm.use_kl_in_reward=False
)

# SDPO subtree is not part of the bundled schema; prefix with `+` to add new keys.
SDPO=(
    +sdpo.modify_ref_prompt.enabled=True
    # Current actor as the SDPO ref -> sdpo_ref_log_prob = π_cur(y|x,e).
    # With use_kl_in_reward=False there is no origin-KL second pass, so
    # ref_policy_wg stays aliased to the actor and no extra worker is spawned.
    +sdpo.modify_ref_prompt.use_current_actor=True
    +sdpo.modify_ref_prompt.source=${EXPERT_SOURCE}
    +sdpo.modify_ref_prompt.dataset_key=${EXPERT_KEY}
    +sdpo.modify_ref_prompt.expert_index=${EXPERT_INDEX}

    +sdpo.hint_rollout.enabled=True
    +sdpo.hint_rollout.n_hint=${N_HINT}
    # NO merge: hint rollouts only label hint_solvable; they do NOT enter the
    # GRPO group. compute_sdpo_grpo_advantage takes the legacy path — vanilla
    # GRPO over the n main rows per uid, sign-flip applied only on prompts whose
    # hint_solvable=True. No regime A/B/C, no n+n_hint expansion.
    +sdpo.hint_rollout.merge_into_group=False

    +sdpo.sdpo_grpo.sign_flip_lambda_pos=${SDPO_SIGN_FLIP_LAMBDA_POS}
    +sdpo.sdpo_grpo.sign_flip_lambda_neg=${SDPO_SIGN_FLIP_LAMBDA_NEG}
    +sdpo.sdpo_grpo.sign_flip_epsilon=${SDPO_SIGN_FLIP_EPSILON}

    # Self-distilled-KL decay schedule for the sign-flip lambdas.
    +sdpo.sdpo_grpo.sign_flip_decay.enabled=${SIGN_FLIP_DECAY}
    +sdpo.sdpo_grpo.sign_flip_decay.decay_type=${SIGN_FLIP_DECAY_TYPE}
    +sdpo.sdpo_grpo.sign_flip_decay.final_scale=${SIGN_FLIP_FINAL_SCALE}
    +sdpo.sdpo_grpo.sign_flip_decay.total_steps=${SIGN_FLIP_DECAY_TOTAL_STEPS}
    +sdpo.sdpo_grpo.sign_flip_decay.warmup_steps=${SIGN_FLIP_WARMUP_STEPS}
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
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"
