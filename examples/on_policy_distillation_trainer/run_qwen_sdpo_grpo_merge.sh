#!/usr/bin/env bash
# SDPO+GRPO on deepmath, MERGED-ROLLOUT variant (current actor as ref).
#
# Same per-uid regime selection as run_qwen_sdpo_grpo_frozen_teacher.sh
# (see compute_sdpo_grpo_advantage for A/B/C dispatch), but the SDPO
# sign-flip uses the CURRENT actor with the expert hint appended:
#     ref_log_prob = log pi_theta(y|x, e)   (live actor weights, not frozen)
#     delta        = ref_log_prob - log pi_theta(y|x)
#
# Optional origin-KL regularizer (KL to a frozen INITIAL student) is
# layered on top: when SDPO_KL_COEF>0, _compute_ref_log_prob does a
# second pass through Role.RefPolicy (loaded from ref.model.path) on the
# ORIGINAL un-hinted batch, stashes it as init_ref_log_prob, and the
# actor's use_kl_loss path prefers that key over the SDPO hinted ref.
# So you get TWO separate refs in the same step:
#     ref_log_prob       = pi_theta(y|x, e)        (current actor, hinted)  -> SDPO sign-flip
#     init_ref_log_prob  = pi_init  (y|x)          (frozen, un-hinted)      -> origin KL
#
# Requires the sdpo_grpo wiring in verl/trainer/ppo/sdpo_ray_trainer.py.
set -xeuo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

############################ Quick Config ############################

ROLLOUT_NAME="vllm"

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_sdpo}
EXP_NAME=${EXP_NAME:-qwen_deepmath_sdpo_grpo_merge}

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=8192
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=256

TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=4
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
N_HINT=${N_HINT:-1}

############################ SDPO knobs ############################
EXPERT_SOURCE=${EXPERT_SOURCE:-dataset}
EXPERT_KEY=${EXPERT_KEY:-generations_wo_think}
EXPERT_INDEX=${EXPERT_INDEX:-0}

# Sign-flip token reweighting (compute_sdpo_grpo_advantage), regime B only:
#   delta    = log pi_theta(y|x,e) - log pi_theta(y|x)   (current actor)
#   w_t      = exp(sign(A) * delta)                       # per-token
#   lambda_t = lambda_pos if A > 0 else lambda_neg
#   scale    = (1 - lambda_t) + lambda_t * clip(w_t, 1-eps, 1+eps)
#   advantage = grpo_adv * scale * response_mask
# Current-actor delta tends to be smaller than frozen-teacher delta because
# both probs come from the same network. Default lambda_pos=0.5 (asymmetric
# default since the delta is less aggressive here).
# Defaults are 0.0 == OFF, so the first run exercises only the merge_into_group
# regime dispatch (regime B falls back to vanilla GRPO on main rows).
# Turn KL on later via env: e.g. SDPO_SIGN_FLIP_LAMBDA_POS=0.5 bash ... .
SDPO_SIGN_FLIP_LAMBDA_POS=${SDPO_SIGN_FLIP_LAMBDA_POS:-0.0}
SDPO_SIGN_FLIP_LAMBDA_NEG=${SDPO_SIGN_FLIP_LAMBDA_NEG:-0.0}
SDPO_SIGN_FLIP_EPSILON=${SDPO_SIGN_FLIP_EPSILON:-0.2}

############################ Origin-KL knobs ############################
# Origin KL = standard verl actor.use_kl_loss against a FROZEN initial-student
# checkpoint, layered on top of the SDPO sign-flip. The two refs are kept in
# different batch keys so they don't collide:
#   sdpo_ref_log_prob = pi_theta(y|x, e)  (current actor, hinted)  -> SDPO sign-flip
#   ref_log_prob      = pi_init  (y|x)    (frozen, un-hinted)      -> origin KL
# Set ORIGIN_KL_COEF=0 (default) to skip the second ref forward entirely
# (use_kl_loss stays False, no RefPolicy worker is spawned). Set >0 to enable.
ORIGIN_KL_COEF=${ORIGIN_KL_COEF:-0.0}
ORIGIN_KL_TYPE=${ORIGIN_KL_TYPE:-low_var_kl}
# Frozen anchor for the origin KL. Defaults to STUDENT_MODEL so this stays
# self-contained (KL is measured against the initial student weights).
INIT_REF_MODEL=${INIT_REF_MODEL:-${STUDENT_MODEL}}

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
    actor_rollout_ref.actor.entropy_coeff=0
)

# Origin-KL toggle. When ORIGIN_KL_COEF>0, enables actor.use_kl_loss against
# the frozen init student (Role.RefPolicy loaded from ref.model.path). The
# SDPO sign-flip ref lives in sdpo_ref_log_prob and is unaffected.
if (( $(awk -v c="$ORIGIN_KL_COEF" 'BEGIN{print (c>0)?1:0}') )); then
    ORIGIN_KL=(
        actor_rollout_ref.actor.use_kl_loss=True
        actor_rollout_ref.actor.kl_loss_coef=${ORIGIN_KL_COEF}
        actor_rollout_ref.actor.kl_loss_type=${ORIGIN_KL_TYPE}
        +actor_rollout_ref.ref.model.path="${INIT_REF_MODEL}"
    )
else
    ORIGIN_KL=(
        actor_rollout_ref.actor.use_kl_loss=False
    )
fi

# Current-actor ref: no separate Role.RefPolicy worker is spun up; the trainer
# aliases ref_policy_wg to actor_rollout_wg. The REF block below is kept for
# dynamic batching settings that flow through _compute_ref_log_prob.
REF=(
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
    # CURRENT actor: ref_log_prob = log pi_theta(y|x, e) — actor re-evaluated
    # on the hinted prompt. No separate Role.RefPolicy worker is created.
    +sdpo.modify_ref_prompt.use_current_actor=True
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
    "${ORIGIN_KL[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"
