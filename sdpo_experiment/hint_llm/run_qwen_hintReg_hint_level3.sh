#!/usr/bin/env bash
# LUFFY-style auxiliary loss on Regime C hint tokens.
#
# On top of the standard SDPO+GRPO+PPO pipeline, adds an ADDITIVE loss term:
#     ratio = π_θ(y|x) / π_θ_start(y|x, e)      (IS ratio, gradient flows)
#     ratio_capped = min(ratio, 1 + clip_eps)
#     objective = ratio_capped / (ratio_capped + γ)
#     loss_hint_reg = -mean(objective over hint_reg_mask)
#     policy_loss = pg_loss + coef · loss_hint_reg
# where hint_reg_mask = is_hint_row & (advantage > 0), i.e. Regime C hint tokens.
#
# Difference from mode="shaping": here the LUFFY shape multiplies onto ratio
# WITH gradient flowing through the shape. Standard `shaping` mode has the
# shape as a detached scalar, weaker gradient magnitude on rare tokens.
#
# Sign-flip OFF, main rows do standard PPO with fixed Regime C.
set -xeuo pipefail
source "$(dirname "$0")/../config.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

############################ Quick Config ############################
ROLLOUT_NAME="vllm"

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_sdpo}
DATE_TAG=${DATE_TAG:-6_4}

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=8192
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=256

TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=4
STUDENT_MAX_TOKEN_LEN_PER_GPU=${STUDENT_MAX_TOKEN_LEN_PER_GPU:-45056}
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=8

SP=1
ENFORCE_EAGER=True

############################ Paths ############################
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-${HOME}/data/deepmath_diff6to8}
TEST_DATA_DIR=${TEST_DATA_DIR:-${HOME}/data/dapo_17k_aime2426-suffix}
TRAIN_PARQUET=${TRAIN_PARQUET:-${TRAIN_DATA_DIR}/train_with_hints.parquet}
TEST_PARQUET=${TEST_PARQUET:-${TEST_DATA_DIR}/test.parquet}

TRAIN_FILES="['${TRAIN_PARQUET}']"
TEST_FILES="['${TEST_PARQUET}']"

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
GRPO_N=${GRPO_N:-8}
N_HINT=${N_HINT:-1}

############################ SDPO knobs ############################
EXPERT_SOURCE=${EXPERT_SOURCE:-dataset}
HINT_LEVEL=${HINT_LEVEL:-level_3}
EXPERT_KEY=${EXPERT_KEY:-${HINT_LEVEL}}
EXPERT_INDEX=${EXPERT_INDEX:-0}

SDPO_SIGN_FLIP_LAMBDA_POS=${SDPO_SIGN_FLIP_LAMBDA_POS:-0.0}
SDPO_SIGN_FLIP_LAMBDA_NEG=${SDPO_SIGN_FLIP_LAMBDA_NEG:-0.0}
SDPO_SIGN_FLIP_EPSILON=${SDPO_SIGN_FLIP_EPSILON:-0.2}

############################ PPO clip (DAPO asymmetric) ############################
PPO_CLIP_LOW=${PPO_CLIP_LOW:-0.2}
PPO_CLIP_HIGH=${PPO_CLIP_HIGH:-0.28}

############################ hint_reg auxiliary loss ############################
HINT_REG_GAMMA=${HINT_REG_GAMMA:-0.1}
HINT_REG_CLIP_EPS=${HINT_REG_CLIP_EPS:-0.2}
HINT_REG_COEF=${HINT_REG_COEF:-1.0}

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
    # Need stored hinted old_log_probs (dp_actor reads for the ratio).
    +actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.clip_ratio_low=${PPO_CLIP_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${PPO_CLIP_HIGH}
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
    algorithm.use_kl_in_reward=False
)

SDPO=(
    +sdpo.modify_ref_prompt.enabled=True
    +sdpo.modify_ref_prompt.use_current_actor=True
    +sdpo.modify_ref_prompt.source=${EXPERT_SOURCE}
    +sdpo.modify_ref_prompt.dataset_key=${EXPERT_KEY}
    +sdpo.modify_ref_prompt.expert_index=${EXPERT_INDEX}

    +sdpo.hint_rollout.enabled=True
    +sdpo.hint_rollout.n_hint=${N_HINT}
    +sdpo.hint_rollout.merge_into_group=True
    +sdpo.sdpo_grpo.always_merge_hint_into_group=False

    +sdpo.sdpo_grpo.sign_flip_lambda_pos=${SDPO_SIGN_FLIP_LAMBDA_POS}
    +sdpo.sdpo_grpo.sign_flip_lambda_neg=${SDPO_SIGN_FLIP_LAMBDA_NEG}
    +sdpo.sdpo_grpo.sign_flip_epsilon=${SDPO_SIGN_FLIP_EPSILON}

    # Implicit IS path: PPO ratio auto-becomes IS ratio; PPO min-clip applies.
    # hint_reg is added on TOP of this by dp_actor (auxiliary loss).
    +sdpo.sdpo_grpo.is_correction.enabled=True
    +sdpo.sdpo_grpo.is_correction.fn_const=0.0

    # Regime C: fixed A_hint=+1, A_main=0 (no punishment).
    +sdpo.sdpo_grpo.regime_c_scale=${REGIME_C_SCALE:-1.0}
    # IMPORTANT: hint_reg auxiliary loss handles the hint gradient signal.
    # Zero out hint advantage in the standard pg_loss to avoid double-counting.
    +sdpo.sdpo_grpo.regime_c_zero_hint_adv=True

    # hint_reg auxiliary loss (dp_actor reads via batch.meta_info).
    +sdpo.hint_reg.enabled=True
    +sdpo.hint_reg.gamma=${HINT_REG_GAMMA}
    +sdpo.hint_reg.clip_eps=${HINT_REG_CLIP_EPS}
    +sdpo.hint_reg.coef=${HINT_REG_COEF}
)

EXP_NAME=${EXP_NAME:-qwen_deepmath_hintReg_gamma${HINT_REG_GAMMA}_coef${HINT_REG_COEF}_ppoMinClip_epsLo${PPO_CLIP_LOW}_epsHi${PPO_CLIP_HIGH}_fixedRegC${REGIME_C_SCALE:-1.0}_n${N_HINT}_hint${HINT_LEVEL}_${DATE_TAG}}

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=40
    trainer.total_epochs=5
    trainer.total_training_steps=200
    trainer.val_before_train=False
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
