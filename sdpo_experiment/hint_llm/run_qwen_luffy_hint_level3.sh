#!/usr/bin/env bash
# LUFFY-only baseline: p/(p+c) reshape on hint rows, NO SDPO advantage manipulation.
#
# Purpose: isolate the effect of LUFFY-style off-policy reshape from the SDPO
# advantage machinery (sign-flip on Regime B). This is the "just LUFFY" arm.
#
# Compared to run_qwen_sdpo_grpo_merge_is_hint_level3.sh:
#   sign_flip_lambda_{pos,neg}=0.0            ← Regime B does NOTHING extra
#                                               (main-row advantage = pure GRPO)
#   is_correction.fn_const=0.1                ← LUFFY p_div_p_0.1 on hint rows
#   use_rollout_log_probs=False               ← PPO ratio≡1; coefficient = f(p)
#
# Effective per-token pg_loss on hint rows:
#     -A_hint · f(p) · 1        where f(p) = π_θ(y_t|x) / (π_θ(y_t|x) + c)
# On main rows:
#     -A_main · ratio · 1       (standard PPO clip, ratio=1 when on-policy)
#
# Regime dispatch still applies (A/B/C) but with sign-flip off:
#   A: hint fails         → GRPO on main only, hint rows advantage=0 (no grad)
#   B: both succeed       → GRPO on main only (no sign-flip); hint rows adv=0
#   C: main all fail      → GRPO on main+hint group; hint rows carry grad, THEN
#                           reshaped by LUFFY f(p) on this loss row.
#
# So the only place LUFFY reshape actually flows into the gradient is Regime C
# (where hint rows have non-zero advantage). This is intentional: LUFFY only
# matters when the off-policy sample provides gradient signal.
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
# With USE_DYNAMIC_BSZ=True, ppo_micro_batch_size_per_gpu is ignored — only
# ppo_max_token_len_per_gpu matters. Baseline 40960 uses ~61/81 GB (~75%).
# NOTE: 57000 OOM'd at loss.backward() (backward activations peak +16 GB on top
# of ~62 GB forward footprint). Falling back to 49152 (+20% over baseline) to
# leave headroom for backward. If this OOMs too, drop to 45056.
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

# Sampling
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

# Regime B sign-flip: HARD-CODED OFF here. LUFFY-only baseline.
SDPO_SIGN_FLIP_LAMBDA_POS=0.0
SDPO_SIGN_FLIP_LAMBDA_NEG=0.0
SDPO_SIGN_FLIP_EPSILON=0.2

############################ LUFFY reshape ############################
# c in f(p) = p / (p + c). LUFFY paper defaults: 0.1 / 0.3 / 0.5.
# Larger c → stronger squashing (rare tokens get more amplification).
LUFFY_C=${LUFFY_C:-0.1}

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
    # LUFFY reshape requires NEUTRAL PPO ratio so coefficient = f(p), not f(p)·ratio.
    +actor_rollout_ref.actor.use_rollout_log_probs=False
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
    # Merge hint rows into batch so LUFFY reshape has hint tokens to weight.
    +sdpo.hint_rollout.merge_into_group=True

    # Sign-flip HARD-OFF: this is the LUFFY-only baseline.
    +sdpo.sdpo_grpo.sign_flip_lambda_pos=${SDPO_SIGN_FLIP_LAMBDA_POS}
    +sdpo.sdpo_grpo.sign_flip_lambda_neg=${SDPO_SIGN_FLIP_LAMBDA_NEG}
    +sdpo.sdpo_grpo.sign_flip_epsilon=${SDPO_SIGN_FLIP_EPSILON}

    # LUFFY explicit reshape (p/(p+c)) on hint rows.
    +sdpo.sdpo_grpo.is_correction.enabled=True
    +sdpo.sdpo_grpo.is_correction.fn_const=${LUFFY_C}

    # Regime C advantage scale α: dampens the |advantage| inflation from having
    # only 1 succeeded row in a group of ~9. 0.25 = quarter update magnitude.
    +sdpo.sdpo_grpo.regime_c_scale=${REGIME_C_SCALE:-0.25}
)

EXP_NAME=${EXP_NAME:-qwen_deepmath_luffyShape_c${LUFFY_C}_regCscale${REGIME_C_SCALE:-0.25}_n${N_HINT}_hint${HINT_LEVEL}_${DATE_TAG}}

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
