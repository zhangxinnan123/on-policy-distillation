#!/usr/bin/env bash
set -xeuo pipefail

############################ OPD-theory-guided4 vote=0.3 — 1.7B 400k_1ep SFT student ############################
# Routing:
#   A = log π_T(u) − log π_S(u)  (advantage)
#   A >= 0  → PG (k1)         student under-confident on sampled token
#   A <  0  → FKL (sup)       student over-confident — steer via teacher top-k

source "$(dirname "$0")/../config.sh"

ROLLOUT_NAME="vllm"

FAMILY="Qwen"
STUDENT_MODEL=/fsx/xinnanzh/checkpoints/sft_qwen3_8b_ot400k
TEACHER_MODEL=Qwen/Qwen3-8B

USE_POLICY_GRADIENT=True   # ignored under hybrid; routing handled internally
DISTILLATION_LOSS_MODE="k1_pg_fkl_topk"   #真 hybrid: PG=k1@sampled + supervised=FKL(top-k)
USE_FUSED_KERNELS=False

HYBRID_MASK_STRATEGY="opd_theory_guided4"
# eps_low=0.5 is the point of this run. Measured FKL share by eps_low, at vote0.3/tp0.95:
#   eps_low=10 -> pg 95-99% (1-5% FKL)   eps_low=5 -> pg 93-98% (2-7% FKL)
# Both are near-inert and every run in that regime landed on its baseline. aopd, the only
# setting that ever cleared the noise floor, runs at pg 67-76% (24-33% FKL). eps_low=0.5 was
# measured at pg 69.5% on 1.7B (run xglsupnb, killed by OOM at step 32 before finishing), so
# it is the one opdt4 setting known to reach aopd's routing regime.
# Best-observed setting from the 8B-Base vote sweep (avg4 30.08 vs 28.72 / 28.62).
# Caveat: that spread (1.46pp) is inside the 1.64pp noise floor and two of the three
# runs saturated the 4096 cap, so vote=0.3 is best-observed, not proven best.
HYBRID_MASK_COVERAGE_THRESHOLD=0.1
HYBRID_MASK_FKL_VOTE_THRESHOLD=0.3
HYBRID_MASK_EPS_LOW=0.5   # R2 门槛 pi_T/pi_S > 1.5 (代码默认值)
HYBRID_MASK_TEACHER_TOP_P=0.95
HYBRID_MASK_PROB_FLOOR=1e-6

PG_LOSS_COEF=1.0
SUPERVISED_LOSS_COEF=1.0

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=16384
VAL_MAX_RESPONSE_LENGTH=16384
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=128
TRAIN_PROMPT_BSZ=128

STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
# Must stay >= the longest single sequence: seqlen_balancing.py:384 asserts
# max_token_len >= max_seq_len, and a single sequence is never split across
# micro-batches. So this is already the floor (1 sequence per micro-batch).
# Halved to match SP=2. Every earlier single-node script left this at the full 18432 while
# still setting SP=2, which throws away the point of Ulysses: each rank packs up to 18432
# tokens of activations instead of 9216. That thin margin is what killed job 16035 (4B) in
# loss.backward() -- 12.32 GiB needed, 11.98 GiB free -- and job xglsupnb (1.7B) at step 32.
# seqlen_balancing.py:384 still holds: 9216 * SP(2) = 18432 >= max_seq_len 18432.
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( (MAX_PROMPT + MAX_RESPONSE_LENGTH) / 2 ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=8
TEACHER_RESOURCE_POOL=False
TEACHER_WORLD_SIZE=8
# SP=2: jobs 15741-15744 OOM'd at step 0 in the FKL top-k backward, short by
# only 0.02-0.38 GiB (14.12 GiB requested vs 13.74 free). At reslen 16384 the
# activation is 4x what the 8_30 runs carry, and SP=2 halves it. Ulysses SP is
# mathematically exact, so results stay comparable to the SP=1 baselines.
SP=2

ROLLOUT_N=1
LR=1e-6
train_batch_size=$(( TRAIN_PROMPT_BSZ ))

EXP_NAME="opd/8b_ot400k_teacher_8b/fklhybrid_opdt4_vote${HYBRID_MASK_FKL_VOTE_THRESHOLD}_cov${HYBRID_MASK_COVERAGE_THRESHOLD}_epslow${HYBRID_MASK_EPS_LOW}_tp${HYBRID_MASK_TEACHER_TOP_P}_b${train_batch_size}_n${ROLLOUT_N}_lr${LR}_reslen${MAX_RESPONSE_LENGTH}"

ENFORCE_EAGER=True

DATA_PATH="${HOME}/data/dapo_17k_aime2426-suffix"
DAPO_TRAIN_PATH=$DATA_PATH/train.parquet
DAPO_TEST_PATH=$DATA_PATH/test.parquet
TRAIN_FILES="['$DAPO_TRAIN_PATH']"
TEST_FILES="['$DAPO_TEST_PATH']"

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.95
val_top_k=20

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    +data.apply_chat_template_kwargs.enable_thinking=True
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
    distillation.teacher_model.inference.gpu_memory_utilization=0.40
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
    +distillation.distillation_loss.hybrid_mask_kwargs.coverage_threshold=$HYBRID_MASK_COVERAGE_THRESHOLD
    +distillation.distillation_loss.hybrid_mask_kwargs.fkl_vote_threshold=$HYBRID_MASK_FKL_VOTE_THRESHOLD
    +distillation.distillation_loss.hybrid_mask_kwargs.eps_low=$HYBRID_MASK_EPS_LOW
    +distillation.distillation_loss.hybrid_mask_kwargs.teacher_top_p=$HYBRID_MASK_TEACHER_TOP_P
    +distillation.distillation_loss.hybrid_mask_kwargs.prob_floor=$HYBRID_MASK_PROB_FLOOR
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}
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
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=1
    trainer.log_val_generations=5
    trainer.total_training_steps=100
)

export VLLM_USE_V1=1
export WANDB_API_KEY=$WANDB_API_KEY

echo "===== 8B SFT + opdt4 vote=$HYBRID_MASK_FKL_VOTE_THRESHOLD cov=$HYBRID_MASK_COVERAGE_THRESHOLD eps_low=$HYBRID_MASK_EPS_LOW top_p=$HYBRID_MASK_TEACHER_TOP_P ====="
echo "STUDENT: $STUDENT_MODEL"
echo "LOSS: $DISTILLATION_LOSS_MODE + mask=$HYBRID_MASK_STRATEGY"
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
