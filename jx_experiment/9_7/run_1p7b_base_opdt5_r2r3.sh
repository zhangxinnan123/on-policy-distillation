#!/usr/bin/env bash
set -xeuo pipefail


source "$(dirname "$0")/../config.sh"

ROLLOUT_NAME="vllm"

FAMILY="Qwen"
############################ opdt4 rule ablation -- 1P7B-Base ############################
# Three arms identical except the two rule toggles in hybrid_masks.py:1123, so the set
# isolates what each rule contributes. Config matches run zrbek8zd (vote0.3, cov0.2,
# eps_low=0.5, tp0.9), the only 8B-Base run that did NOT collapse -- 4.7% clip ratio and
# 95.3% completion, versus 100% clip / 0% completion for the other five.
#
#   R1 (coverage): student mass on the teacher top-p nucleus < coverage_threshold -> FKL
#   R2 (M3 vote):  teacher-prob-weighted vote of [pi_T(c)/pi_S(c) > ratio_low] > vote -> FKL
#   fkl_position = R1 or R2
#
# Prediction to check, not to assume: on 5.06M saved positions R1 at cov=0.2 fired on 1.07%
# of positions and added only 0.01% beyond R2. If that holds, r1only should sit at ~99% PG
# (i.e. indistinguishable from baseline) and r2only should match r1r2. The arm that makes
# this worth running is r1r2 -- it repeats zrbek8zd under an identical config, which gives
# 8B-Base its FIRST noise estimate. Every 8B-Base number so far is a single run with no
# floor to compare against.
# This arm: R1 + R2, both on -- also a repeat of zrbek8zd
################ opd_theory_guided5: adds the over-confidence rule ################
# v5 = v4's two rules plus a third, over the same teacher top-p nucleus:
#   R1  DISABLED here. The 3x3 base ablation measured it firing on ~0.3% of positions
#       (the r1only arms sat at pg 99.6-99.8%, i.e. indistinguishable from baseline),
#       and offline it added only 0.01% beyond R2 over 5.06M positions.
#   R2  vote of [pi_T(c)/pi_S(c) > ratio_low] weighted by pi_T > vote    -> FKL
#   R3  ANY c with pi_S(c)/pi_T(c) > ratio_high AND pi_S(c) > floor       -> FKL  (new)
#       The floor is what makes ANY-over-K meaningful: without it a ratio of 3 fires on
#       pi_S=0.003 vs pi_T=0.001, pure noise. At floor=0.9 the trigger is 'student is
#       >90% certain on a token the teacher gives <30%'. A floor >= 0.5 also admits at
#       most one candidate, since probabilities sum to 1.
#
# ratio_high=3.0 comes from the code default eps_high=2.0 (i.e. ratio 3.0), which the
# v1/v3 docstring justifies as deliberately conservative: undercoverage is a structural
# RKL failure and needs a sensitive trigger, overshoot is recoverable and needs a blunt
# one. The 5_11 scripts actually ran eps_high=0.5 (ratio 1.5), making both sides equally
# sensitive and contradicting that reasoning -- not copied here.
#
# R3 uses ANY over the nucleus, not a vote, so it is far more sensitive than the v1/v3
# high-side rule which tested only the sampled token. WATCH opd_overconfident_ratio in
# the first steps: if R3 alone routes >50% of positions, ratio_high is too low.
#
# NOTE on thresholds: v5 takes raw ratios. v4's eps_low=0.5 meant ratio 1.5 and
# eps_low=10 meant ratio 11; here ratio_low=10 means exactly pi_T/pi_S > 10.
STUDENT_MODEL=Qwen/Qwen3-1.7B-Base
TEACHER_MODEL=Qwen/Qwen3-8B

USE_POLICY_GRADIENT=True   # ignored under hybrid; routing handled internally
DISTILLATION_LOSS_MODE="k1_pg_fkl_topk"   # use_hybrid=True — mask actually routes
USE_FUSED_KERNELS=False

HYBRID_MASK_STRATEGY="opd_theory_guided5"
HYBRID_MASK_USE_COVERAGE_RULE=False   # R1 关闭
HYBRID_MASK_USE_LOW_COVERAGE_RULE=True  # R2
HYBRID_MASK_COVERAGE_THRESHOLD=0.2
HYBRID_MASK_FKL_VOTE_THRESHOLD=0.3
HYBRID_MASK_RATIO_LOW=10    # 欠覆盖: pi_T(c)/pi_S(c) > 10  (v5 直接取比值，非 1+eps)
HYBRID_MASK_RATIO_HIGH=3.0  # 过度自信: pi_S(c)/pi_T(c) > 3.0
HYBRID_MASK_OVERSHOOT_FLOOR=0.9  # 且 pi_S(c) > 0.9 (绝对质量下限)
HYBRID_MASK_TEACHER_TOP_P=0.9
HYBRID_MASK_PROB_FLOOR=1e-6

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
# SP=2 + gpu_mem 0.4: the 8_29 *_fkl runs OOM'd in actor_rollout_update_actor at
# SP=1/0.5 because the true-hybrid FKL arm materializes and backprops through the
# student top-k logprobs, which k1_topk_overlap never did. Matches the known-good
# 8B recipe in jx_experiment/8_25. Neither knob changes the math: ulysses SP is
# exact, and gpu_memory_utilization only sizes the vLLM KV cache.
SP=2

ROLLOUT_N=1
LR=1e-6
train_batch_size=$(( TRAIN_PROMPT_BSZ ))

EXP_NAME="opd_ablate/1p7b_base_8b/opdt5_r2r3_rlow${HYBRID_MASK_RATIO_LOW}_rhigh${HYBRID_MASK_RATIO_HIGH}_floor${HYBRID_MASK_OVERSHOOT_FLOOR}_vote${HYBRID_MASK_FKL_VOTE_THRESHOLD}_b${train_batch_size}_n${ROLLOUT_N}_lr${LR}_reslen${MAX_RESPONSE_LENGTH}"

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
    distillation.teacher_model.inference.gpu_memory_utilization=0.4
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
    +distillation.distillation_loss.hybrid_mask_kwargs.ratio_low=$HYBRID_MASK_RATIO_LOW
    +distillation.distillation_loss.hybrid_mask_kwargs.ratio_high=$HYBRID_MASK_RATIO_HIGH
    +distillation.distillation_loss.hybrid_mask_kwargs.overshoot_floor=$HYBRID_MASK_OVERSHOOT_FLOOR
    +distillation.distillation_loss.hybrid_mask_kwargs.use_overconfident_rule=True
    +distillation.distillation_loss.hybrid_mask_kwargs.teacher_top_p=$HYBRID_MASK_TEACHER_TOP_P
    +distillation.distillation_loss.hybrid_mask_kwargs.prob_floor=$HYBRID_MASK_PROB_FLOOR
    +distillation.distillation_loss.hybrid_mask_kwargs.use_coverage_rule=$HYBRID_MASK_USE_COVERAGE_RULE
    +distillation.distillation_loss.hybrid_mask_kwargs.use_low_coverage_rule=$HYBRID_MASK_USE_LOW_COVERAGE_RULE
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
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

echo "========== opd_theory_guided4  vote=$HYBRID_MASK_FKL_VOTE_THRESHOLD cov=$HYBRID_MASK_COVERAGE_THRESHOLD =========="
echo "DISTILLATION_LOSS_MODE: $DISTILLATION_LOSS_MODE  (use_hybrid=True)"
echo "HYBRID_MASK_STRATEGY: $HYBRID_MASK_STRATEGY"
echo "  R1  sum_{c in teacher top-$HYBRID_MASK_TEACHER_TOP_P} pi_S(c) < $HYBRID_MASK_COVERAGE_THRESHOLD  -> FKL"
echo "  R2  use_low_coverage_rule=$HYBRID_MASK_USE_LOW_COVERAGE_RULE  vote of [pi_T/pi_S > $HYBRID_MASK_RATIO_LOW] > $HYBRID_MASK_FKL_VOTE_THRESHOLD"
echo "  R3  ANY [pi_S/pi_T > $HYBRID_MASK_RATIO_HIGH and pi_S > $HYBRID_MASK_OVERSHOOT_FLOOR]"
echo "  else                                                     -> PG (k1)"
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
