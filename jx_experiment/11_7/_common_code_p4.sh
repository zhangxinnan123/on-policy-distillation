#!/usr/bin/env bash
# Shared configuration for the 11_7 coding sweep. NOT runnable on its own --
# sourced by each run_*.sh arm after that arm has set the variables below.
#
# Why a shared file instead of eight self-contained scripts (the convention in
# 8_29 / 9_7 / 9_16): the whole point of this set is that the four methods are
# compared against each other, and 8_29/RESULTS.md records that the only
# effects which survived were smaller than the drift an accidental config
# difference would introduce. Eight copies of 230 lines cannot be kept
# byte-identical by hand. Here every arm is identical by construction and the
# differing variables are the arm file's entire contents.
#
# Each arm MUST set:
#   STUDENT_MODEL              HF id of the student
#   SIZE_TAG                   short tag used in EXP_NAME (e.g. 1p7b_base)
#   METHOD_TAG                 short tag used in EXP_NAME (opd|eopd|aopd|opdt4)
#   DISTILLATION_LOSS_MODE     k1_topk_overlap (opd baseline) or k1_pg_fkl_topk
#   HYBRID_MASK_STRATEGY       all|teacher_topk_entropy|aopd|opd_theory_guided4
#   MASK_KWARGS                bash array of extra hydra overrides (may be empty)

set -xeuo pipefail

: "${STUDENT_MODEL:?arm must set STUDENT_MODEL}"
: "${SIZE_TAG:?arm must set SIZE_TAG}"
: "${METHOD_TAG:?arm must set METHOD_TAG}"
: "${DISTILLATION_LOSS_MODE:?arm must set DISTILLATION_LOSS_MODE}"
: "${HYBRID_MASK_STRATEGY:?arm must set HYBRID_MASK_STRATEGY}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/jx_experiment/config.sh"

TEACHER_MODEL=Qwen/Qwen3-8B
ROLLOUT_NAME="vllm"

############################ Lengths ############################
MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=4096
# Validation uses the SAME budget as training. For math these differed (4096
# train / 8192 val) because a truncated proof scores zero; here the prompt asks
# for a single code block and the measured prompts are 162-670 tokens, so 4096
# is not the binding constraint and matching the two keeps the vLLM KV cache
# sized once.
VAL_MAX_RESPONSE_LENGTH="${VAL_MAX_RESPONSE_LENGTH:-4096}"   # TTS offline eval uses 8192
MAX_NUM_TOKENS=$(( MAX_PROMPT + VAL_MAX_RESPONSE_LENGTH + 1 ))
MAX_NUM_SEQS=128
TRAIN_PROMPT_BSZ=128

############################ Placement (p4d.24xlarge, A100 40GB) ############################
# Single node, teacher COLOCATED with the student. Copied from
# 9_16/run_4b_base_8b_eopd_thr1p0_p4_colocated.sh, which documents why: the
# 2-node split teacher died twice inside the vLLM sleep-mode allocator
# (c10::Error "Trying to free a pointer not allocated here", jobs 7638/7646)
# with 4B, 0/2, while colocated completed.
STUDENT_WORLD_SIZE=8
TEACHER_RESOURCE_POOL=False
TEACHER_WORLD_SIZE=8
TEACHER_TP=2                 # 8B bf16 is 16.4GB -> 8.2GB/GPU on a 40GB card
TEACHER_GPU_MEM_UTIL=0.6
STUDENT_GPU_MEM_UTIL=0.5

STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
# SP * budget must be >= max_seq_len (seqlen_balancing assert):
#   2 * 3072 = 6144 >= 2048 + 4096 = 6144
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( (MAX_PROMPT + MAX_RESPONSE_LENGTH) / 2 ))
USE_DYNAMIC_BSZ=True
# SP=2, not the SP=1 the colocated eopd script used. The hybrid FKL arm
# materializes and backprops through the student top-k logprobs, which the
# all-PG k1_topk_overlap baseline never did; 9_16/run_1p7b_opdt4_eps2_cov0p1_p4.sh
# records the 8_29 *_fkl runs OOMing in actor_rollout_update_actor at SP=1.
# ulysses SP is exact, so this changes memory and step time, not the loss.
# Held at 2 for the baseline arm too, so the four arms differ only in routing.
SP=2
# ppo_mini_batch_size must divide world_size/SP = 4; 128/4 = 32.

############################ Optimization ############################
ROLLOUT_N=1
LR=1e-6
train_batch_size=$(( TRAIN_PROMPT_BSZ ))
USE_FUSED_KERNELS=False
ENFORCE_EAGER=True

DISTILLATION_TOPK=16
DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0
PG_LOSS_COEF=1.0
SUPERVISED_LOSS_COEF=1.0
# Ignored under a hybrid loss mode (routing is internal); kept at the value every
# previous arm used so the baseline arm, which does read it, matches 8_9.
USE_POLICY_GRADIENT=True

############################ Schedule ############################
TOTAL_STEPS="${TOTAL_STEPS:-200}"
# Validation cost drives this. 717 eval prompts x n=8 = 5736 generations per
# validation, about 45 training steps' worth of rollout, against 128 generations
# per training step. At TEST_FREQ=50 the run validates at step 0, 50, 100, 150
# and 200 -- 5 points, ~2.1x the job's total generation cost.
#
# 5 points matters: 8_29/RESULTS.md reports "best @ step", which is not a
# meaningful selection over 3 points. This became affordable only once LCB was
# narrowed to the v6 shard (175 problems); with the full cumulative release_v5
# suite (879) the same 5 points would have cost ~4x instead.
TEST_FREQ="${TEST_FREQ:-50}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
EXP_SUFFIX="${EXP_SUFFIX:-}"

EXP_NAME="opd_code/${SIZE_TAG}_8b_p4/${METHOD_TAG}_b${train_batch_size}_n${ROLLOUT_N}_lr${LR}_reslen${MAX_RESPONSE_LENGTH}${EXP_SUFFIX}"

############################ Paths ############################
# Built by jx_experiment/11_7/prepare_code_data.py; see its MANIFEST.json for the
# exact counts, the dropped prompts, and the caveats that apply to every number
# these runs produce.
DATA_PATH="${CODE_DATA_PATH:-${HOME}/data/code_opd}"
TRAIN_FILES="['$DATA_PATH/train.parquet']"
# CODE_TEST_FILE picks a smaller eval set, e.g. test_100.parquet from
# subsample_eval.py (same prompts for every arm). Default: the full 717.
TEST_FILE="${CODE_TEST_FILE:-$DATA_PATH/test.parquet}"
TEST_FILES="['$TEST_FILE']"

CODE_REWARD_PY="${REPO_ROOT}/jx_experiment/11_7/code_reward.py"
test -f "$DATA_PATH/train.parquet"
test -f "$TEST_FILE"
test -f "$CODE_REWARD_PY"

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
    # Overlong prompts were already removed at build time so the eval
    # denominator is fixed and recorded; nothing should trip this.
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    # Non-thinking, matching every math arm in this project.
    #
    # This is only coherent because the parquet was built with
    # --prompt-mode nonthink, which STRIPS CodeR1-12k's own system prompt. That
    # prompt demands reasoning in <think>...</think> and the answer in
    # <answer>...</answer>, while enable_thinking=False makes the Qwen3 template
    # append a pre-closed "<think>\n\n</think>" to the assistant turn -- the
    # model would be instructed to fill a block that is already closed. Verified
    # by rendering both ways against the built parquet. If you rebuild the data
    # with --prompt-mode coder1, flip this to True.
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
    distillation.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP
    distillation.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEM_UTIL
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=$MAX_NUM_SEQS
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=$DISTILLATION_TOPK
    # Pure distillation: no task reward is mixed into the training loss. The
    # train parquet therefore carries no usable ground truth, and the executing
    # reward function below is used for VALIDATION ONLY.
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
    distillation.distillation_loss.hybrid_mask_strategy=$HYBRID_MASK_STRATEGY
    distillation.distillation_loss.pg_loss_coef=$PG_LOSS_COEF
    distillation.distillation_loss.supervised_loss_coef=$SUPERVISED_LOSS_COEF
    distillation.loop_metrics.enabled=False
)

# Executing reward for HumanEval+/MBPP+/LiveCodeBench validation.
#
# reward_manager is left at its default `naive`, deliberately. With
# trainer.use_legacy_worker_impl=disable (set below) rewards go through
# verl/experimental/reward_loop/, whose registry holds only
# naive/dapo/gdpo/remote/visual/rate_limited -- asking for `batch` there dies at
# startup with "Unknown reward manager: batch" (job 14359).
#
# reward_loop's `naive` is NOT the serial legacy one: run_single is async, a sync
# compute_score is dispatched via run_in_executor, and reward.num_workers=8
# RewardLoopWorker actors shard the batch, so the ~5.7k executions per
# validation run concurrently. A dict result has every key copied into
# reward_extra_info, which is what puts the metric at val-core/<ds>/acc/mean@8 --
# the key every table in 8_29/RESULTS.md and dump_wandb_results.py reads.
#
# code_reward.py caps its OWN subprocess concurrency (OPD_CODE_REWARD_CONCURRENCY
# per process x 8 actors); see the sbatch.
REWARD=(
    reward.custom_reward_function.path="$CODE_REWARD_PY"
    reward.custom_reward_function.name=compute_score
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
    actor_rollout_ref.rollout.gpu_memory_utilization=$STUDENT_GPU_MEM_UTIL
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
    trainer.save_freq=${SAVE_FREQ:--1}
    trainer.test_freq=$TEST_FREQ
    trainer.total_epochs=2
    trainer.val_before_train=$VAL_BEFORE_TRAIN
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=${RESUME_MODE:-disable}
    # Every validation generation + its score as jsonl, so a later harness fix can
    # be re-scored offline instead of re-training (the wandb table keeps only 5).
    trainer.validation_data_dir=${VAL_DUMP_DIR:-null}
    trainer.max_actor_ckpt_to_keep=1
    trainer.log_val_generations=5
    trainer.total_training_steps=$TOTAL_STEPS
)

export VLLM_USE_V1=1
export WANDB_API_KEY=$WANDB_API_KEY

echo "========== 11_7 CODE SWEEP: ${METHOD_TAG} / ${SIZE_TAG} =========="
echo "student        : $STUDENT_MODEL"
echo "teacher        : $TEACHER_MODEL  (colocated, TP=$TEACHER_TP, mem=$TEACHER_GPU_MEM_UTIL)"
echo "loss_mode      : $DISTILLATION_LOSS_MODE"
echo "mask_strategy  : $HYBRID_MASK_STRATEGY"
echo "mask_kwargs    : ${MASK_KWARGS[*]+${MASK_KWARGS[*]}}"
echo "reslen         : $MAX_RESPONSE_LENGTH   SP=$SP   budget/gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU"
echo "data           : $DATA_PATH"
echo "steps          : $TOTAL_STEPS   test_freq=$TEST_FREQ   val_before_train=$VAL_BEFORE_TRAIN"
echo "=================================================================="

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    ${MASK_KWARGS[@]+"${MASK_KWARGS[@]}"} \
    "${REWARD[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
