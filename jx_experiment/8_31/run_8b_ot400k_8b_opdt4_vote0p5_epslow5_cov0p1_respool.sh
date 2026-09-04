#!/usr/bin/env bash
set -xeuo pipefail

####################### OPD-theory-guided4 — 8B ot400k SFT student #######################
# First opdt4 run on the 8B SFT student; 8B previously only had baseline / aopd / tent.
# Routing (hybrid_masks.py:1061), under k1_pg_fkl_topk:
#   R1  Σ_{c ∈ teacher top-0.9} π_S(c) < coverage_threshold          → FKL
#   R2  teacher-prob-weighted vote of [π_T(c)/π_S(c) > 1 + eps_low] > vote  → FKL
#   else                                                             → PG (k1)
#
# eps_low=5 (ratio > 6) sits between the two settings measured so far. eps_low=10 drove
# pg% to 98.5 at 4B and 99.5 at 8B-Base, i.e. it nearly switched the FKL arm off, so 5
# should keep R2 actually firing. coverage_threshold is tightened 0.2 -> 0.1 (R1 fires when
# coverage < threshold, so a lower value fires less): at 0.2 R1 already fired on only 1.07%
# of positions and added just 0.01% beyond R2, so cutting it to 0.1 isolates R2.
#
# vote=0.5 plus eps_low=5 both push toward less FKL, so expect a high pg%. Watch two things
# in the first 20 steps: `pg_token_ratio` (if it pins at ~1.0 the mask is inert and this is
# just the baseline) and `response_length/mean` (8B-Base vote0.5 saturated its cap at 4033
# and collapsed entropy to 0.161 -- the 16384 cap here gives more headroom, but the failure
# mode is the same).

source "$(dirname "$0")/../config.sh"

ROLLOUT_NAME="vllm"

FAMILY="Qwen"
STUDENT_MODEL=/fsx/xinnanzh/checkpoints/sft_qwen3_8b_ot400k
TEACHER_MODEL=Qwen/Qwen3-8B

USE_POLICY_GRADIENT=True   # ignored under hybrid; routing handled internally
DISTILLATION_LOSS_MODE="k1_pg_fkl_topk"   #真 hybrid: PG=k1@sampled + supervised=FKL(top-k)
USE_FUSED_KERNELS=False

HYBRID_MASK_STRATEGY="opd_theory_guided4"
HYBRID_MASK_COVERAGE_THRESHOLD=0.1
HYBRID_MASK_FKL_VOTE_THRESHOLD=0.5
HYBRID_MASK_EPS_LOW=5   # R2 门槛 pi_T/pi_S > 6
HYBRID_MASK_TEACHER_TOP_P=0.9
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
# With SP=2 the assertion (seqlen_balancing.py:384) checks
#   ppo_max_token_len_per_gpu * SP >= max_seq_len  ->  9216*2 = 18432 >= 18432 OK
# while each rank only materializes half the sequence, so the (tokens, vocab) bf16
# logits peak drops from ~15.4 GiB to ~7.7 GiB.
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( (MAX_PROMPT + MAX_RESPONSE_LENGTH) / 2 ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=8   # student 独占 node0 全部 8 卡: 每卡负载与已验证的共置配置相同
TEACHER_RESOURCE_POOL=True   # teacher 独占资源池: main_ppo.py:249 建 teacher_pool=[8]*1
TEACHER_WORLD_SIZE=8   # teacher 独占 node1 全部 8 卡
SP=2   # ulysses: halves per-rank logits; max_token_len = ppo_max_token_len_per_gpu * SP

ROLLOUT_N=1
LR=1e-6
train_batch_size=$(( TRAIN_PROMPT_BSZ ))

EXP_NAME="opd/8b_ot400k_teacher_8b/fklhybrid_respool_opdt4_vote${HYBRID_MASK_FKL_VOTE_THRESHOLD}_cov${HYBRID_MASK_COVERAGE_THRESHOLD}_epslow${HYBRID_MASK_EPS_LOW}_b${train_batch_size}_n${ROLLOUT_N}_lr${LR}_reslen${MAX_RESPONSE_LENGTH}"

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
    distillation.teacher_model.inference.gpu_memory_utilization=0.5
    # Teacher gets a dedicated Ray pool, but teacher_model.py:96 still inits it via
    # init_colocated() -> RolloutMode.COLOCATED, so wake_up()/sleep() really execute
    # (teacher_model.py:132 compute_logprobs wraps every call in wake_up/sleep).
    # Turning enable_sleep_mode off means vLLM never registers its
    # CUDAPluggableAllocator, which is what crashed the teacher engine with
    # "Trying to free a pointer not allocated here" (raw_delete).
    # Do NOT also set free_cache_engine=False: sleep() is guarded by it
    # (vllm_async_server.py:569) but wake_up() is NOT (:554), so disabling it leaves
    # wake without a matching sleep -- that asymmetry was the previous failure.
    +distillation.teacher_model.inference.enable_sleep_mode=False
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
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

echo "===== 8B ot400k SFT + opdt4 vote=$HYBRID_MASK_FKL_VOTE_THRESHOLD cov=$HYBRID_MASK_COVERAGE_THRESHOLD eps_low=$HYBRID_MASK_EPS_LOW ====="
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
