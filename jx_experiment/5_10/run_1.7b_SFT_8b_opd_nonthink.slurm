#!/bin/bash
#SBATCH --job-name=opd_1.7bSFT_8b_nonthink_0510
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --output='/home/jasonjx/on-policy-distillation/logs/slurm_outputs/%x_%j.out'
#SBATCH --account=mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest

set -xeuo pipefail

############################ Quick Config ############################

# --- inlined config.sh ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

WANDB_API_KEY="c4b67c713ad88ef65b62908bcaa8b5c5cb72d1a9"
export WANDB_ENTITY="${WANDB_ENTITY:-rl_agent}"
PROJECT_NAME="verl_opd_dapo"

ROLLOUT_NAME="vllm" # sglang or vllm

FAMILY="Qwen"
STUDENT_MODEL=lllyx/Qwen3-1.7B-SFT
TEACHER_MODEL=Qwen/Qwen3-8B

# Hybrid per-token partition of k1 PG and supervised forward_kl_topk.
# The outer use_policy_gradient switch is ignored for hybrid modes — the registered
# loss routes tokens between the two arms internally via hybrid_mask_strategy.
USE_POLICY_GRADIENT=True
# Loss combine: RKL (k1 PG) over ALL response tokens + FKL-topk supervised
# restricted to the mask-selected subset. The opd_theory_guided mask still
# decides which tokens additionally get FKL.
#   total = pg_loss_coef * RKL(all)  +  supervised_loss_coef * FKL(sup_mask)
# This mode also routes through compute_forward_kl_topk, so student_topk_probs /
# student_s2 (required by opd_theory_guided) are populated.
DISTILLATION_LOSS_MODE="k1_topk_overlap"
USE_FUSED_KERNELS=False

# Theory-guided OPD masking strategy.
# See verl/trainer/distillation/hybrid_masks.py:_mask_opd_theory_guided for the
# full routing rules (low-coverage, negative-k1, positive-k1).
HYBRID_MASK_STRATEGY="opd_theory_guided"
HYBRID_MASK_STUDENT_LOW_THRESHOLD=5e-2   # Rule 1: pi_c < this  → FKL
HYBRID_MASK_TEACHER_TOP_P=0.95            # nucleus mass for teacher candidates c
HYBRID_MASK_ADV_EPS=0.0                  # dead-zone around k1=0
HYBRID_MASK_USE_LOW_COVERAGE_RULE=True
HYBRID_MASK_USE_NEGATIVE_RULE=True
HYBRID_MASK_USE_POSITIVE_RULE=True

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

STUDENT_WORLD_SIZE=4

TEACHER_RESOURCE_POOL=False
TEACHER_WORLD_SIZE=4

SP=1

ROLLOUT_N=1
LR=1e-6
train_batch_size=$(( TRAIN_PROMPT_BSZ ))

EXP_NAME="fsdp/student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}/mask-opdt_p${HYBRID_MASK_TEACHER_TOP_P}_eps${HYBRID_MASK_ADV_EPS}_pg${PG_LOSS_COEF}_sup${SUPERVISED_LOSS_COEF}_b${train_batch_size}_n${ROLLOUT_N}_lr${LR}_reslen${MAX_RESPONSE_LENGTH}"

ENFORCE_EAGER=True # true for faster debugging

############################ Paths ############################
DATA_PATH="/datasets/jasonjx/data"

DAPO_TRAIN_PATH=$DATA_PATH/train.parquet
DAPO_TEST_PATH=$DATA_PATH/test.parquet

TRAIN_FILES="['$DAPO_TRAIN_PATH']"
TEST_FILES="['$DAPO_TEST_PATH']"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
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
    distillation.teacher_model.inference.gpu_memory_utilization=0.5
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
    +distillation.distillation_loss.hybrid_mask_kwargs.student_low_threshold=$HYBRID_MASK_STUDENT_LOW_THRESHOLD
    +distillation.distillation_loss.hybrid_mask_kwargs.teacher_top_p=$HYBRID_MASK_TEACHER_TOP_P
    +distillation.distillation_loss.hybrid_mask_kwargs.adv_eps=$HYBRID_MASK_ADV_EPS
    +distillation.distillation_loss.hybrid_mask_kwargs.use_low_coverage_rule=$HYBRID_MASK_USE_LOW_COVERAGE_RULE
    +distillation.distillation_loss.hybrid_mask_kwargs.use_negative_rule=$HYBRID_MASK_USE_NEGATIVE_RULE
    +distillation.distillation_loss.hybrid_mask_kwargs.use_positive_rule=$HYBRID_MASK_USE_POSITIVE_RULE
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
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
    trainer.test_freq=40
    trainer.total_epochs=2
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=1
    trainer.log_val_generations=5
    trainer.total_training_steps=240
)

############################ Launch ############################

echo "=========================================="
echo "On-Policy Distillation Job (5_10 Qwen3-1.7B-SFT k1_topk_overlap NonThink)"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Total GPUs: 4 (1 node x 4 GPUs)"
echo "=========================================="

cd /home/jasonjx/on-policy-distillation

export VLLM_USE_V1=1
export WANDB_API_KEY=$WANDB_API_KEY

echo "================== Theory-Guided OPD Mask Configuration =================="
echo "DISTILLATION_LOSS_MODE: $DISTILLATION_LOSS_MODE"
echo "HYBRID_MASK_STRATEGY: $HYBRID_MASK_STRATEGY"
echo "HYBRID_MASK_STUDENT_LOW_THRESHOLD: $HYBRID_MASK_STUDENT_LOW_THRESHOLD"
echo "HYBRID_MASK_TEACHER_TOP_P: $HYBRID_MASK_TEACHER_TOP_P"
echo "HYBRID_MASK_ADV_EPS: $HYBRID_MASK_ADV_EPS"
echo "HYBRID_MASK_USE_LOW_COVERAGE_RULE: $HYBRID_MASK_USE_LOW_COVERAGE_RULE"
echo "HYBRID_MASK_USE_NEGATIVE_RULE: $HYBRID_MASK_USE_NEGATIVE_RULE"
echo "HYBRID_MASK_USE_POSITIVE_RULE: $HYBRID_MASK_USE_POSITIVE_RULE"
echo "PG_LOSS_COEF: $PG_LOSS_COEF"
echo "SUPERVISED_LOSS_COEF: $SUPERVISED_LOSS_COEF"
echo "NON-PARTIAL: MAX_PROMPT=$MAX_PROMPT, DATA_PATH=$DATA_PATH"
echo "================================================================================"

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
