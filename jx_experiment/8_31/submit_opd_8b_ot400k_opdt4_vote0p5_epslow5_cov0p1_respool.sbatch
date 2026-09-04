#!/bin/bash
# OPD opd_theory_guided4 (vote=0.5, cov=0.1, eps_low=5) with the real hybrid loss
# (k1_pg_fkl_topk), teacher on its own resource pool. Logs online to wandb.
#
# Two nodes: student pool = 8 GPUs on the head node, teacher pool = 8 GPUs on the worker
# node (16 total). main_ppo's ResourcePoolManager needs Ray to see all 16, which a plain
# single-node launch does not do ("Total available GPUs 8.0 is less than total desired 16").
# So build the Ray cluster explicitly, following examples/slurm/ray_on_slurm.slurm
# (adapted: conda instead of apptainer, readarray for the node list, --gpus-per-node so
# SLURM_GPUS_PER_NODE is populated).
#SBATCH --job-name=seller
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/xinnanzh/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%j.err

set -xeu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Log straight to wandb.ai instead of syncing afterwards. Verified reachable from a compute
# node (curl https://api.wandb.ai/ -> HTTP 404, which is that endpoint's normal response).
export WANDB_MODE=online
export WANDB_ENTITY=rl_agent

# Ray caches the last cluster it joined under /tmp/ray on each node, so a node that
# previously hosted a different Ray cluster can make this job try to rejoin a dead head
# ("ConnectionError: Failed to connect to Ray cluster at <stale_ip>:6379", job 15997).
# This job starts its own head below, so clear any leftover state first.
unset RAY_ADDRESS
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray/ray_current_cluster

cd /fsx/xinnanzh/on-policy-distillation

python opd_inference/fix_llamafactory_tokenizer_config.py \
    /fsx/xinnanzh/checkpoints/sft_qwen3_8b_ot400k --also_generation_config

############################ Ray cluster across the allocation ############################
# NOTE: upstream writes nodes_array=("$nodes"), which collapses the whole multi-line list
# into one element. readarray keeps one node per element.
readarray -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then head_node_ip=${ADDR[1]}; else head_node_ip=${ADDR[0]}; fi
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head   nodes: ${nodes_array[*]}"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate opd
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    ray start --head --node-ip-address=$head_node_ip --port=$port \
        --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus ${SLURM_GPUS_PER_NODE} --block" &
sleep 20

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh && conda activate opd
        unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
        ray start --address $ip_head \
            --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus ${SLURM_GPUS_PER_NODE} --block" &
    sleep 10
done
sleep 20

# Sanity-check that Ray really sees 16 GPUs before launching; otherwise the job would fail
# ~90s later inside create_resource_pool with a much less obvious message.
RAY_ADDRESS="$ip_head" ray status || true

export RAY_ADDRESS="$ip_head"
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate opd
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    export RAY_ADDRESS=$ip_head
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export WANDB_MODE=online WANDB_ENTITY=rl_agent
    cd /fsx/xinnanzh/on-policy-distillation
    bash jx_experiment/8_31/run_8b_ot400k_8b_opdt4_vote0p5_epslow5_cov0p1_respool.sh"
