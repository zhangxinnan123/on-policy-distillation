#!/bin/bash
#SBATCH --job-name=install_fa
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/fsx/xinnanzh/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/logs/%x_%j.err

set -xeu   # no pipefail — nvidia-smi | head sends SIGPIPE and trips the shell

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_factory

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TMPDIR=/fsx/xinnanzh/.tmp
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

nvcc --version
nvidia-smi | head -20

# H100 = SM 9.0
export TORCH_CUDA_ARCH_LIST="9.0"

# Build parallelism (H100 nodes have many CPUs)
export MAX_JOBS=32

pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -30

echo "---verify---"
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"
