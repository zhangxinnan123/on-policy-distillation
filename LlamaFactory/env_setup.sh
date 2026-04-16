conda create -n llama_factory python=3.12
conda activate llama_factory

cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
pip install wandb
