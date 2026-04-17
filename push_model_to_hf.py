import argparse
import os

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--repo_id", default="XinnanZhang/Qwen3-1.7B-Base-Openthought400K-SFT")
parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
args = parser.parse_args()

api = HfApi(token=args.token)
api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=args.model_path,
    repo_id=args.repo_id,
    repo_type="model",
)

