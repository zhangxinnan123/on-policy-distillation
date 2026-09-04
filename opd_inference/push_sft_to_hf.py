"""Push an SFT checkpoint to the Hugging Face Hub with a generated model card.

Uploads only inference-relevant files. The `checkpoint-*/` subdirectories hold optimizer and
scheduler state (they are why the local dirs are 113 GB / 226 GB rather than 8 GB / 16 GB) and
are skipped, as are the `.bak` files left by the rope/tokenizer repair and `training_args.bin`.

Usage:
    python opd_inference/push_sft_to_hf.py \
        --ckpt /fsx/xinnanzh/checkpoints/sft_qwen3_4b_ot400k \
        --repo XinnanZhang/Qwen3-4B-openthoughts3-math-400k-sft \
        --base Qwen/Qwen3-4B [--private] [--dry_run]
"""

import argparse
import json
from pathlib import Path

ALLOW = [
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "all_results.json",
    "train_results.json",
    "trainer_state.json",
    "trainer_log.jsonl",
    "training_loss.png",
]
ALLOW_GLOBS = ["*.safetensors", "*.safetensors.index.json"]

# mean@8 / pass@8 on ~/data/dapo_17k_aime2426-suffix, n=8, 16k max response, T=0.6/top_p=0.95/
# top_k=20, thinking mode. pass@8 is verl's val-core/<ds>/acc/best@8/mean.
EVALS = {
    "Qwen/Qwen3-4B": {
        "aime24": (42.08, 68.37), "aime25": (34.17, 54.87),
        "aime26": (37.08, 56.09), "amc23": (74.69, 90.27), "avg": (47.00, 67.40),
    },
    "Qwen/Qwen3-8B": {
        "aime24": (46.25, 70.60), "aime25": (37.08, 53.34),
        "aime26": (46.25, 62.62), "amc23": (82.19, 93.61), "avg": (52.94, 70.04),
    },
}
REF_8B = {"aime24": (67.50, 78.23), "aime25": (53.33, 73.68),
          "aime26": (55.83, 71.37), "amc23": (91.56, 99.77), "avg": (67.06, 80.76)}


def card(repo: str, base: str, ckpt: Path) -> str:
    ev = EVALS[base]
    rows = "\n".join(
        f"| {k} | {v[0]:.2f} | {v[1]:.2f} |" for k, v in ev.items() if k != "avg"
    )
    ref = ""
    if base == "Qwen/Qwen3-8B":
        ref = (
            "\nFor reference, the **untuned** `Qwen/Qwen3-8B` scores "
            f"**{REF_8B['avg'][0]:.2f} / {REF_8B['avg'][1]:.2f}** on the same eval, i.e. this "
            f"SFT model is **{REF_8B['avg'][0] - ev['avg'][0]:.1f}pp worse** than the base "
            "model it was fine-tuned from.\n"
        )

    return f"""---
license: apache-2.0
base_model: {base}
datasets:
- XinnanZhang/openthoughts3-math-50k8
language:
- en
pipeline_tag: text-generation
tags:
- math
- reasoning
- sft
---

# {repo.split("/")[-1]}

`{base}` supervised-fine-tuned on [XinnanZhang/openthoughts3-math-50k8](
https://huggingface.co/datasets/XinnanZhang/openthoughts3-math-50k8) (50k prompts x 8 samples =
400k examples), used as the student initialization for on-policy distillation experiments.

## Read this before using the model

**This checkpoint scores *below* its own base model.** It is published for reproducibility of
the distillation experiments that start from it, not as an improved model.
{ref}
The cause is in the training data, not the recipe: only **29.4%** of the SFT targets actually
close `</think>` and emit a `\\boxed{{}}` answer. The remaining 70.6% hit a ~16k generation cap
mid-reasoning, after which the chat template still appends `<|im_end|>` — so the model is
trained to emit EOS in the middle of a thought. This is inherited from upstream
(OpenThoughts3-1.2M math is ~30% complete under the same measurement). Prompt formatting and
the boxed-answer suffix were verified byte-exact and are *not* the cause.

## Training

Official `openthinker3` hyperparameters, via LLaMA-Factory:

| | |
|---|---|
| learning rate | 8e-5 |
| global batch size | 512 |
| epochs | 1 |
| cutoff length | 20000 |
| packing | neat_packing (requires per-device batch size 1; scale with gradient accumulation) |
| other | liger kernel, FlashAttention-2, bf16 |

## Evaluation

mean@8 / pass@8 in percent, n=8 samples per prompt, 16384 max response tokens,
T=0.6 / top_p=0.95 / top_k=20, thinking mode. aime24/25/26 are 30 questions each, amc23 is 40.
pass@8 is verl's `val-core/<ds>/acc/best@8/mean` estimator.

| dataset | mean@8 | pass@8 |
|---|---|---|
{rows}
| **average** | **{ev['avg'][0]:.2f}** | **{ev['avg'][1]:.2f}** |

## A packaging note that matters

This checkpoint was saved by `transformers` 5.2.0, which nests `rope_theta` under
`rope_parameters`. Older versions (e.g. 4.55.2) do not know that field and silently fall back
to `rope_theta=10000` instead of `1000000`; generations then start coherent and collapse into
character garbage with **no error raised**. On this model that difference was aime24 mean@8
**1.25 vs 42.08**.

The `config.json` here carries `rope_theta` **both** flat and nested with the same value
`1000000`, so it loads correctly under either version. If you re-save it, verify the flat key
survives.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", required=True, choices=sorted(EVALS))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    files = [p for p in sorted(ckpt.iterdir()) if p.is_file() and p.name in ALLOW]
    for g in ALLOW_GLOBS:
        files += sorted(p for p in ckpt.glob(g) if p.is_file())
    files = sorted(set(files))

    total = sum(p.stat().st_size for p in files)
    print(f"repo   : {args.repo}  ({'private' if args.private else 'PUBLIC'})")
    print(f"ckpt   : {ckpt}")
    print(f"files  : {len(files)}, {total / 1e9:.2f} GB")
    for p in files:
        print(f"   {p.name:<42} {p.stat().st_size / 1e6:>10.1f} MB")

    cfg = json.loads((ckpt / "config.json").read_text())
    flat, nested = cfg.get("rope_theta"), (cfg.get("rope_parameters") or {}).get("rope_theta")
    print(f"rope   : flat={flat} nested={nested}")
    if flat != 1000000:
        raise SystemExit(f"refusing to upload: flat rope_theta is {flat}, expected 1000000")

    card_text = card(args.repo, args.base, ckpt)
    if args.dry_run:
        print("\n--- README.md (dry run) ---")
        print(card_text)
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    print("repo ready")

    (ckpt / "README.hf.md").write_text(card_text)
    api.upload_file(path_or_fileobj=str(ckpt / "README.hf.md"),
                    path_in_repo="README.md", repo_id=args.repo)
    print("README uploaded")

    for p in files:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=args.repo)
        print(f"  uploaded {p.name}")

    print(f"\ndone: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
