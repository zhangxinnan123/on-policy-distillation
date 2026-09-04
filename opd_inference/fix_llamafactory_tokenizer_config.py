"""Make a LlamaFactory-saved checkpoint loadable by the inference env.

Root cause: the training env (llama_factory) runs transformers 5.2.0, which
writes `extra_special_tokens` into tokenizer_config.json as a LIST. The
inference env (opd) runs transformers 4.55.2, whose
`_set_model_specific_special_tokens` does `special_tokens.keys()` and therefore
raises:

    AttributeError: 'list' object has no attribute 'keys'

Upstream Qwen3 has no `extra_special_tokens` key at all -- those 13 tokens live
under `additional_special_tokens` -- so the fix is to rename the key.

Optionally also repairs generation_config.json when its eos_token_id omits
<|im_end|> (151645). vLLM ignores that field (it resolves EOS from the
tokenizer), but transformers' .generate() does not, so a chat-tuned checkpoint
with the base model's eos_token_id will fail to stop there.

Idempotent: safe to run repeatedly, and on already-correct checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IM_END = 151645
ENDOFTEXT = 151643


def fix_rope_theta(ckpt: Path, backup: bool) -> str:
    """Re-flatten rope_theta so an older transformers can still find it.

    transformers 5.2.0 (the training env) moved rope_theta out of config.json's
    top level into a nested `rope_parameters` dict. transformers 4.55.2 (the
    inference env) does not know that field, fails to find `rope_theta`, and
    silently falls back to Qwen3Config's default of 10000.0 -- 100x off from
    Qwen3's actual 1000000. Positional encoding is then wrong, so generations
    start coherent and degenerate into garbage as the context grows.
    """
    p = ckpt / "config.json"
    if not p.exists():
        return "config.json missing"
    c = json.loads(p.read_text())
    nested = c.get("rope_parameters")
    if not isinstance(nested, dict) or "rope_theta" not in nested:
        return f"ok (rope_theta={c.get('rope_theta', '<absent>')})"
    if "rope_theta" in c:
        return f"ok (rope_theta={c['rope_theta']} already flat)"
    if backup:
        shutil.copy(p, str(p) + ".bak")
    c["rope_theta"] = nested["rope_theta"]
    p.write_text(json.dumps(c, indent=2, ensure_ascii=False))
    return f"FIXED: rope_theta re-flattened to {nested['rope_theta']} (was resolving to 10000.0)"


def fix_tokenizer_config(ckpt: Path, backup: bool) -> str:
    p = ckpt / "tokenizer_config.json"
    if not p.exists():
        return "tokenizer_config.json missing"
    c = json.loads(p.read_text())
    extra = c.get("extra_special_tokens")
    if not isinstance(extra, list):
        return "ok (no list-form extra_special_tokens)"
    if backup:
        shutil.copy(p, str(p) + ".bak")
    c.pop("extra_special_tokens")
    # Don't clobber an existing correct field.
    c.setdefault("additional_special_tokens", extra)
    p.write_text(json.dumps(c, indent=2, ensure_ascii=False))
    return f"FIXED: extra_special_tokens(list, {len(extra)}) -> additional_special_tokens"


def fix_generation_config(ckpt: Path, backup: bool) -> str:
    p = ckpt / "generation_config.json"
    if not p.exists():
        return "generation_config.json missing"
    c = json.loads(p.read_text())
    eos = c.get("eos_token_id")
    ids = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
    if IM_END in ids:
        return "ok (eos already includes <|im_end|>)"
    if backup:
        shutil.copy(p, str(p) + ".bak")
    c["eos_token_id"] = [IM_END, ENDOFTEXT]
    p.write_text(json.dumps(c, indent=2, ensure_ascii=False))
    return f"FIXED: eos_token_id {eos} -> [{IM_END}, {ENDOFTEXT}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="Checkpoint dirs to repair.")
    ap.add_argument("--no_backup", action="store_true", help="Skip writing .bak files.")
    ap.add_argument(
        "--also_generation_config",
        action="store_true",
        help="Also add <|im_end|> to generation_config.json's eos_token_id.",
    )
    args = ap.parse_args()

    for d in args.checkpoints:
        ckpt = Path(d)
        print(f"=== {ckpt}")
        if not ckpt.is_dir():
            print("    SKIP: not a directory")
            continue
        print(f"    config(rope_theta): {fix_rope_theta(ckpt, not args.no_backup)}")
        print(f"    tokenizer_config  : {fix_tokenizer_config(ckpt, not args.no_backup)}")
        if args.also_generation_config:
            print(f"    generation_config : {fix_generation_config(ckpt, not args.no_backup)}")


if __name__ == "__main__":
    main()
