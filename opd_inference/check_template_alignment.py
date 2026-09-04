#!/usr/bin/env python3
"""Check whether the templated (prompt + response) sequence aligns between the
student and the teacher, for on-policy distillation.

It reproduces both sides of the real pipeline:

  * Student side  — what the student rolls out from. Mirrors
    verl/utils/dataset/rl_dataset.py + single_turn_agent_loop.py:
        student_tok.apply_chat_template(messages, add_generation_prompt=True,
                                        enable_thinking=<data.apply_chat_template_kwargs>)
    then the student's response ids are appended.

  * Teacher side  — what TeacherModelManager feeds to the teacher server.
    Mirrors verl/experimental/teacher_loop/teacher_manager.py:
        - reapply_chat_template=False: teacher scores the student's EXACT tokens
          (optionally swapping the trailing student-EOS for the teacher-EOS).
        - reapply_chat_template=True:  discard student's prompt tokens, rebuild the
          prompt with the TEACHER tokenizer/template (add_generation_prompt=True,
          enable_thinking=<teacher_model.enable_thinking>), then append the student's
          response ids verbatim; only the response region is used for loss.

The script then reports, for the response region (the only thing that carries loss):
    1. whether the response token ids are identical on both sides;
    2. whether decoding those ids gives the same text under both tokenizers
       (i.e. the shared-vocab assumption holds);
    3. the two prompt renderings, side by side, so you can eyeball the frames.

Run on the cluster (tokenizers must be in the local HF cache):
    ssh sfm-science-sfm-p5-cluster "cd /fsx/xinnanzh/on-policy-distillation && \
        python opd_inference/check_template_alignment.py \
            --student-model Qwen/Qwen3-4B-Base \
            --teacher-model Qwen/Qwen3-8B \
            --prompt 'What is 2+2?' \
            --answer '<think>\n\n</think>\n\nThe answer is 4.' \
            --data-enable-thinking false"
"""

from __future__ import annotations

import argparse
import json
import sys

from transformers import AutoTokenizer


def _bool(s):
    if s is None:
        return None
    return str(s).strip().lower() in ("1", "true", "yes", "y", "t")


def _normalize(tokenized):
    """Mirror verl.utils.tokenizer.normalize_token_ids: flatten transformers>=5
    list[list[int]] into list[int]."""
    if len(tokenized) > 0 and isinstance(tokenized[0], list):
        assert len(tokenized) == 1, "expected a single sequence"
        return list(tokenized[0])
    return list(tokenized)


def _apply(tok, messages, *, add_generation_prompt, **kwargs):
    return _normalize(
        tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    )


def build_student_sequence(student_tok, messages, answer_text, *, enable_thinking, add_student_eos):
    apply_kwargs = {}
    if enable_thinking is not None:
        apply_kwargs["enable_thinking"] = enable_thinking
    prompt_ids = _apply(student_tok, messages, add_generation_prompt=True, **apply_kwargs)
    response_ids = student_tok.encode(answer_text, add_special_tokens=False)
    if add_student_eos and student_tok.eos_token_id is not None:
        response_ids = response_ids + [student_tok.eos_token_id]
    return prompt_ids, response_ids


def build_teacher_sequence(
    teacher_tok,
    messages,
    student_response_ids,
    *,
    reapply_chat_template,
    enable_thinking,
    substitute_eos_token,
    student_eos_token_id,
    student_prompt_ids,
):
    """Return (teacher_prompt_ids, teacher_response_ids, response_start).

    Mirrors teacher_manager._build_teacher_sequence_ids (reapply path) and the
    else-branch of compute_teacher_logprobs_batch (non-reapply path).
    """
    response_list = list(student_response_ids)
    # EOS substitution: swap trailing student-EOS -> teacher '<|im_end|>' if requested.
    teacher_eos_id = teacher_tok.convert_tokens_to_ids("<|im_end|>") if substitute_eos_token else None
    if (
        substitute_eos_token
        and response_list
        and student_eos_token_id is not None
        and response_list[-1] == student_eos_token_id
        and teacher_eos_id is not None
    ):
        response_list = list(response_list)
        response_list[-1] = teacher_eos_id

    if reapply_chat_template:
        apply_kwargs = {}
        if enable_thinking is not None:
            apply_kwargs["enable_thinking"] = enable_thinking
        teacher_prompt_ids = _apply(teacher_tok, messages, add_generation_prompt=True, **apply_kwargs)
    else:
        # Teacher scores the student's exact prompt tokens (shared tokenizer assumed).
        teacher_prompt_ids = list(student_prompt_ids)

    return teacher_prompt_ids, response_list, len(teacher_prompt_ids)


def _dec(tok, ids):
    return tok.decode(ids, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--teacher-model", default="Qwen/Qwen3-8B")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="Raw user prompt string (wrapped as one user message).")
    src.add_argument(
        "--messages-json",
        help="JSON list of {role, content} messages (verl 'raw_prompt' style). Overrides --prompt.",
    )
    ap.add_argument("--answer", required=True, help="The student's response text (verbatim, incl. any <think> block).")
    ap.add_argument(
        "--data-enable-thinking",
        default=None,
        help="Student-side data.apply_chat_template_kwargs.enable_thinking (true/false/unset).",
    )
    ap.add_argument(
        "--teacher-enable-thinking",
        default=None,
        help="Teacher-side distillation.teacher_model.enable_thinking (only used when --reapply).",
    )
    ap.add_argument("--reapply", action="store_true", help="distillation.teacher_model.reapply_chat_template=True")
    ap.add_argument(
        "--substitute-eos", action="store_true", help="distillation.teacher_model.substitute_eos_token=True"
    )
    ap.add_argument("--no-student-eos", action="store_true", help="Do not append the student EOS to the response.")
    args = ap.parse_args()

    if args.messages_json:
        messages = json.loads(args.messages_json)
    elif args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
    else:
        ap.error("Provide --prompt or --messages-json")

    data_enable_thinking = _bool(args.data_enable_thinking)
    teacher_enable_thinking = _bool(args.teacher_enable_thinking)

    print(f"Loading student tokenizer: {args.student_model}")
    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    print(f"Loading teacher tokenizer: {args.teacher_model}")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    student_prompt_ids, student_response_ids = build_student_sequence(
        student_tok,
        messages,
        args.answer,
        enable_thinking=data_enable_thinking,
        add_student_eos=not args.no_student_eos,
    )
    teacher_prompt_ids, teacher_response_ids, response_start = build_teacher_sequence(
        teacher_tok,
        messages,
        student_response_ids,
        reapply_chat_template=args.reapply,
        enable_thinking=teacher_enable_thinking,
        substitute_eos_token=args.substitute_eos,
        student_eos_token_id=student_tok.eos_token_id,
        student_prompt_ids=student_prompt_ids,
    )

    hr = "=" * 78
    print(f"\n{hr}\nCONFIG\n{hr}")
    print(f"  reapply_chat_template   = {args.reapply}")
    print(f"  substitute_eos_token    = {args.substitute_eos}")
    print(f"  data enable_thinking    = {data_enable_thinking}  (student prompt)")
    print(f"  teacher enable_thinking = {teacher_enable_thinking}  (used only if reapply)")
    print(f"  student eos_token_id    = {student_tok.eos_token_id} ({student_tok.eos_token!r})")
    tim = teacher_tok.convert_tokens_to_ids("<|im_end|>")
    print(f"  teacher <|im_end|> id   = {tim}")

    print(f"\n{hr}\nSTUDENT PROMPT (decoded)\n{hr}\n{_dec(student_tok, student_prompt_ids)}")
    print(f"\n{hr}\nTEACHER PROMPT (decoded)\n{hr}\n{_dec(teacher_tok, teacher_prompt_ids)}")

    print(f"\n{hr}\nRESPONSE REGION\n{hr}")
    print(f"  student response len = {len(student_response_ids)}  |  teacher response len = {len(teacher_response_ids)}")
    print(f"  response_start (teacher prompt len) = {response_start}")

    # --- Alignment checks -----------------------------------------------------
    print(f"\n{hr}\nALIGNMENT CHECKS\n{hr}")
    ok = True

    # 1) Response ids identical (ignoring an intentional EOS swap on the last token).
    def _strip_eos_swap(s_ids, t_ids):
        if (
            args.substitute_eos
            and s_ids
            and t_ids
            and s_ids[-1] == student_tok.eos_token_id
            and t_ids[-1] == tim
        ):
            return s_ids[:-1], t_ids[:-1]
        return s_ids, t_ids

    s_body, t_body = _strip_eos_swap(list(student_response_ids), list(teacher_response_ids))
    ids_match = s_body == t_body
    ok &= ids_match
    print(f"  [{'PASS' if ids_match else 'FAIL'}] response token ids identical (excl. EOS swap)")
    if not ids_match:
        # Show first divergence.
        for i, (a, b) in enumerate(zip(s_body, t_body)):
            if a != b:
                print(f"        first diff at response idx {i}: student={a} teacher={b}")
                break
        if len(s_body) != len(t_body):
            print(f"        length mismatch: student={len(s_body)} teacher={len(t_body)}")

    # 2) Shared-vocab: decoding the response ids gives the same text under both tokenizers.
    s_txt = _dec(student_tok, student_response_ids)
    t_txt = _dec(teacher_tok, teacher_response_ids)
    # Compare with EOS-swap normalized away for text comparison too.
    s_txt_n = _dec(student_tok, s_body)
    t_txt_n = _dec(teacher_tok, t_body)
    vocab_ok = s_txt_n == t_txt_n
    ok &= vocab_ok
    print(f"  [{'PASS' if vocab_ok else 'FAIL'}] response decodes to same text under both tokenizers (shared vocab)")
    if not vocab_ok:
        print(f"        student: {s_txt_n!r}")
        print(f"        teacher: {t_txt_n!r}")

    # 3) Vocab-size sanity (a quick shared-tokenizer heuristic).
    same_vocab_size = student_tok.vocab_size == teacher_tok.vocab_size
    print(
        f"  [{'PASS' if same_vocab_size else 'WARN'}] vocab_size match "
        f"(student={student_tok.vocab_size}, teacher={teacher_tok.vocab_size})"
    )

    print(f"\n  Response text (student): {s_txt!r}")
    print(f"  Response text (teacher): {t_txt!r}")

    print(f"\n{hr}")
    if ok:
        print("RESULT: ALIGNED ✅  (teacher will score the same response tokens the student produced)")
        rc = 0
    else:
        print("RESULT: MISALIGNED ❌  (see failing checks above)")
        rc = 1
    print(hr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
