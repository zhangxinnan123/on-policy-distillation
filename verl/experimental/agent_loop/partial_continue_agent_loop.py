# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-turn agent loop that continues a teacher-provided assistant prefix.

Each row carries a ``partial`` text field — a token-aligned prefix of the
teacher's response. The student conditions on ``chat_template(prompt) + partial``
and generates the remainder. The partial is appended to the rendered prompt
(NOT injected as an assistant message) because Qwen3's chat template
auto-injects ``<think>\\n\\n</think>\\n\\n`` whenever it formats an assistant
turn, which would corrupt the partial bytes.
"""
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_partial_continue_agent")
class SingleTurnPartialContinueAgentLoop(AgentLoopBase):
    """Single-turn loop that conditions on prompt + partial assistant prefix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        partial = kwargs.get("partial", "")
        if not isinstance(partial, str) or partial == "":
            raise ValueError(
                f"{type(self).__name__} requires a non-empty 'partial' string in non_tensor_batch; "
                f"empty-partial rows must dispatch to single_turn_agent instead."
            )

        # Multimodal partials are not supported: the partial is bare text and
        # cannot reference image/video placeholders that the processor expects.
        if kwargs.get("multi_modal_data") or kwargs.get("image_data") or self.processor is not None:
            raise NotImplementedError(
                "single_turn_partial_continue_agent does not support multimodal inputs."
            )

        # Render the prompt (no partial) as text. Honor user's
        # apply_chat_template_kwargs (e.g., enable_thinking) but defensively
        # strip keys we control so they can't be double-set below.
        ct_kwargs = dict(self.apply_chat_template_kwargs)
        for reserved in ("add_generation_prompt", "continue_final_message", "tokenize"):
            ct_kwargs.pop(reserved, None)

        prompt_text: str = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **ct_kwargs,
            ),
        )

        # Tokenize prompt and partial separately and concatenate ids: avoids
        # BPE merges across the prompt/partial boundary that would yield a
        # different token sequence than the teacher's actual prefix.
        def _encode_and_concat() -> list[int]:
            prompt_ids_local = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            partial_ids_local = self.tokenizer.encode(partial, add_special_tokens=False)
            return normalize_token_ids(prompt_ids_local) + normalize_token_ids(partial_ids_local)

        prompt_ids = await self.loop.run_in_executor(None, _encode_and_concat)

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        truncation_length = self.response_length
        response_ids = output.token_ids[:truncation_length]
        response_mask = [1] * len(response_ids)

        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=output.log_probs[:truncation_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + truncation_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=None,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        result.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "partial_idx": kwargs.get("partial_idx"),
                "cutoff_tokens": kwargs.get("cutoff_tokens"),
            }
        )
        return result
