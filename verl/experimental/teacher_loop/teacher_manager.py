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
import asyncio
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn import functional as F

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import hf_tokenizer, normalize_token_ids
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _get_teacher_sampling_params(
    distillation_config: DistillationConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if distillation_config.teacher_model.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": distillation_config.teacher_model.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_next_token_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding_2d = (0, 0, left_pad_size, right_pad_size)  # for (S, K) tensors
    padding_1d = (left_pad_size, right_pad_size)  # for (S,) tensor
    return (
        F.pad(teacher_ids, padding_2d, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding_2d, value=0.0).unsqueeze(0),
        F.pad(teacher_next_token_logprobs, padding_1d, value=0.0).unsqueeze(0),
    )


def _unpad_teacher_inputs(data: DataProto) -> tuple[list[int], int, int]:
    """Unpad valid sequence ids and prompt/response lengths from a single sample.
    The sample is a left-padded prompt concatenated with a right-padded response.
    TODO(wuxibin): remove padding and use tensordict.
    """
    assert len(data) == 1, "Teacher logprob computation expects a single sample"

    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]
    prompt_width = data.batch["prompts"][0].shape[0]
    response_width = data.batch["responses"][0].shape[0]
    assert attention_mask.shape[0] == prompt_width + response_width, (
        "attention_mask sequence length must match prompt and response widths"
    )
    valid_prompt_length = int(attention_mask[:prompt_width].sum().item())
    valid_response_length = int(attention_mask[-response_width:].sum().item())
    prompt_num_padding = prompt_width - valid_prompt_length
    sequence_ids = input_ids[prompt_num_padding : prompt_width + valid_response_length]
    sequence_ids = normalize_token_ids(sequence_ids)
    return sequence_ids, valid_prompt_length, valid_response_length


class AsyncTeacherLLMServerManager(AsyncLLMServerManager):
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        distillation_config: DictConfig | DistillationConfig,
        pad_token_id: int,
        student_eos_token_id: Optional[int] = None,
        teacher_eos_token_id: Optional[int] = None,
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id
        self.student_eos_token_id = student_eos_token_id
        self.reapply_chat_template = self.distillation_config.teacher_model.reapply_chat_template
        self.substitute_eos_token = self.distillation_config.teacher_model.substitute_eos_token
        self.enable_thinking = self.distillation_config.teacher_model.enable_thinking
        if self.reapply_chat_template:
            teacher_model_path = self.distillation_config.teacher_model.model_path
            if not teacher_model_path:
                raise ValueError(
                    "distillation.teacher_model.model_path is required when reapply_chat_template is True."
                )
            self.teacher_tokenizer = hf_tokenizer(teacher_model_path)
            self.teacher_eos_token_id = self.teacher_tokenizer.convert_tokens_to_ids("<|im_end|>")
        else:
            self.teacher_tokenizer = None
            self.teacher_eos_token_id = teacher_eos_token_id if self.substitute_eos_token else None

    def _build_teacher_sequence_ids(
        self,
        raw_prompt: Any,
        student_response_ids: list[int],
    ) -> tuple[list[int], int]:
        """Apply teacher chat template to the prompt, then concat the student's response ids.

        Response is appended verbatim (no assistant-turn wrapping) — relies on the student
        and teacher sharing a tokenizer so the ids are valid in the teacher's frame.

        Returns:
            (teacher_sequence_ids, response_start) where response_start is the index in
            teacher_sequence_ids at which the student's response begins.
        """
        messages = list(raw_prompt)
        teacher_prompt_ids = self.teacher_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        teacher_prompt_ids = normalize_token_ids(teacher_prompt_ids)
        response_list = (
            student_response_ids.tolist()
            if hasattr(student_response_ids, "tolist")
            else list(student_response_ids)
        )
        if (
            response_list
            and self.student_eos_token_id is not None
            and response_list[-1] == self.student_eos_token_id
        ):
            response_list[-1] = self.teacher_eos_token_id
        teacher_sequence_ids = list(teacher_prompt_ids) + response_list
        return teacher_sequence_ids, len(teacher_prompt_ids)

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        response_slice: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence.

        If `response_slice=(start, end)` is provided, the returned tensors are sliced
        along the sequence dimension to cover only `[start, end)`. Used by the
        `reapply_chat_template` path to return response-only tensors when the teacher
        sequence has a different prompt tokenization than the student.
        """
        multi_modal_data = multi_modal_data or {}
        teacher_output = await self.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
        )
        # Shapes: (S, K) where S is sequence length, K is 1 or topk
        # int64 required because downstream losses use teacher_ids as a gather index.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int64)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        # Shape: (S,) — teacher logprob for the actual next token at each position
        teacher_next_token_logprobs = torch.tensor(teacher_output.extra_fields["prompt_next_token_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == teacher_next_token_logprobs.shape[0] == len(sequence_ids)
        # breakpoint()  # for debugging; remove later
        if response_slice is not None:
            start, end = response_slice
            teacher_ids = teacher_ids[start:end]
            teacher_logprobs = teacher_logprobs[start:end]
            teacher_next_token_logprobs = teacher_next_token_logprobs[start:end]
        return teacher_ids, teacher_logprobs, teacher_next_token_logprobs

    async def compute_teacher_logprobs_batch(self, data: DataProto) -> DataProto:
        """Compute teacher log probabilities for a batch of prompt-response pairs."""
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        raw_prompt_batch = data.non_tensor_batch.get("raw_prompt") if self.reapply_chat_template else None
        if self.reapply_chat_template and raw_prompt_batch is None:
            raise ValueError(
                "reapply_chat_template=True requires 'raw_prompt' in non_tensor_batch; "
                "ensure the dataset attaches raw_prompt to each sample."
            )
        tasks = []
        lengths = []
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        # Compute logprobs for each sample in the batch
        for i in range(len(data)):
            item = data[i : i + 1]
            sequence_ids, prompt_length, response_length = _unpad_teacher_inputs(item)
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            # breakpoint()  # for debugging; remove later
            if self.reapply_chat_template:
                student_response_ids = sequence_ids[prompt_length:]
                teacher_sequence_ids, response_start = self._build_teacher_sequence_ids(
                    raw_prompt=raw_prompt_batch[i],
                    student_response_ids=student_response_ids,
                )
                # After slicing in compute_teacher_logprobs_single, returned tensors span
                # only the response; _pad_teacher_outputs is called with prompt_length=0
                # so the entire student prompt region is left-padded.
                lengths.append((0, response_length))
                tasks.append(
                    asyncio.create_task(
                        self.compute_teacher_logprobs_single(
                            sequence_ids=teacher_sequence_ids,
                            multi_modal_data=multi_modal_data,
                            response_slice=(response_start, response_start + response_length),
                        )
                    )
                )
            else:
                if (
                    self.student_eos_token_id is not None
                    and self.teacher_eos_token_id is not None
                    and len(sequence_ids) > 0
                    and sequence_ids[-1] == self.student_eos_token_id
                ):
                    sequence_ids = list(sequence_ids)
                    sequence_ids[-1] = self.teacher_eos_token_id
                lengths.append((prompt_length, response_length))
                tasks.append(
                    asyncio.create_task(
                        self.compute_teacher_logprobs_single(
                            sequence_ids=sequence_ids,
                            multi_modal_data=multi_modal_data,
                        )
                    )
                )
        outputs = await asyncio.gather(*tasks)

        # Pad the teacher logprobs, ids, and next-token logprobs
        padded_teacher_ids = []
        padded_teacher_logprobs = []
        padded_teacher_next_token_logprobs = []
        for (teacher_ids, teacher_logprobs, teacher_next_token_logprobs), (prompt_length, response_length) in zip(
            outputs, lengths, strict=True
        ):
            padded_ids, padded_logprobs, padded_next_token_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                teacher_next_token_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=prompt_length,
                response_length=response_length,
                pad_token_id=self.pad_token_id,
            )
            padded_teacher_ids.append(padded_ids)
            padded_teacher_logprobs.append(padded_logprobs)
            padded_teacher_next_token_logprobs.append(padded_next_token_logprobs)

        batch = TensorDict(
            {
                "teacher_ids": torch.cat(padded_teacher_ids),
                "teacher_logprobs": torch.cat(padded_teacher_logprobs),
                "teacher_next_token_logprobs": torch.cat(padded_teacher_next_token_logprobs),
            },
            batch_size=len(data),
        )
        return DataProto(batch=batch)
