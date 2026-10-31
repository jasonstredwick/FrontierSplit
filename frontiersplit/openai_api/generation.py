# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM; modified for FrontierSplit:
# delta/final output assembly around the FrontierSplit backend.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

import asyncio
import collections.abc
import transformers

from frontiersplit.backend import types as fs_backend_types
from frontiersplit.openai_api import detokenizer as fs_detokenizer
from frontiersplit.openai_api import logprobs as fs_logprobs
from frontiersplit.openai_api import types as fs_types

FINISH_REASONS: dict[fs_backend_types.FinishCause, fs_types.FinishReason] = {"budget": "length", "eos": "stop"}


class GenerationAdapter:
    def __init__(
        self,
        *,
        backend: fs_backend_types.InferenceBackend,
        chat_template: str | None = None,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> None:
        self._backend = backend
        self._tokenizer = tokenizer
        self._chat_template = chat_template

    @property
    def chat_template(self) -> str | None:
        return self._chat_template

    @property
    def model_config(self) -> fs_backend_types.ModelConfig:
        return self._backend.model_config

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return self._tokenizer

    async def Abort(self, *, request_id: str) -> None:
        await self._backend.Abort(request_id=request_id)

    async def CheckHealth(self) -> None:
        await self._backend.CheckHealth()

    async def Generate(
        self,
        *,
        delta: bool,
        prompt_text: str | None,
        prompt_token_ids: list[int],
        request_id: str,
        params: fs_types.GenerationParams,
    ) -> collections.abc.AsyncGenerator[fs_types.RequestOutput]:
        sampling_params = params.sampling
        backend_request = fs_backend_types.GenerationRequest(
            prompt_token_ids=prompt_token_ids,
            request_id=request_id,
            sampling_params=sampling_params,
        )
        detokenizer = fs_detokenizer.IncrementalDetokenizer(
            prompt_token_ids=prompt_token_ids,
            stop=params.stop,
            tokenizer=self._tokenizer,
        )
        logprobs_processor = fs_logprobs.LogprobsProcessor(
            num_logprobs=sampling_params.logprobs,
            num_prompt_logprobs=sampling_params.prompt_logprobs,
            prompt_token_ids=prompt_token_ids,
            tokenizer=self._tokenizer,
        )

        sent_offset = 0
        backend_finished = False
        yielded_finished = False
        backend_stream = self._backend.Submit(request=backend_request)
        try:
            async for step in backend_stream:
                if step.prompt_logprobs is not None:
                    logprobs_processor.UpdatePrompt(raw_prompt_logprobs=step.prompt_logprobs)

                matched_stop, num_kept = detokenizer.Update(
                    new_token_ids=step.new_token_ids,
                    stop_terminated=(step.finish_cause == "eos"),
                )
                if sampling_params.logprobs is not None and num_kept > 0:
                    if step.new_logprobs is None or len(step.new_logprobs) < num_kept:
                        raise fs_backend_types.GenerationError(
                            f"Backend omitted sample logprobs for request {request_id}."
                        )
                    logprobs_processor.UpdateSample(raw_step_logprobs=step.new_logprobs[:num_kept])
                elif sampling_params.logprobs is None and step.new_logprobs is not None:
                    raise fs_backend_types.GenerationError(
                        f"Backend emitted unexpected sample logprobs for request {request_id}."
                    )

                if step.finish_cause is not None:
                    backend_finished = True
                finish_reason = FINISH_REASONS[step.finish_cause] if step.finish_cause is not None else None
                if matched_stop:
                    finish_reason = "stop"
                finished = finish_reason is not None
                if finished and not logprobs_processor.prompt_complete:
                    raise fs_backend_types.GenerationError(
                        f"Backend omitted prompt logprobs for request {request_id}."
                    )

                if not delta and not finished:
                    continue

                prompt_lps = logprobs_processor.PopPromptLogprobs(delta=delta)
                text = detokenizer.GetNextOutputText(delta=delta, finished=finished)
                if delta:
                    token_ids = detokenizer.output_token_ids[sent_offset:]
                    if not token_ids and not finished and prompt_lps is None:
                        continue
                    sample_lps = (
                        None
                        if logprobs_processor.sample_logprobs is None
                        else logprobs_processor.sample_logprobs[sent_offset:]
                    )
                    sent_offset = len(detokenizer.output_token_ids)
                else:
                    token_ids = list(detokenizer.output_token_ids)
                    sample_lps = logprobs_processor.sample_logprobs

                yield fs_types.RequestOutput(
                    finished=finished,
                    outputs=[
                        fs_types.CompletionOutput(
                            finish_reason=finish_reason,
                            index=0,
                            logprobs=sample_lps,
                            text=text,
                            token_ids=token_ids,
                        )
                    ],
                    prompt=prompt_text,
                    prompt_logprobs=prompt_lps,
                    prompt_token_ids=prompt_token_ids,
                    request_id=request_id,
                )
                if finished:
                    yielded_finished = True
                    break
            if not yielded_finished:
                raise fs_backend_types.GenerationError(
                    f"Backend stream ended without finish_cause for request {request_id}."
                )
        finally:
            try:
                await asyncio.shield(backend_stream.aclose())
            finally:
                if not backend_finished:
                    await asyncio.shield(self._backend.Abort(request_id=request_id))
