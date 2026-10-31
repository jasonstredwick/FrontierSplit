# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM; modified for FrontierSplit:
# completion formatting, prompt echo, logprobs, and SSE output.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

import asyncio
import contextlib
import itertools
import json
import math
import time
import transformers
import uuid

from collections.abc import AsyncGenerator

from frontiersplit.backend import types as fs_backend_types
from frontiersplit.openai_api import detokenizer as fs_detokenizer
from frontiersplit.openai_api import errors as fs_errors
from frontiersplit.openai_api import protocol as fs_protocol
from frontiersplit.openai_api import renderer as fs_renderer
from frontiersplit.openai_api import types as fs_types

LOGPROB_FLOOR = -9999.0


def _ClampLogprob(*, value: float) -> float:
    if math.isnan(value) or value == math.inf:
        raise fs_backend_types.GenerationError(f"Backend produced invalid logprob: {value}")
    return max(value, LOGPROB_FLOOR)


class OpenAIServing:
    def __init__(self, *, engine: fs_types.GenerationClient | None) -> None:
        self._engine: fs_types.GenerationClient | None = None
        self._renderer: fs_renderer.OnlineRenderer | None = None
        self.SetEngine(engine=engine)

    async def CheckReady(self) -> None:
        await self._RequireReady()

    async def CreateChatCompletion(
        self,
        *,
        request: fs_protocol.ChatCompletionRequest,
    ) -> fs_protocol.ChatCompletionResponse | AsyncGenerator[str]:
        engine, renderer = await self._RequireModel(model=request.model)
        rendered = await renderer.RenderChatAsync(request=request)
        params = fs_types.GenerationParams(
            sampling=fs_backend_types.SamplingParams(
                ignore_eos=request.ignore_eos,
                logprobs=None,
                max_tokens=rendered.effective_max_tokens,
                prompt_logprobs=None,
                seed=request.seed,
                temperature=request.temperature,
                top_p=request.top_p,
            ),
            stop=request.NormalizedStop(),
        )
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            return self._StreamChat(
                created=created,
                engine=engine,
                model_name=request.model,
                rendered=rendered,
                request=request,
                request_id=request_id,
                params=params,
            )

        final_output = await self._CollectSingleOutput(
            engine=engine,
            rendered=rendered,
            request_id=request_id,
            params=params,
        )
        completion = final_output.outputs[0]
        finish_reason = self._RequireFinishReason(
            finish_reason=completion.finish_reason,
            request_id=request_id,
        )
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(completion.token_ids)
        return fs_protocol.ChatCompletionResponse(
            choices=[
                fs_protocol.ChatCompletionChoice(
                    finish_reason=finish_reason,
                    index=0,
                    message=fs_protocol.ChatMessageOutput(content=completion.text),
                )
            ],
            created=created,
            id=request_id,
            model=request.model,
            usage=fs_protocol.UsageInfo(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def CreateCompletion(
        self,
        *,
        request: fs_protocol.CompletionRequest,
    ) -> fs_protocol.CompletionResponse | AsyncGenerator[str]:
        engine, renderer = await self._RequireModel(model=request.model)
        rendered_batch = await renderer.RenderCompletionAsync(request=request)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            rendered_single = rendered_batch[0]
            params = self._BuildCompletionGenerationParams(
                effective_max_tokens=rendered_single.effective_max_tokens,
                request=request,
            )
            return self._StreamCompletion(
                created=created,
                engine=engine,
                model_name=request.model,
                rendered=rendered_single,
                request=request,
                request_id=request_id,
                params=params,
            )

        batch_outputs = await self._CollectBatchOutputs(
            engine=engine,
            rendered_batch=rendered_batch,
            request=request,
            request_id=request_id,
        )

        choices: list[fs_protocol.CompletionChoice] = []
        total_prompt = 0
        total_completion = 0
        for idx, result in enumerate(batch_outputs):
            completion = result.outputs[0]
            finish_reason = self._RequireFinishReason(
                finish_reason=completion.finish_reason,
                request_id=result.request_id,
            )
            total_prompt += len(result.prompt_token_ids)
            total_completion += len(completion.token_ids)

            if request.echo:
                if result.prompt is None:
                    raise fs_backend_types.GenerationError(
                        f"Completed request {result.request_id} missing echoed prompt."
                    )
                text = result.prompt + completion.text
                all_ids = [*result.prompt_token_ids, *completion.token_ids]
            else:
                text = completion.text
                all_ids = completion.token_ids

            if request.logprobs is None:
                logprobs_obj = None
            else:
                if completion.logprobs is None:
                    raise fs_backend_types.GenerationError(
                        f"Completed request {result.request_id} missing completion logprobs."
                    )
                if request.echo:
                    if result.prompt_logprobs is None:
                        raise fs_backend_types.GenerationError(
                            f"Completed request {result.request_id} missing prompt logprobs."
                        )
                    all_lps: list[dict[int, fs_types.TokenLogprob] | None] = [
                        *result.prompt_logprobs,
                        *completion.logprobs,
                    ]
                else:
                    all_lps = list(completion.logprobs)
                logprobs_obj = self._BuildCompletionLogprobs(
                    num_top=request.logprobs,
                    position_logprobs=all_lps,
                    token_ids=all_ids,
                    tokenizer=engine.tokenizer,
                )

            choices.append(
                fs_protocol.CompletionChoice(
                    finish_reason=finish_reason,
                    index=idx,
                    logprobs=logprobs_obj,
                    text=text,
                )
            )

        return fs_protocol.CompletionResponse(
            choices=choices,
            created=created,
            id=request_id,
            model=request.model,
            usage=fs_protocol.UsageInfo(
                completion_tokens=total_completion,
                prompt_tokens=total_prompt,
                total_tokens=total_prompt + total_completion,
            ),
        )

    async def ListModels(self) -> fs_protocol.ModelListResponse:
        engine, _ = await self._RequireReady()
        cfg = engine.model_config
        return fs_protocol.ModelListResponse(
            data=[
                fs_protocol.ModelCard(
                    id=cfg.model_id,
                    max_model_len=cfg.max_model_len,
                    root=cfg.model_id,
                )
            ]
        )

    def SetEngine(self, *, engine: fs_types.GenerationClient | None) -> None:
        if self._renderer is not None:
            self._renderer.Shutdown()
        self._engine = engine
        self._renderer = fs_renderer.OnlineRenderer(engine=engine) if engine is not None else None

    @staticmethod
    def _BuildCompletionLogprobs(
        *,
        num_top: int,
        position_logprobs: list[dict[int, fs_types.TokenLogprob] | None],
        token_ids: list[int],
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> fs_protocol.CompletionLogProbs:
        out_offsets: list[int] = []
        out_token_lps: list[float | None] = []
        out_tokens: list[str] = []
        out_top_lps: list[dict[str, float] | None] = []
        last_len = 0

        for tid, pos_map in zip(token_ids, position_logprobs, strict=True):
            if pos_map is None:
                tok = fs_detokenizer.ConvertIdsToTokens(token_ids=[tid], tokenizer=tokenizer)[0]
                out_tokens.append(tok)
                out_token_lps.append(None)
                out_top_lps.append(None)
            else:
                selected = pos_map[tid]
                tok = (
                    selected.decoded_token
                    if selected.decoded_token is not None
                    else fs_detokenizer.ConvertIdsToTokens(token_ids=[tid], tokenizer=tokenizer)[0]
                )
                selected_lp = _ClampLogprob(value=selected.logprob)
                out_tokens.append(tok)
                out_token_lps.append(selected_lp)

                top_map: dict[str, float] = {}
                for cand_id, cand_info in itertools.islice(pos_map.items(), num_top + 1):
                    cand_tok = (
                        cand_info.decoded_token
                        if cand_info.decoded_token is not None
                        else fs_detokenizer.ConvertIdsToTokens(token_ids=[cand_id], tokenizer=tokenizer)[0]
                    )
                    cand_lp = selected_lp if cand_id == tid else _ClampLogprob(value=cand_info.logprob)
                    prev = top_map.get(cand_tok)
                    if prev is None or cand_lp > prev:
                        top_map[cand_tok] = cand_lp
                out_top_lps.append(top_map)

            out_offsets.append(0 if not out_offsets else out_offsets[-1] + last_len)
            last_len = len(tok)

        return fs_protocol.CompletionLogProbs(
            text_offset=out_offsets,
            token_logprobs=out_token_lps,
            tokens=out_tokens,
            top_logprobs=out_top_lps,
        )

    @staticmethod
    def _BuildCompletionGenerationParams(
        *,
        effective_max_tokens: int,
        request: fs_protocol.CompletionRequest,
    ) -> fs_types.GenerationParams:
        prompt_lps = request.logprobs if (request.echo and request.logprobs is not None) else None
        return fs_types.GenerationParams(
            sampling=fs_backend_types.SamplingParams(
                ignore_eos=request.ignore_eos,
                logprobs=request.logprobs,
                max_tokens=effective_max_tokens,
                prompt_logprobs=prompt_lps,
                seed=request.seed,
                temperature=request.temperature,
                top_p=request.top_p,
            ),
            stop=request.NormalizedStop(),
        )

    async def _CollectBatchOutputs(
        self,
        *,
        engine: fs_types.GenerationClient,
        rendered_batch: list[fs_renderer.RenderedPrompt],
        request: fs_protocol.CompletionRequest,
        request_id: str,
    ) -> list[fs_types.RequestOutput]:
        tasks: list[asyncio.Task[fs_types.RequestOutput]] = []
        try:
            for idx, item in enumerate(rendered_batch):
                params = self._BuildCompletionGenerationParams(
                    effective_max_tokens=item.effective_max_tokens,
                    request=request,
                )
                tasks.append(
                    asyncio.create_task(
                        self._CollectSingleOutput(
                            engine=engine,
                            rendered=item,
                            request_id=f"{request_id}-{idx}",
                            params=params,
                        )
                    )
                )
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
            raise

    @staticmethod
    async def _CollectSingleOutput(
        *,
        engine: fs_types.GenerationClient,
        rendered: fs_renderer.RenderedPrompt,
        request_id: str,
        params: fs_types.GenerationParams,
    ) -> fs_types.RequestOutput:
        last: fs_types.RequestOutput | None = None
        async with contextlib.aclosing(
            engine.Generate(
                delta=False,
                prompt_text=rendered.prompt_text,
                prompt_token_ids=rendered.prompt_token_ids,
                request_id=request_id,
                params=params,
            )
        ) as stream:
            async for step in stream:
                last = step
        if last is None:
            raise fs_backend_types.GenerationError(f"Engine yielded no output for request {request_id}.")
        return last

    @staticmethod
    def _FormatStreamError(*, message: str) -> str:
        payload = fs_errors.FormatErrorPayload(
            code="stream_interrupted",
            error_type="server_error",
            message=message,
        )
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def _RequireFinishReason(
        *,
        finish_reason: fs_types.FinishReason | None,
        request_id: str,
    ) -> fs_types.FinishReason:
        if finish_reason is None:
            raise fs_backend_types.GenerationError(f"Completed request {request_id} missing finish_reason.")
        return finish_reason

    async def _RequireModel(
        self,
        *,
        model: str,
    ) -> tuple[fs_types.GenerationClient, fs_renderer.OnlineRenderer]:
        engine, renderer = await self._RequireReady()
        if model != engine.model_config.model_id:
            raise fs_errors.NotFoundError(
                message=f"The model '{model}' does not exist.",
                param="model",
            )
        return engine, renderer

    async def _RequireReady(self) -> tuple[fs_types.GenerationClient, fs_renderer.OnlineRenderer]:
        if self._engine is None or self._renderer is None:
            raise fs_errors.ServiceUnavailableError(message="No model backend is attached.")
        await self._engine.CheckHealth()
        return self._engine, self._renderer

    async def _StreamChat(
        self,
        *,
        created: int,
        engine: fs_types.GenerationClient,
        model_name: str,
        rendered: fs_renderer.RenderedPrompt,
        request: fs_protocol.ChatCompletionRequest,
        request_id: str,
        params: fs_types.GenerationParams,
    ) -> AsyncGenerator[str]:
        include_usage, continuous_usage = fs_protocol.ShouldIncludeUsage(
            stream_options=request.stream_options
        )
        prompt_tokens = len(rendered.prompt_token_ids)
        completion_tokens = 0
        sent_role = False

        try:
            async with contextlib.aclosing(
                engine.Generate(
                    delta=True,
                    prompt_text=rendered.prompt_text,
                    prompt_token_ids=rendered.prompt_token_ids,
                    request_id=request_id,
                    params=params,
                )
            ) as stream:
                async for result in stream:
                    if not sent_role:
                        role_chunk = fs_protocol.ChatCompletionStreamResponse(
                            choices=[
                                fs_protocol.ChatCompletionStreamChoice(
                                    delta=fs_protocol.ChatDelta(content="", role="assistant"),
                                    finish_reason=None,
                                    index=0,
                                )
                            ],
                            created=created,
                            id=request_id,
                            model=model_name,
                            object="chat.completion.chunk",
                        )
                        if continuous_usage:
                            role_chunk.usage = fs_protocol.UsageInfo(
                                completion_tokens=0,
                                prompt_tokens=prompt_tokens,
                                total_tokens=prompt_tokens,
                            )
                        yield f"data: {role_chunk.model_dump_json(exclude_unset=True)}\n\n"
                        sent_role = True

                    completion = result.outputs[0]
                    completion_tokens += len(completion.token_ids)
                    step_usage = (
                        fs_protocol.UsageInfo(
                            completion_tokens=completion_tokens,
                            prompt_tokens=prompt_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        if continuous_usage
                        else None
                    )

                    if completion.text:
                        content_chunk = fs_protocol.ChatCompletionStreamResponse(
                            choices=[
                                fs_protocol.ChatCompletionStreamChoice(
                                    delta=fs_protocol.ChatDelta(content=completion.text),
                                    finish_reason=None,
                                    index=0,
                                )
                            ],
                            created=created,
                            id=request_id,
                            model=model_name,
                            object="chat.completion.chunk",
                        )
                        if step_usage is not None:
                            content_chunk.usage = step_usage
                        yield f"data: {content_chunk.model_dump_json(exclude_unset=True)}\n\n"

                    if completion.finish_reason is not None:
                        close_chunk = fs_protocol.ChatCompletionStreamResponse(
                            choices=[
                                fs_protocol.ChatCompletionStreamChoice(
                                    delta=fs_protocol.ChatDelta(),
                                    finish_reason=completion.finish_reason,
                                    index=0,
                                )
                            ],
                            created=created,
                            id=request_id,
                            model=model_name,
                            object="chat.completion.chunk",
                        )
                        if step_usage is not None:
                            close_chunk.usage = step_usage
                        yield f"data: {close_chunk.model_dump_json(exclude_unset=True)}\n\n"

            if include_usage:
                final_chunk = fs_protocol.ChatCompletionStreamResponse(
                    choices=[],
                    created=created,
                    id=request_id,
                    model=model_name,
                    object="chat.completion.chunk",
                    usage=fs_protocol.UsageInfo(
                        completion_tokens=completion_tokens,
                        prompt_tokens=prompt_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield self._FormatStreamError(message=str(exc) or type(exc).__name__)
        yield "data: [DONE]\n\n"

    async def _StreamCompletion(
        self,
        *,
        created: int,
        engine: fs_types.GenerationClient,
        model_name: str,
        rendered: fs_renderer.RenderedPrompt,
        request: fs_protocol.CompletionRequest,
        request_id: str,
        params: fs_types.GenerationParams,
    ) -> AsyncGenerator[str]:
        include_usage, continuous_usage = fs_protocol.ShouldIncludeUsage(
            stream_options=request.stream_options
        )
        prompt_tokens = len(rendered.prompt_token_ids)
        completion_tokens = 0

        try:
            async with contextlib.aclosing(
                engine.Generate(
                    delta=True,
                    prompt_text=rendered.prompt_text,
                    prompt_token_ids=rendered.prompt_token_ids,
                    request_id=request_id,
                    params=params,
                )
            ) as stream:
                async for result in stream:
                    completion = result.outputs[0]
                    completion_tokens += len(completion.token_ids)
                    if not completion.text and completion.finish_reason is None:
                        continue

                    chunk = fs_protocol.CompletionStreamResponse(
                        choices=[
                            fs_protocol.CompletionStreamChoice(
                                finish_reason=completion.finish_reason,
                                index=0,
                                logprobs=None,
                                text=completion.text,
                            )
                        ],
                        created=created,
                        id=request_id,
                        model=model_name,
                        object="text_completion",
                    )
                    if continuous_usage:
                        chunk.usage = fs_protocol.UsageInfo(
                            completion_tokens=completion_tokens,
                            prompt_tokens=prompt_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

            if include_usage:
                final_chunk = fs_protocol.CompletionStreamResponse(
                    choices=[],
                    created=created,
                    id=request_id,
                    model=model_name,
                    object="text_completion",
                    usage=fs_protocol.UsageInfo(
                        completion_tokens=completion_tokens,
                        prompt_tokens=prompt_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield self._FormatStreamError(message=str(exc) or type(exc).__name__)
        yield "data: [DONE]\n\n"
