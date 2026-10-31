# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM; modified for FrontierSplit:
# reduced output records and the FrontierSplit generation interface.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

import collections.abc
import dataclasses
import transformers
import typing

from frontiersplit.backend import types as fs_backend_types

FinishReason = typing.Literal["stop", "length"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerationParams:
    sampling: fs_backend_types.SamplingParams
    stop: tuple[str, ...] = ()


@dataclasses.dataclass(kw_only=True)
class TokenLogprob:
    decoded_token: str | None = None
    logprob: float


@dataclasses.dataclass(kw_only=True)
class CompletionOutput:
    finish_reason: FinishReason | None = None
    index: int = 0
    logprobs: list[dict[int, TokenLogprob]] | None = None
    text: str
    token_ids: list[int]


@dataclasses.dataclass(kw_only=True)
class RequestOutput:
    finished: bool
    outputs: list[CompletionOutput]
    prompt: str | None = None
    prompt_logprobs: list[dict[int, TokenLogprob] | None] | None = None
    prompt_token_ids: list[int]
    request_id: str


class GenerationClient(typing.Protocol):
    @property
    def chat_template(self) -> str | None: ...

    @property
    def model_config(self) -> fs_backend_types.ModelConfig: ...

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase: ...

    async def Abort(self, *, request_id: str) -> None: ...

    async def CheckHealth(self) -> None: ...

    def Generate(
        self,
        *,
        delta: bool,
        prompt_text: str | None,
        prompt_token_ids: list[int],
        request_id: str,
        params: GenerationParams,
    ) -> collections.abc.AsyncGenerator[RequestOutput]: ...
