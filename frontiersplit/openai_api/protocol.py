# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM; modified for FrontierSplit:
# a restricted API schema with FrontierSplit validation.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

# vLLM's protocol definitions were adapted from FastChat (Apache-2.0):
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py

import pydantic
import typing

from collections.abc import Callable

INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
MAX_TOP_LOGPROBS = 20
MIN_SAMPLING_TEMPERATURE = 0.01
STRICT_CONFIG = pydantic.ConfigDict(allow_inf_nan=False, extra="forbid")

StrictNumber = pydantic.StrictFloat | pydantic.StrictInt


def _CoerceFloat(value: float) -> float:
    return float(value)


def _DefaultIfNone(*, default: float | int) -> Callable[[typing.Any], typing.Any]:
    def Apply(value: typing.Any) -> typing.Any:
        return default if value is None else value

    return Apply


StrictFiniteFloat = typing.Annotated[StrictNumber, pydantic.AfterValidator(_CoerceFloat)]


class StreamOptions(pydantic.BaseModel):
    model_config = STRICT_CONFIG

    continuous_usage_stats: pydantic.StrictBool | None = False
    include_usage: pydantic.StrictBool | None = False


def ShouldIncludeUsage(*, stream_options: StreamOptions | None) -> tuple[bool, bool]:
    if stream_options is None:
        return False, False
    include = bool(stream_options.include_usage)
    continuous = include and bool(stream_options.continuous_usage_stats)
    return include, continuous


class GenerationRequestBase(pydantic.BaseModel):
    model_config = STRICT_CONFIG

    best_of: pydantic.StrictInt | None = 1
    frequency_penalty: StrictNumber | None = 0.0
    ignore_eos: pydantic.StrictBool = False
    logit_bias: dict[pydantic.StrictStr, StrictNumber] | None = None
    metadata: dict[pydantic.StrictStr, typing.Any] | None = None
    model: pydantic.StrictStr
    n: pydantic.StrictInt | None = 1
    presence_penalty: StrictNumber | None = 0.0
    repetition_penalty: StrictNumber | None = 1.0
    seed: pydantic.StrictInt | None = None
    stop: pydantic.StrictStr | list[pydantic.StrictStr] | None = None
    stream: pydantic.StrictBool = False
    stream_options: StreamOptions | None = None
    suffix: None = None
    temperature: typing.Annotated[StrictFiniteFloat, pydantic.BeforeValidator(_DefaultIfNone(default=1.0))] = 1.0
    top_p: typing.Annotated[StrictFiniteFloat, pydantic.BeforeValidator(_DefaultIfNone(default=1.0))] = 1.0
    user: pydantic.StrictStr | None = None

    def NormalizedStop(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        if isinstance(self.stop, str):
            return (self.stop,)
        return tuple(self.stop)

    @pydantic.field_validator("logit_bias")
    @classmethod
    def ValidateEmptyLogitBias(
        cls,
        value: dict[str, float | int] | None,
    ) -> dict[str, float | int] | None:
        if value is not None and len(value) != 0:
            raise ValueError("Non-empty logit_bias is not supported.")
        return value

    @pydantic.field_validator("seed")
    @classmethod
    def ValidateSeed(cls, value: int | None) -> int | None:
        if value is not None and not (INT64_MIN <= value <= INT64_MAX):
            raise ValueError("seed must fit in a signed 64-bit integer.")
        return None if value == -1 else value

    @pydantic.field_validator("n", "best_of")
    @classmethod
    def ValidateSingleCandidate(cls, value: int | None) -> int | None:
        if value not in (None, 1):
            raise ValueError("Only 1 or null is supported.")
        return value

    @pydantic.field_validator("stop")
    @classmethod
    def ValidateStop(cls, value: str | list[str] | None) -> str | list[str] | None:
        if isinstance(value, str) and not value:
            raise ValueError("stop string must be non-empty.")
        if isinstance(value, list) and any(not entry for entry in value):
            raise ValueError("stop list entries must be non-empty strings.")
        return value

    @pydantic.field_validator("temperature")
    @classmethod
    def ValidateTemperature(cls, value: float) -> float:
        if not (0.0 <= value <= 2.0):
            raise ValueError("temperature must satisfy 0.0 <= temperature <= 2.0.")
        # vLLM floors positive temperatures for numerical stability; zero stays greedy.
        if 0.0 < value < MIN_SAMPLING_TEMPERATURE:
            return MIN_SAMPLING_TEMPERATURE
        return value

    @pydantic.field_validator("top_p")
    @classmethod
    def ValidateTopP(cls, value: float) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError("top_p must satisfy 0.0 < top_p <= 1.0.")
        return value

    @pydantic.field_validator("repetition_penalty")
    @classmethod
    def ValidateUnitRepetitionPenalty(cls, value: float | None) -> float | None:
        if value is None:
            return None
        val = float(value)
        if val != 1.0:
            raise ValueError("Only 1.0 or null is supported.")
        return val

    @pydantic.field_validator("frequency_penalty", "presence_penalty")
    @classmethod
    def ValidateZeroPenalty(cls, value: float | None) -> float | None:
        if value is None:
            return None
        val = float(value)
        if val != 0.0:
            raise ValueError("Only 0.0 or null is supported.")
        return val

    @pydantic.model_validator(mode="after")
    def ValidateStreamOptions(self) -> typing.Self:
        if not self.stream and self.stream_options is not None:
            raise ValueError("stream_options requires stream=true.")
        return self


class CompletionRequest(GenerationRequestBase):
    add_special_tokens: pydantic.StrictBool = True
    echo: pydantic.StrictBool = False
    logprobs: pydantic.StrictInt | None = None
    max_tokens: typing.Annotated[pydantic.StrictInt, pydantic.BeforeValidator(_DefaultIfNone(default=16))] = 16
    prompt: (
        pydantic.StrictStr
        | list[pydantic.StrictStr]
        | list[pydantic.StrictInt]
        | list[list[pydantic.StrictInt]]
    )

    @pydantic.field_validator("logprobs")
    @classmethod
    def ValidateLogprobs(cls, value: int | None) -> int | None:
        if value is not None and not (0 <= value <= MAX_TOP_LOGPROBS):
            raise ValueError(f"logprobs must satisfy 0 <= logprobs <= {MAX_TOP_LOGPROBS}.")
        return value

    @pydantic.field_validator("max_tokens")
    @classmethod
    def ValidateMaxTokens(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_tokens must be >= 0.")
        return value

    @pydantic.field_validator("prompt")
    @classmethod
    def ValidatePromptNonEmpty(
        cls,
        value: str | list[str] | list[int] | list[list[int]],
    ) -> str | list[str] | list[int] | list[list[int]]:
        if isinstance(value, str):
            if not value:
                raise ValueError("prompt must be non-empty.")
            return value
        if not value:
            raise ValueError("prompt batch must be non-empty.")
        first = value[0]
        if isinstance(first, str) and any(not item for item in value):
            raise ValueError("All prompts in batch must be non-empty.")
        if isinstance(first, list) and any(not item for item in value):
            raise ValueError("All token-ID prompts in batch must be non-empty.")
        return value

    @pydantic.model_validator(mode="after")
    def ValidateStreamingConstraints(self) -> typing.Self:
        if not self.stream:
            return self
        is_multi_prompt = (
            isinstance(self.prompt, list)
            and isinstance(self.prompt[0], (str, list))
            and len(self.prompt) > 1
        )
        if is_multi_prompt:
            raise ValueError("Streaming completions require a single prompt.")
        if self.max_tokens == 0:
            raise ValueError("Streaming completions require a positive generation budget.")
        if self.echo:
            raise ValueError("Streaming completions do not support echo=true.")
        if self.logprobs is not None:
            raise ValueError("Streaming completions do not support logprobs.")
        return self


class ChatTextPart(pydantic.BaseModel):
    model_config = STRICT_CONFIG

    text: pydantic.StrictStr
    type: typing.Literal["text"]


class ChatMessageInput(pydantic.BaseModel):
    model_config = STRICT_CONFIG

    content: pydantic.StrictStr | list[ChatTextPart]
    role: typing.Literal["system", "user", "assistant"]

    @pydantic.field_validator("content")
    @classmethod
    def ValidateContent(cls, value: str | list[ChatTextPart]) -> str | list[ChatTextPart]:
        if isinstance(value, list) and not value:
            raise ValueError("Message content array must be non-empty.")
        return value


class ChatResponseFormat(pydantic.BaseModel):
    model_config = STRICT_CONFIG

    type: typing.Literal["text"]


class ChatCompletionRequest(GenerationRequestBase):
    add_generation_prompt: pydantic.StrictBool = True
    add_special_tokens: pydantic.StrictBool = False
    max_completion_tokens: pydantic.StrictInt | None = None
    max_tokens: pydantic.StrictInt | None = None
    messages: list[ChatMessageInput]
    response_format: ChatResponseFormat | None = None
    tool_choice: typing.Literal["none"] | None = None
    tools: None = None

    def RequestedMaxTokens(self) -> int | None:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens

    @pydantic.field_validator("messages")
    @classmethod
    def ValidateMessagesNonEmpty(cls, value: list[ChatMessageInput]) -> list[ChatMessageInput]:
        if not value:
            raise ValueError("messages must be non-empty.")
        return value

    @pydantic.field_validator("max_tokens", "max_completion_tokens")
    @classmethod
    def ValidatePositiveChatBudget(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("Chat token budget must be >= 1.")
        return value

    @pydantic.model_validator(mode="after")
    def ValidateMatchingBudgets(self) -> typing.Self:
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError("max_tokens and max_completion_tokens must match when both are set.")
        return self


class ModelCard(pydantic.BaseModel):
    created: int = 0
    id: str
    max_model_len: int
    object: typing.Literal["model"] = "model"
    owned_by: str = "frontiersplit"
    parent: str | None = None
    root: str


class ModelListResponse(pydantic.BaseModel):
    data: list[ModelCard]
    object: typing.Literal["list"] = "list"


class UsageInfo(pydantic.BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class CompletionLogProbs(pydantic.BaseModel):
    text_offset: list[int]
    token_logprobs: list[float | None]
    tokens: list[str]
    top_logprobs: list[dict[str, float] | None]


class CompletionChoice(pydantic.BaseModel):
    finish_reason: typing.Literal["stop", "length"]
    index: int
    logprobs: CompletionLogProbs | None = None
    text: str


class CompletionResponse(pydantic.BaseModel):
    choices: list[CompletionChoice]
    created: int
    id: str
    model: str
    object: typing.Literal["text_completion"] = "text_completion"
    usage: UsageInfo


class CompletionStreamChoice(pydantic.BaseModel):
    finish_reason: typing.Literal["stop", "length"] | None
    index: int
    logprobs: None
    text: str


class CompletionStreamResponse(pydantic.BaseModel):
    choices: list[CompletionStreamChoice]
    created: int
    id: str
    model: str
    object: typing.Literal["text_completion"]
    usage: UsageInfo | None = None


class ChatMessageOutput(pydantic.BaseModel):
    content: str
    role: typing.Literal["assistant"] = "assistant"


class ChatCompletionChoice(pydantic.BaseModel):
    finish_reason: typing.Literal["stop", "length"]
    index: int
    message: ChatMessageOutput


class ChatCompletionResponse(pydantic.BaseModel):
    choices: list[ChatCompletionChoice]
    created: int
    id: str
    model: str
    object: typing.Literal["chat.completion"] = "chat.completion"
    usage: UsageInfo


class ChatDelta(pydantic.BaseModel):
    content: str | None = None
    role: typing.Literal["assistant"] | None = None


class ChatCompletionStreamChoice(pydantic.BaseModel):
    delta: ChatDelta
    finish_reason: typing.Literal["stop", "length"] | None
    index: int


class ChatCompletionStreamResponse(pydantic.BaseModel):
    choices: list[ChatCompletionStreamChoice]
    created: int
    id: str
    model: str
    object: typing.Literal["chat.completion.chunk"]
    usage: UsageInfo | None = None
