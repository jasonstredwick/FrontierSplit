import collections.abc
import dataclasses
import typing

FinishCause = typing.Literal["budget", "eos"]


class BackendUnavailableError(RuntimeError):
    pass


class GenerationError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelConfig:
    eos_token_ids: frozenset[int] = dataclasses.field(default_factory=frozenset)
    max_model_len: int
    model_id: str
    vocab_size: int

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.max_model_len < 1:
            raise ValueError(f"max_model_len must be >= 1, got {self.max_model_len}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplingParams:
    """Resolved execution controls; callers supply defaults before submitting work.

    A zero budget requests prefill only. Temperature zero selects greedy decoding;
    top_p filters sampling only at positive temperature. A seed applies per request;
    ignore_eos prevents EOS from terminating generation. Scores use the unmodified
    full-vocabulary FP32 log_softmax, before temperature or sampling filters.
    A non-null score count requests the selected token and the top-N candidates.
    """

    ignore_eos: bool
    logprobs: int | None
    max_tokens: int
    prompt_logprobs: int | None
    seed: int | None
    temperature: float
    top_p: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class RawStepLogprobs:
    """Unrounded scores for one position, keyed by token ID rather than text.

    Index 0 is the selected (or actual prompt) token, followed by top-N candidates
    in descending score order. The selected token may also occur among candidates.
    """

    logprobs: list[float]
    token_ids: list[int]


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerationRequest:
    """A caller-validated prompt and controls within the backend's model limits.

    Prompt IDs are nonempty and in vocabulary, the nonnegative generation budget
    fits the remaining context, and the request ID is unique among active work.
    """

    prompt_token_ids: list[int]
    request_id: str
    sampling_params: SamplingParams


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerationStep:
    """Ordered token and score deltas, ending with an explicit execution cause.

    For 'eos', the last new token is the terminating EOS and is included in scores
    and token counts. 'budget' includes prefill-only completion. Prompt score chunks
    cover prompt positions in order, with None only for the first unconditioned
    token; requested prompt scores must be complete by the terminal step.
    Requested sample scores align one-to-one with new_token_ids; unrequested
    scores are None. Every step carries the submitted request ID.
    """

    finish_cause: FinishCause | None = None
    new_logprobs: list[RawStepLogprobs] | None = None
    new_token_ids: list[int]
    prompt_logprobs: list[RawStepLogprobs | None] | None = None
    request_id: str


class InferenceBackend(typing.Protocol):
    """Token-level boundary implemented by local or distributed execution.

    Submit preserves prompt IDs and emits ordered deltas without decoding text.
    Normal completion emits one terminal step; execution failures raise
    GenerationError. BackendUnavailableError signals that work cannot be served.
    Closing the generator releases stream resources; Abort cancels outstanding
    execution for the request, including work stopped by a caller's text policy.
    """

    @property
    def model_config(self) -> ModelConfig: ...

    async def Abort(self, *, request_id: str) -> None: ...

    async def CheckHealth(self) -> None: ...

    def Submit(self, *, request: GenerationRequest) -> collections.abc.AsyncGenerator[GenerationStep]: ...
