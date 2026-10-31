import asyncio
import collections.abc
import torch

from frontiersplit.backend import types as fs_backend_types


class ScriptedBackend:
    """Deterministic backend computing genuine FP32 log_softmax over vocabulary logits."""

    def __init__(
        self,
        *,
        default_tokens: list[int],
        eos_token_ids: frozenset[int],
        max_model_len: int = 32,
        model_id: str = "test-model",
        vocab_size: int,
    ) -> None:
        self._model_config = fs_backend_types.ModelConfig(
            eos_token_ids=eos_token_ids,
            max_model_len=max_model_len,
            model_id=model_id,
            vocab_size=vocab_size,
        )
        self._default_tokens = list(default_tokens)
        self.aborted_request_ids: list[str] = []
        self.active_request_ids: set[str] = set()
        self.fail_after_steps: int | None = None
        self.fail_before_first_step: bool = False
        self.fail_exception: Exception | None = None
        self.fail_prompt_token_ids: tuple[int, ...] | None = None
        self.healthy: bool = True
        self.hold_gate: asyncio.Event | None = None
        self.logit_overrides: dict[int, float] = {}
        self.pause_after_steps: int | None = None
        self.paused_event: asyncio.Event = asyncio.Event()
        self.prefill_chunk_size: int | None = None
        self.step_chunk_size: int = 1
        self.step_started: asyncio.Event = asyncio.Event()
        self.submitted_requests: list[fs_backend_types.GenerationRequest] = []

    @property
    def model_config(self) -> fs_backend_types.ModelConfig:
        return self._model_config

    async def Abort(self, *, request_id: str) -> None:
        self.aborted_request_ids.append(request_id)

    async def CheckHealth(self) -> None:
        if not self.healthy:
            raise fs_backend_types.BackendUnavailableError("Backend is unhealthy.")

    def SetDefaultTokens(self, *, tokens: list[int]) -> None:
        self._default_tokens = list(tokens)

    async def Submit(
        self,
        *,
        request: fs_backend_types.GenerationRequest,
    ) -> collections.abc.AsyncGenerator[fs_backend_types.GenerationStep]:
        self.submitted_requests.append(request)
        if self.fail_before_first_step:
            raise self.fail_exception or fs_backend_types.GenerationError("Simulated pre-step backend failure.")

        self.active_request_ids.add(request.request_id)
        try:
            params = request.sampling_params
            prompt_lps = (
                self._BuildPromptLogprobs(
                    num_top=params.prompt_logprobs,
                    prompt_ids=request.prompt_token_ids,
                )
                if params.prompt_logprobs is not None
                else None
            )

            prompt_len = len(request.prompt_token_ids)
            if self.prefill_chunk_size is not None and prompt_len > self.prefill_chunk_size:
                last_prefill_start = 0
                for p_start in range(0, prompt_len, self.prefill_chunk_size):
                    p_end = min(p_start + self.prefill_chunk_size, prompt_len)
                    if p_end < prompt_len:
                        yield fs_backend_types.GenerationStep(
                            finish_cause=None,
                            new_logprobs=None,
                            new_token_ids=[],
                            prompt_logprobs=prompt_lps[p_start:p_end] if prompt_lps is not None else None,
                            request_id=request.request_id,
                        )
                        await asyncio.sleep(0)
                    else:
                        last_prefill_start = p_start
                first_step_prompt_lps = prompt_lps[last_prefill_start:] if prompt_lps is not None else None
            else:
                first_step_prompt_lps = prompt_lps

            if params.max_tokens == 0:
                yield fs_backend_types.GenerationStep(
                    finish_cause="budget",
                    new_logprobs=[] if params.logprobs is not None else None,
                    new_token_ids=[],
                    prompt_logprobs=first_step_prompt_lps,
                    request_id=request.request_id,
                )
                return

            planned: list[int] = []
            finish_cause: fs_backend_types.FinishCause = "budget"
            for idx in range(params.max_tokens):
                tok = self._default_tokens[idx % len(self._default_tokens)]
                planned.append(tok)
                if not params.ignore_eos and tok in self._model_config.eos_token_ids:
                    finish_cause = "eos"
                    break

            matches_fail_target = (
                self.fail_prompt_token_ids is None
                or tuple(request.prompt_token_ids) == self.fail_prompt_token_ids
            )
            skip_gate = self.fail_prompt_token_ids is not None and matches_fail_target
            for emitted_steps, start in enumerate(range(0, len(planned), self.step_chunk_size)):
                self.step_started.set()
                if not skip_gate:
                    if self.pause_after_steps is not None and emitted_steps >= self.pause_after_steps:
                        self.paused_event.set()
                        if self.hold_gate is not None:
                            await self.hold_gate.wait()
                    elif self.hold_gate is not None and self.pause_after_steps is None:
                        await self.hold_gate.wait()
                if (
                    matches_fail_target
                    and self.fail_after_steps is not None
                    and emitted_steps >= self.fail_after_steps
                ):
                    raise self.fail_exception or fs_backend_types.GenerationError(
                        "Simulated mid-stream backend failure."
                    )

                chunk_ids = planned[start : start + self.step_chunk_size]
                is_last = (start + self.step_chunk_size) >= len(planned)
                chunk_lps = (
                    [
                        self._BuildStepLogprob(num_top=params.logprobs, selected_id=tid)
                        for tid in chunk_ids
                    ]
                    if params.logprobs is not None
                    else None
                )
                yield fs_backend_types.GenerationStep(
                    finish_cause=finish_cause if is_last else None,
                    new_logprobs=chunk_lps,
                    new_token_ids=chunk_ids,
                    prompt_logprobs=first_step_prompt_lps if start == 0 else None,
                    request_id=request.request_id,
                )
                await asyncio.sleep(0)
        finally:
            self.active_request_ids.discard(request.request_id)

    def _BuildPromptLogprobs(
        self,
        *,
        num_top: int,
        prompt_ids: list[int],
    ) -> list[fs_backend_types.RawStepLogprobs | None]:
        out: list[fs_backend_types.RawStepLogprobs | None] = [None]
        for tid in prompt_ids[1:]:
            out.append(self._BuildStepLogprob(num_top=num_top, selected_id=tid))
        return out

    def _BuildStepLogprob(
        self,
        *,
        num_top: int,
        selected_id: int,
    ) -> fs_backend_types.RawStepLogprobs:
        vocab_size = self._model_config.vocab_size
        logits = torch.linspace(0.0, -5.0, steps=vocab_size, dtype=torch.float32)
        logits[selected_id] = 4.0
        for tid, val in self.logit_overrides.items():
            logits[tid] = val
        log_probs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)

        top_k = min(num_top, vocab_size)
        top_vals, top_indices = torch.topk(log_probs, k=top_k)

        ids = [selected_id]
        vals = [float(log_probs[selected_id].item())]
        for cand_val, cand_idx in zip(top_vals.tolist(), top_indices.tolist(), strict=True):
            cand_id = int(cand_idx)
            if cand_id != selected_id:
                ids.append(cand_id)
                vals.append(float(cand_val))
        return fs_backend_types.RawStepLogprobs(logprobs=vals, token_ids=ids)
