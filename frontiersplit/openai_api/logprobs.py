# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM; modified for FrontierSplit:
# prompt/sample score processing and UTF-8 token reconstruction.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

import transformers

from frontiersplit.backend import types as fs_backend_types
from frontiersplit.openai_api import detokenizer as fs_detokenizer
from frontiersplit.openai_api import types as fs_types

MAX_BYTE_CONTEXT_TOKENS = 4


class LogprobsProcessor:
    def __init__(
        self,
        *,
        num_logprobs: int | None,
        num_prompt_logprobs: int | None,
        prompt_token_ids: list[int],
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_token_ids = prompt_token_ids
        self._num_prompt_processed = 0
        self._prompt_popped = False
        self._sample_logprobs: list[dict[int, fs_types.TokenLogprob]] | None = (
            None if num_logprobs is None else []
        )
        self._prompt_logprobs: list[dict[int, fs_types.TokenLogprob] | None] | None = (
            None if num_prompt_logprobs is None else []
        )

    @property
    def prompt_complete(self) -> bool:
        return self._prompt_logprobs is None or self._num_prompt_processed == len(self._prompt_token_ids)

    @property
    def sample_logprobs(self) -> list[dict[int, fs_types.TokenLogprob]] | None:
        return self._sample_logprobs

    def PopPromptLogprobs(self, *, delta: bool) -> list[dict[int, fs_types.TokenLogprob] | None] | None:
        if self._prompt_logprobs is None or self._num_prompt_processed < len(self._prompt_token_ids):
            return None
        if not delta:
            return self._prompt_logprobs
        if self._prompt_popped:
            return None
        self._prompt_popped = True
        return self._prompt_logprobs

    def UpdatePrompt(self, *, raw_prompt_logprobs: list[fs_backend_types.RawStepLogprobs | None]) -> None:
        if self._prompt_logprobs is None:
            raise fs_backend_types.GenerationError("Backend emitted unexpected prompt_logprobs.")
        start = self._num_prompt_processed
        end = start + len(raw_prompt_logprobs)
        if end > len(self._prompt_token_ids):
            raise fs_backend_types.GenerationError(
                f"Backend emitted {end} prompt_logprobs for {len(self._prompt_token_ids)} prompt tokens."
            )
        context_ids = list(self._prompt_token_ids[max(0, start - MAX_BYTE_CONTEXT_TOKENS) : start])
        for prompt_tid, entry in zip(self._prompt_token_ids[start:end], raw_prompt_logprobs, strict=True):
            if entry is None:
                self._prompt_logprobs.append(None)
                context_ids.append(prompt_tid)
            else:
                if not entry.token_ids or entry.token_ids[0] != prompt_tid:
                    raise fs_backend_types.GenerationError(
                        "Prompt RawStepLogprobs[0] must match the prompt token at that position."
                    )
                pos_map = self._BuildPositionMap(context_ids=context_ids, raw=entry)
                self._prompt_logprobs.append(pos_map)
                context_ids.append(entry.token_ids[0])
            if len(context_ids) > MAX_BYTE_CONTEXT_TOKENS:
                del context_ids[:-MAX_BYTE_CONTEXT_TOKENS]
        self._num_prompt_processed = end

    def UpdateSample(self, *, raw_step_logprobs: list[fs_backend_types.RawStepLogprobs]) -> None:
        if self._sample_logprobs is None:
            raise fs_backend_types.GenerationError("Backend emitted unexpected sample logprobs.")
        for entry in raw_step_logprobs:
            context_ids = self._RecentSampleContextIds()
            pos_map = self._BuildPositionMap(context_ids=context_ids, raw=entry)
            self._sample_logprobs.append(pos_map)

    def _BuildPositionMap(
        self,
        *,
        context_ids: list[int],
        raw: fs_backend_types.RawStepLogprobs,
    ) -> dict[int, fs_types.TokenLogprob]:
        if not raw.token_ids:
            raise fs_backend_types.GenerationError("RawStepLogprobs.token_ids must be non-empty.")
        decoded = fs_detokenizer.ConvertIdsToTokens(token_ids=raw.token_ids, tokenizer=self._tokenizer)
        for idx, text in enumerate(decoded):
            if text.endswith(fs_detokenizer.REPLACEMENT_CHAR):
                decoded[idx] = self._CorrectDecodedToken(context_ids=context_ids, token_id=raw.token_ids[idx])
        return {
            tid: fs_types.TokenLogprob(decoded_token=tok, logprob=lp)
            for tid, lp, tok in zip(raw.token_ids, raw.logprobs, decoded, strict=True)
        }

    def _CorrectDecodedToken(self, *, context_ids: list[int], token_id: int) -> str:
        """Reconstruct completed UTF-8 chars split across byte-fallback tokens."""
        max_ctx = min(len(context_ids), MAX_BYTE_CONTEXT_TOKENS)
        for num_ctx in range(1, max_ctx + 1):
            ctx = context_ids[-num_ctx:]
            full_decoded = self._tokenizer.decode(ctx + [token_id])
            if full_decoded.endswith(fs_detokenizer.REPLACEMENT_CHAR):
                continue
            clean_end = len(ctx)
            for j in range(len(ctx) - 1, -1, -1):
                if self._tokenizer.decode([ctx[j]]).endswith(fs_detokenizer.REPLACEMENT_CHAR):
                    clean_end = j
                else:
                    break
            clean_prefix = self._tokenizer.decode(ctx[:clean_end]) if clean_end > 0 else ""
            if full_decoded.startswith(clean_prefix):
                return full_decoded[len(clean_prefix) :]
            common_len = 0
            for a, b in zip(clean_prefix, full_decoded):
                if a != b:
                    break
                common_len += 1
            return full_decoded[common_len:]
        return ""

    def _RecentSampleContextIds(self) -> list[int]:
        if not self._sample_logprobs:
            return self._prompt_token_ids[-MAX_BYTE_CONTEXT_TOKENS:]
        tail = self._sample_logprobs[-MAX_BYTE_CONTEXT_TOKENS:]
        sampled = [next(iter(pos)) for pos in tail]
        if len(sampled) < MAX_BYTE_CONTEXT_TOKENS:
            needed = MAX_BYTE_CONTEXT_TOKENS - len(sampled)
            return self._prompt_token_ids[-needed:] + sampled
        return sampled
