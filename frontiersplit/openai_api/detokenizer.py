# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM; modified for FrontierSplit:
# incremental decoding, token conversion, and stop handling.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

import bisect
import json
import sys
import tokenizers.decoders
import transformers

INITIAL_DETOKENIZATION_OFFSET = 5
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"
REPLACEMENT_CHAR = "\ufffd"
SPACE_MARKER_ATTR = "_fs_space_marker"
UNSET_MARKER = "__unset__"


def GetLeadingSpaceMarker(*, tokenizer: transformers.PreTrainedTokenizerBase) -> str | None:
    """Return the Metaspace replacement char stripped by single-token decode(), if any."""
    cached = getattr(tokenizer, SPACE_MARKER_ATTR, UNSET_MARKER)
    if cached != UNSET_MARKER:
        return cached

    result: str | None = None
    if isinstance(tokenizer, transformers.TokenizersBackend):
        config = json.loads(tokenizer.backend_tokenizer.to_str())
        pre = config.get("pre_tokenizer")
        if isinstance(pre, dict):
            pre_type = pre.get("type")
            if pre_type == "Metaspace":
                result = pre.get("replacement", "▁")
            elif pre_type == "Sequence":
                for sub in pre.get("pretokenizers", ()):
                    if sub.get("type") == "Metaspace":
                        result = sub.get("replacement", "▁")
                        break

    setattr(tokenizer, SPACE_MARKER_ATTR, result)
    return result


def _RestoreLeadingSpaces(*, marker: str, raw_token: str | None, token_str: str) -> str:
    if not raw_token:
        return token_str
    num_markers = 0
    for ch in raw_token:
        if ch != marker:
            break
        num_markers += 1
    if num_markers == 0:
        return token_str
    existing = len(token_str) - len(token_str.lstrip(" "))
    missing = num_markers - existing
    return (" " * missing + token_str) if missing > 0 else token_str


def ConvertIdsToTokens(*, token_ids: list[int], tokenizer: transformers.PreTrainedTokenizerBase) -> list[str]:
    if not token_ids:
        return []
    marker = GetLeadingSpaceMarker(tokenizer=tokenizer)
    if marker is None:
        return [tokenizer.decode([tid]) for tid in token_ids]
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return [
        _RestoreLeadingSpaces(marker=marker, raw_token=raw, token_str=tokenizer.decode([tid]))
        for tid, raw in zip(token_ids, raw_tokens, strict=True)
    ]


def CheckStopStrings(*, new_char_count: int, output_text: str, stop: tuple[str, ...]) -> tuple[str, int] | None:
    """Find the stop string that completes earliest in the newly appended suffix."""
    if not new_char_count or not stop:
        return None

    best_stop: str | None = None
    best_index = 0
    best_end = sys.maxsize
    for stop_str in stop:
        search_start = max(0, len(output_text) - new_char_count - len(stop_str) + 1)
        idx = output_text.find(stop_str, search_start)
        if idx == -1:
            continue
        end = idx + len(stop_str)
        if end < best_end:
            best_stop = stop_str
            best_index = idx
            best_end = end

    if best_stop is None:
        return None
    return best_stop, best_index


def _ConvertPromptIdsToTokens(
    *,
    prompt_ids: list[int],
    skip_special_tokens: bool,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> tuple[list[str], int, int]:
    tail = prompt_ids[-INITIAL_DETOKENIZATION_OFFSET - 2 :]
    raw_tokens = tokenizer.convert_ids_to_tokens(tail, skip_special_tokens=skip_special_tokens)
    tokens = ["" if tok is None else tok for tok in raw_tokens]
    read_offset = len(tokens)
    prefix_offset = max(read_offset - INITIAL_DETOKENIZATION_OFFSET, 0)
    return tokens, prefix_offset, read_offset


class IncrementalDetokenizer:
    def __init__(
        self,
        *,
        prompt_token_ids: list[int],
        stop: tuple[str, ...],
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> None:
        self._tokenizer = tokenizer
        self._stop = stop
        self._stop_buffer_len = (max(len(s) for s in stop) - 1) if stop else 0
        self._last_offset = 0
        self._output_text = ""
        self._output_token_ids: list[int] = []

        if isinstance(tokenizer, transformers.TokenizersBackend):
            self._fast_backend = tokenizer.backend_tokenizer
            self._stream: tokenizers.decoders.DecodeStream | None = tokenizers.decoders.DecodeStream(
                ids=prompt_token_ids,
                skip_special_tokens=True,
            )
            self._slow_tokens: list[str] = []
            self._prefix_offset = 0
            self._read_offset = 0
        else:
            self._fast_backend = None
            self._stream = None
            self._slow_tokens, self._prefix_offset, self._read_offset = _ConvertPromptIdsToTokens(
                prompt_ids=prompt_token_ids,
                skip_special_tokens=True,
                tokenizer=tokenizer,
            )

    @property
    def output_token_ids(self) -> list[int]:
        return self._output_token_ids

    def GetNextOutputText(self, *, delta: bool, finished: bool) -> str:
        buffer_len = 0 if finished else self._stop_buffer_len
        visible_len = max(len(self._output_text) - buffer_len, 0)
        if not delta:
            return self._output_text[:visible_len]
        if self._last_offset < visible_len:
            chunk = self._output_text[self._last_offset : visible_len]
            self._last_offset = visible_len
            return chunk
        return ""

    def Update(self, *, new_token_ids: list[int], stop_terminated: bool) -> tuple[bool, int]:
        """Return (matched_stop_string, num_tokens_kept_from_step)."""
        if not new_token_ids:
            return False, 0

        if stop_terminated:
            skipped_eos_id: int | None = new_token_ids[-1]
            decode_ids = new_token_ids[:-1]
        else:
            skipped_eos_id = None
            decode_ids = new_token_ids

        first_new_idx = len(self._output_token_ids)
        prev_text_len = len(self._output_text)
        token_offsets: list[int] | None = [prev_text_len] if self._stop else None

        for tid in decode_ids:
            self._output_token_ids.append(tid)
            self._output_text += self._DecodeNext(token_id=tid)
            if token_offsets is not None:
                token_offsets.append(len(self._output_text))

        if skipped_eos_id is not None:
            self._output_token_ids.append(skipped_eos_id)
            if token_offsets is not None:
                token_offsets.append(len(self._output_text))

        if token_offsets is not None:
            match = CheckStopStrings(
                new_char_count=len(self._output_text) - prev_text_len,
                output_text=self._output_text,
                stop=self._stop,
            )
            if match is not None:
                stop_str, truncate_to = match
                stop_end = truncate_to + len(stop_str)
                num_kept = bisect.bisect_left(token_offsets, stop_end, lo=1)
                del self._output_token_ids[first_new_idx + num_kept :]
                self._output_text = self._output_text[:truncate_to]
                return True, num_kept

        return False, len(new_token_ids)

    def _DecodeNext(self, *, token_id: int) -> str:
        if self._stream is not None:
            try:
                piece = self._stream.step(self._fast_backend, token_id)
            except Exception as exc:
                if not str(exc).startswith(INVALID_PREFIX_ERR_MSG):
                    raise
                self._stream = tokenizers.decoders.DecodeStream(skip_special_tokens=True)
                piece = self._stream.step(self._fast_backend, token_id)
            return "" if piece is None else piece

        # vLLM's slow incremental decoder is based on Hugging Face Text Generation Inference:
        # https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py
        # Adapted here under Apache-2.0; see THIRD_PARTY_NOTICES.md.
        raw_new = self._tokenizer.convert_ids_to_tokens([token_id], skip_special_tokens=True)
        new_toks = ["" if tok is None else tok for tok in raw_new]
        window_start = min(self._prefix_offset, self._read_offset, len(self._slow_tokens))
        window = self._slow_tokens[window_start:] + new_toks
        prefix_text = self._tokenizer.convert_tokens_to_string(
            window[self._prefix_offset - window_start : self._read_offset - window_start]
        )
        full_text = self._tokenizer.convert_tokens_to_string(window[self._prefix_offset - window_start :])
        self._slow_tokens.extend(new_toks)
        if len(full_text) <= len(prefix_text) or full_text.endswith(REPLACEMENT_CHAR):
            return ""
        new_piece = full_text[len(prefix_text) :]
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._slow_tokens)
        return new_piece
