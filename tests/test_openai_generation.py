import asyncio
import pytest
import re
import transformers

import tests.backend
import tests.helpers

from frontiersplit.backend import types as fs_backend_types
from frontiersplit.openai_api import detokenizer as fs_detokenizer
from frontiersplit.openai_api import generation as fs_generation
from frontiersplit.openai_api import types as fs_types


class SlowPythonTokenizer(transformers.PreTrainedTokenizer):
    """Minimal slow tokenizer without Rust backend_tokenizer to exercise prefix-offset detokenization."""

    def __init__(self) -> None:
        self._id_to_tok = {
            0: "<unk>",
            1: "<s>",
            2: "</s>",
            3: "Hello",
            4: " ,",
            5: " world",
            6: " !",
            7: "<0xE2>",
            8: "<0x82>",
            9: "<0xAC>",
        }
        self._tok_to_id = {v: k for k, v in self._id_to_tok.items()}
        super().__init__(bos_token="<s>", eos_token="</s>", unk_token="<unk>")

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_tok)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raw_bytes = bytearray()
        pieces: list[str] = []
        byte_map = {"<0x82>": 0x82, "<0xAC>": 0xAC, "<0xE2>": 0xE2}
        for tok in tokens:
            if tok in byte_map:
                raw_bytes.append(byte_map[tok])
            else:
                if raw_bytes:
                    pieces.append(raw_bytes.decode("utf-8", errors="replace"))
                    raw_bytes.clear()
                pieces.append(tok)
        if raw_bytes:
            pieces.append(raw_bytes.decode("utf-8", errors="replace"))
        joined = "".join(pieces)
        return re.sub(r"\s+([,!?])", r"\1", joined)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._tok_to_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_tok.get(index, "<unk>")

    def _convert_token_to_id(self, token: str) -> int:
        return self._tok_to_id.get(token, 0)

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        return [text] if text in self._tok_to_id else ["<unk>"]


@pytest.mark.parametrize(
    "ignore_eos,max_tokens,stop,raw_cause,finish_reason,token_count",
    [
        (False, 4, (), "eos", "stop", 3),
        (True, 4, (), "budget", "length", 4),
        (False, 0, (), "budget", "length", 0),
        (False, 4, (" END",), "eos", "stop", 2),
    ],
)
def test_backend_causes_and_text_stops(
    *,
    ignore_eos: bool,
    max_tokens: int,
    stop: tuple[str, ...],
    raw_cause: fs_backend_types.FinishCause,
    finish_reason: fs_types.FinishReason,
    token_count: int,
) -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    backend = tests.backend.ScriptedBackend(
        default_tokens=[bundle.world_id, bundle.stop_word_id, bundle.eos_id, bundle.hello_id],
        eos_token_ids=frozenset({bundle.eos_id}),
        vocab_size=bundle.vocab_size,
    )
    sampling = fs_backend_types.SamplingParams(
        ignore_eos=ignore_eos,
        logprobs=1,
        max_tokens=max_tokens,
        prompt_logprobs=1,
        seed=7,
        temperature=0.0,
        top_p=1.0,
    )
    prompt_ids = [bundle.bos_id, bundle.hello_id]
    adapter = fs_generation.GenerationAdapter(backend=backend, tokenizer=bundle.tokenizer)

    async def Run() -> None:
        # The same backend can be called directly without the compatibility processor.
        raw_steps = [
            step async for step in backend.Submit(
                request=fs_backend_types.GenerationRequest(
                    prompt_token_ids=prompt_ids,
                    request_id="direct",
                    sampling_params=sampling,
                )
            )
        ]
        assert raw_steps[-1].finish_cause == raw_cause
        outputs = [
            output async for output in adapter.Generate(
                delta=False,
                params=fs_types.GenerationParams(sampling=sampling, stop=stop),
                prompt_text="Hello",
                prompt_token_ids=prompt_ids,
                request_id="adapted",
            )
        ]
        assert len(outputs) == 1
        completion = outputs[0].outputs[0]
        assert completion.finish_reason == finish_reason
        assert len(completion.token_ids) == token_count
        assert len(completion.logprobs) == token_count
        assert len(outputs[0].prompt_logprobs) == len(prompt_ids)
        assert backend.submitted_requests[-1].sampling_params == sampling
        if stop:
            assert completion.text == " world"
            assert backend.aborted_request_ids == ["adapted"]
        elif max_tokens == 0:
            assert completion.text == ""
            assert backend.aborted_request_ids == []
        else:
            assert "</s>" not in completion.text
            assert backend.aborted_request_ids == []

    asyncio.run(Run())


def test_metaspace_leading_space_restoration() -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    # Single-token HuggingFace decode strips the Metaspace leading space, while ConvertIdsToTokens restores it.
    assert bundle.tokenizer.decode([bundle.world_id]) == "world"
    restored = fs_detokenizer.ConvertIdsToTokens(
        token_ids=[bundle.hello_id, bundle.world_id],
        tokenizer=bundle.tokenizer,
    )
    assert restored == [" Hello", " world"]


def test_fast_detokenizer_multibyte_utf8_and_stop_holdback() -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    detokenizer = fs_detokenizer.IncrementalDetokenizer(
        prompt_token_ids=[bundle.bos_id, bundle.hello_id],
        stop=(" END",),
        tokenizer=bundle.tokenizer,
    )

    # Stream the 3 UTF-8 bytes of "€" one token at a time; partial bytes must never emit U+FFFD.
    b0, b1, b2 = bundle.euro_byte_ids
    matched, kept = detokenizer.Update(new_token_ids=[b0], stop_terminated=False)
    assert (matched, kept) == (False, 1)
    assert detokenizer.GetNextOutputText(delta=True, finished=False) == ""

    matched, kept = detokenizer.Update(new_token_ids=[b1], stop_terminated=False)
    assert (matched, kept) == (False, 1)
    assert detokenizer.GetNextOutputText(delta=True, finished=False) == ""

    matched, kept = detokenizer.Update(new_token_ids=[b2, bundle.world_id], stop_terminated=False)
    assert (matched, kept) == (False, 2)
    # "€ world" is 7 chars; stop=" END" holds back 3 chars, so 4 chars ("€ wo") release immediately.
    chunk_1 = detokenizer.GetNextOutputText(delta=True, finished=False)
    assert "\ufffd" not in chunk_1
    assert chunk_1 == "€ wo"

    # Now emit stop_word_id (" END") followed by extra trailing world_id in the same chunk.
    matched, kept = detokenizer.Update(
        new_token_ids=[bundle.stop_word_id, bundle.world_id],
        stop_terminated=False,
    )
    assert matched is True
    # Only the stop-completing token is kept; the trailing world_id after " END" is truncated.
    assert kept == 1
    chunk_2 = detokenizer.GetNextOutputText(delta=True, finished=True)
    assert chunk_1 + chunk_2 == "€ world"
    assert detokenizer.output_token_ids == [b0, b1, b2, bundle.world_id, bundle.stop_word_id]


def test_slow_fallback_detokenizer_punctuation_and_utf8() -> None:
    slow_tok = SlowPythonTokenizer()
    assert not isinstance(slow_tok, transformers.TokenizersBackend)

    detok = fs_detokenizer.IncrementalDetokenizer(
        prompt_token_ids=[1],
        stop=(),
        tokenizer=slow_tok,
    )
    pieces: list[str] = []
    for tid in [3, 4, 5, 6, 7, 8, 9]:
        detok.Update(new_token_ids=[tid], stop_terminated=False)
        pieces.append(detok.GetNextOutputText(delta=True, finished=False))
    pieces.append(detok.GetNextOutputText(delta=True, finished=True))

    streamed = "".join(pieces)
    assert "\ufffd" not in streamed
    assert streamed == "Hello, world!€"


@pytest.mark.parametrize("step_chunk_size", [1, 2, 5])
def test_generation_adapter_utf8_logprob_repair_and_stop_abort(*, step_chunk_size: int) -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    b0, b1, b2 = bundle.euro_byte_ids
    core = tests.backend.ScriptedBackend(
        default_tokens=[b0, b1, b2, bundle.stop_word_id, bundle.world_id],
        eos_token_ids=frozenset({bundle.eos_id}),
        vocab_size=bundle.vocab_size,
    )
    core.step_chunk_size = step_chunk_size
    engine = fs_generation.GenerationAdapter(backend=core, tokenizer=bundle.tokenizer)

    async def Run() -> fs_types.RequestOutput:
        outputs: list[fs_types.RequestOutput] = []
        async for out in engine.Generate(
            delta=False,
            prompt_text="Hello",
            prompt_token_ids=[bundle.bos_id, bundle.hello_id],
            request_id="req-stop-1",
            params=fs_types.GenerationParams(
                sampling=fs_backend_types.SamplingParams(
                    ignore_eos=False,
                    logprobs=1,
                    max_tokens=5,
                    prompt_logprobs=None,
                    seed=None,
                    temperature=0.0,
                    top_p=1.0,
                ),
                stop=(" END",),
            ),
        ):
            outputs.append(out)
        assert len(outputs) == 1
        return outputs[0]

    final_out = asyncio.run(Run())
    comp = final_out.outputs[0]
    assert comp.finish_reason == "stop"
    assert comp.text == "€"
    assert comp.token_ids == [b0, b1, b2, bundle.stop_word_id]
    # A terminal five-token step needs trimming but no abort; smaller steps still have pending work.
    expected_aborts = [] if step_chunk_size == 5 else ["req-stop-1"]
    assert core.aborted_request_ids == expected_aborts
    assert comp.logprobs is not None
    assert len(comp.logprobs) == len(comp.token_ids)
    # The final byte token <0xAC> completes "€" given context [b0, b1] rather than leaving U+FFFD.
    assert comp.logprobs[2][b2].decoded_token == "€"


def test_chunked_prefill_prompt_logprobs_utf8_across_boundary() -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    b0, b1, b2 = bundle.euro_byte_ids
    core = tests.backend.ScriptedBackend(
        default_tokens=[bundle.world_id],
        eos_token_ids=frozenset({bundle.eos_id}),
        vocab_size=bundle.vocab_size,
    )
    # Chunk size 3 splits [bos, hello, b0 | b1, b2] across two prefill chunks.
    core.prefill_chunk_size = 3
    engine = fs_generation.GenerationAdapter(backend=core, tokenizer=bundle.tokenizer)
    prompt_ids = [bundle.bos_id, bundle.hello_id, b0, b1, b2]

    async def RunDelta() -> list[fs_types.RequestOutput]:
        steps: list[fs_types.RequestOutput] = []
        async for step in engine.Generate(
            delta=True,
            prompt_text=None,
            prompt_token_ids=prompt_ids,
            request_id="req-chunked-delta",
            params=fs_types.GenerationParams(
                sampling=fs_backend_types.SamplingParams(
                    ignore_eos=False,
                    logprobs=1,
                    max_tokens=2,
                    prompt_logprobs=1,
                    seed=None,
                    temperature=0.0,
                    top_p=1.0,
                ),
            ),
        ):
            steps.append(step)
        return steps

    delta_steps = asyncio.run(RunDelta())
    # Intermediate prefill-only step is suppressed; prompt_logprobs arrive intact once prefill completes.
    assert len(delta_steps) == 2
    assert delta_steps[0].prompt_logprobs is not None
    assert len(delta_steps[0].prompt_logprobs) == 5
    assert delta_steps[0].prompt_logprobs[0] is None
    # b2 in chunk 2 reconstructs "€" using b0 from chunk 1 and b1 from chunk 2.
    assert delta_steps[0].prompt_logprobs[2] is not None
    assert delta_steps[0].prompt_logprobs[2][b0].decoded_token == ""
    assert delta_steps[0].prompt_logprobs[3] is not None
    assert delta_steps[0].prompt_logprobs[3][b1].decoded_token == ""
    assert delta_steps[0].prompt_logprobs[4] is not None
    assert delta_steps[0].prompt_logprobs[4][b2].decoded_token == "€"
    assert delta_steps[1].prompt_logprobs is None
