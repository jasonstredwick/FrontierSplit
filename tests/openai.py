import httpx
import json
import typing

import tests.backend
import tests.helpers

from frontiersplit.openai_api import generation as fs_generation
from frontiersplit.openai_api import server as fs_server


def BuildHarness(
    *,
    api_key: str | None = None,
    chat_template: str | dict[str, str] | None = tests.helpers.DEFAULT_CHAT_TEMPLATE,
    default_tokens: list[int] | None = None,
    max_model_len: int = 32,
    model_id: str = "test-model",
) -> tuple[tests.helpers.TokenizerBundle, tests.backend.ScriptedBackend, httpx.ASGITransport]:
    bundle = tests.helpers.BuildTestTokenizer(chat_template=chat_template)
    tokens = [bundle.world_id, bundle.eos_id] if default_tokens is None else default_tokens
    core = tests.backend.ScriptedBackend(
        default_tokens=tokens,
        eos_token_ids=frozenset({bundle.eos_id}),
        max_model_len=max_model_len,
        model_id=model_id,
        vocab_size=bundle.vocab_size,
    )
    engine = fs_generation.GenerationAdapter(backend=core, tokenizer=bundle.tokenizer)
    app = fs_server.BuildApp(api_key=api_key, engine=engine)
    return bundle, core, httpx.ASGITransport(app=app)


def ParseSsePayloads(*, body: str) -> tuple[list[dict[str, typing.Any]], bool]:
    lines = [line for line in body.split("\n") if line]
    saw_done = bool(lines) and lines[-1] == "data: [DONE]"
    data_lines = lines[:-1] if saw_done else lines
    chunks: list[dict[str, typing.Any]] = []
    for line in data_lines:
        assert line.startswith("data: ")
        chunks.append(json.loads(line.removeprefix("data: ")))
    return chunks, saw_done
