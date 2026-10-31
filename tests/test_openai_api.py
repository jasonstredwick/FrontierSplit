import asyncio
import httpx
import importlib.metadata
import packaging.requirements
import pathlib
import pytest
import threading
import tomllib
import transformers
import typing

import tests.backend
import tests.helpers
import tests.openai

from frontiersplit.backend import types as fs_backend_types
from frontiersplit.openai_api import generation as fs_generation
from frontiersplit.openai_api import renderer as fs_renderer
from frontiersplit.openai_api import server as fs_server


class GatedFastTokenizer(transformers.PreTrainedTokenizerFast):
    """Fast tokenizer whose encode() blocks on a threading.Event to test off-loop rendering."""

    def __init__(
        self,
        *,
        gate: threading.Event,
        started: threading.Event,
        **kwargs: typing.Any,
    ) -> None:
        self._enabled = False
        self._gate = gate
        self._started = started
        super().__init__(**kwargs)
        self._enabled = True

    def encode(self, *args: typing.Any, **kwargs: typing.Any) -> list[int]:
        if self._enabled:
            self._started.set()
            if not self._gate.wait(timeout=5.0):
                raise TimeoutError("Timed out waiting for tokenizer gate.")
        return list(super().encode(*args, **kwargs))


def test_completion_resolves_controls_before_backend_submission() -> None:
    bundle, backend, transport = tests.openai.BuildHarness(max_model_len=32)
    prompt_ids = [bundle.hello_id] * 30

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "echo": True,
                    "ignore_eos": True,
                    "logprobs": 1,
                    "max_tokens": None,
                    "model": "test-model",
                    "prompt": prompt_ids,
                    "seed": 7,
                    "stop": " END",
                    "temperature": 0.25,
                    "top_p": 0.5,
                },
            )
        assert response.status_code == 200
        submitted = backend.submitted_requests[-1]
        assert submitted.prompt_token_ids == prompt_ids
        assert submitted.sampling_params == fs_backend_types.SamplingParams(
            ignore_eos=True,
            logprobs=1,
            max_tokens=2,
            prompt_logprobs=1,
            seed=7,
            temperature=0.25,
            top_p=0.5,
        )

    asyncio.run(Run())


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "controls,expected_temperature,expected_seed",
    [
        ({"seed": -1, "temperature": 0.001}, 0.01, None),
        ({"seed": 7, "temperature": 1e-6}, 0.01, 7),
        ({"seed": -2, "temperature": 0.0}, 0.0, -2),
        ({"seed": 0, "temperature": 0.01}, 0.01, 0),
        ({"seed": 1234, "temperature": 0.25}, 0.25, 1234),
        ({"seed": None, "temperature": None}, 1.0, None),
        ({}, 1.0, None),
    ],
)
def test_sampling_normalization_at_http_boundary(
    *,
    controls: dict[str, float | int | None],
    endpoint: str,
    expected_seed: int | None,
    expected_temperature: float,
    stream: bool,
) -> None:
    _, backend, transport = tests.openai.BuildHarness(max_model_len=64)
    payload: dict[str, typing.Any] = {
        "max_tokens": 1,
        "model": "test-model",
        "stream": stream,
        **controls,
    }
    if endpoint == "/v1/completions":
        payload["prompt"] = "Hello"
    else:
        payload["messages"] = [{"content": "Hello", "role": "user"}]

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(endpoint, json=payload)
        assert response.status_code == 200
        if stream:
            chunks, saw_done = tests.openai.ParseSsePayloads(body=response.text)
            assert saw_done
            assert all("error" not in chunk for chunk in chunks)
            assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        else:
            assert response.json()["choices"][0]["finish_reason"] == "length"
        assert len(backend.submitted_requests) == 1
        sampling = backend.submitted_requests[0].sampling_params
        assert sampling.seed == expected_seed
        assert sampling.temperature == expected_temperature

    asyncio.run(Run())


def test_parse_server_config_cli_and_env() -> None:
    default_cfg = fs_server.ParseServerConfig(argv=[], env={})
    assert default_cfg.host == "127.0.0.1"
    assert default_cfg.port == 8000
    assert default_cfg.api_key is None

    env_cfg = fs_server.ParseServerConfig(
        argv=["--port", "9001"],
        env={fs_server.API_KEY_ENV_VAR: "env-secret"},
    )
    assert env_cfg.port == 9001
    assert env_cfg.api_key == "env-secret"

    cli_cfg = fs_server.ParseServerConfig(
        argv=["--api-key", "cli-secret", "--host", "0.0.0.0"],
        env={fs_server.API_KEY_ENV_VAR: "env-secret"},
    )
    assert cli_cfg.host == "0.0.0.0"
    assert cli_cfg.api_key == "cli-secret"


def test_health_readiness_transitions_and_unauthenticated_probe() -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    core = tests.backend.ScriptedBackend(
        default_tokens=[bundle.world_id],
        eos_token_ids=frozenset({bundle.eos_id}),
        vocab_size=bundle.vocab_size,
    )
    engine = fs_generation.GenerationAdapter(backend=core, tokenizer=bundle.tokenizer)
    app = fs_server.BuildApp(api_key="top-secret", engine=None)

    async def Run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            # Unattached backend -> 503 (unauthenticated probe allowed even with api_key enabled).
            r_unattached = await client.get("/health")
            assert r_unattached.status_code == 503

            # Unattached /v1/models with valid auth -> 503 service_unavailable JSON envelope.
            r_models_503 = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer top-secret"},
            )
            assert r_models_503.status_code == 503
            assert r_models_503.json()["error"]["type"] == "service_unavailable"

            # Attach healthy engine -> 200 without auth header.
            fs_server.SetEngine(app=app, engine=engine)
            r_healthy = await client.get("/health")
            assert r_healthy.status_code == 200

            # Unhealthy engine -> 503.
            core.healthy = False
            r_unhealthy = await client.get("/health")
            assert r_unhealthy.status_code == 503

    asyncio.run(Run())


def test_bearer_authentication_and_model_discovery() -> None:
    _, _, transport = tests.openai.BuildHarness(api_key="secret-token", max_model_len=32768, model_id="mistral-7b")

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for bad_headers in [
                {},
                {"Authorization": "Basic secret-token"},
                {"Authorization": "Bearer wrong-token"},
                {"Authorization": "Bearer "},
            ]:
                resp = await client.get("/v1/models", headers=bad_headers)
                assert resp.status_code == 401
                err = resp.json()["error"]
                assert err["type"] == "authentication_error"
                assert err["code"] is None

            auth = {"Authorization": "Bearer secret-token"}
            ok = await client.get("/v1/models", headers=auth)
            assert ok.status_code == 200
            body = ok.json()
            assert body == {
                "data": [
                    {
                        "created": 0,
                        "id": "mistral-7b",
                        "max_model_len": 32768,
                        "object": "model",
                        "owned_by": "frontiersplit",
                        "parent": None,
                        "root": "mistral-7b",
                    }
                ],
                "object": "list",
            }

            not_found_route = await client.get("/v1/embeddings", headers=auth)
            assert not_found_route.status_code == 404
            assert not_found_route.json()["error"]["type"] == "not_found_error"

            not_found_model = await client.post(
                "/v1/completions",
                headers=auth,
                json={"model": "unknown-model", "prompt": "Hello"},
            )
            assert not_found_model.status_code == 404
            assert not_found_model.json()["error"]["type"] == "not_found_error"
            assert not_found_model.json()["error"]["param"] == "model"

    asyncio.run(Run())


def test_strict_json_types_and_unsupported_fields() -> None:
    _, _, transport = tests.openai.BuildHarness()

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Wrong Content-Type rejected.
            r_ct = await client.post(
                "/v1/completions",
                content=b'{"model":"test-model","prompt":"Hello"}',
                headers={"Content-Type": "text/plain"},
            )
            assert r_ct.status_code == 400
            assert r_ct.json()["error"]["type"] == "invalid_request_error"

            # Non-standard JSON numeric literals (NaN / Infinity / -Infinity), even inside metadata.
            for raw in [
                b'{"model":"test-model","prompt":"Hello","temperature":NaN}',
                b'{"model":"test-model","prompt":"Hello","temperature":Infinity}',
                b'{"model":"test-model","prompt":"Hello","metadata":{"score":-Infinity}}',
                b'[{"model":"test-model","prompt":"Hello"}]',
                b'{"model":"test-model","prompt":',
            ]:
                resp = await client.post(
                    "/v1/completions",
                    content=raw,
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status_code == 400
                assert resp.json()["error"]["type"] == "invalid_request_error"

            # Strict type & capability rejection matrix.
            invalid_payloads: list[dict[str, object]] = [
                {"model": "test-model", "prompt": "Hello", "max_tokens": True},
                {"model": "test-model", "prompt": "Hello", "max_tokens": 2.5},
                {"model": "test-model", "prompt": "Hello", "temperature": False},
                {"model": "test-model", "prompt": [True, False]},
                {"model": "test-model", "prompt": "Hello", "stream": None},
                {"model": "test-model", "prompt": "Hello", "echo": None},
                {"model": "test-model", "prompt": "Hello", "ignore_eos": None},
                {"model": "test-model", "prompt": "Hello", "n": 2},
                {"model": "test-model", "prompt": "Hello", "best_of": 2},
                {"model": "test-model", "prompt": "Hello", "frequency_penalty": 0.1},
                {"model": "test-model", "prompt": "Hello", "presence_penalty": -0.1},
                {"model": "test-model", "prompt": "Hello", "repetition_penalty": 1.2},
                {"model": "test-model", "prompt": "Hello", "suffix": "tail"},
                {"model": "test-model", "prompt": "Hello", "logit_bias": {"1": 1.0}},
                {"model": "test-model", "prompt": "Hello", "top_k": 50},
                {"model": "test-model", "prompt": "Hello", "temperature": -0.01},
                {"model": "test-model", "prompt": "Hello", "temperature": 2.01},
                {"model": "test-model", "prompt": "Hello", "top_p": 0.0},
                {"model": "test-model", "prompt": "Hello", "top_p": 1.01},
                {"model": "test-model", "prompt": "Hello", "seed": 1 << 63},
                {"model": "test-model", "prompt": "Hello", "seed": -(1 << 63) - 1},
                {"model": "test-model", "prompt": "Hello", "stop": ""},
                {"model": "test-model", "prompt": "Hello", "stop": ["ok", ""]},
                {"model": "test-model", "prompt": "Hello", "stop": [None]},
                {"model": "test-model", "prompt": "Hello", "logprobs": -1},
                {"model": "test-model", "prompt": "Hello", "logprobs": 21},
                {"model": "test-model", "prompt": "Hello", "stream_options": {"include_usage": True}},
            ]
            for payload in invalid_payloads:
                resp = await client.post("/v1/completions", json=payload)
                assert resp.status_code == 400, f"Expected 400 for {payload}, got {resp.status_code}"
                assert resp.json()["error"]["type"] == "invalid_request_error"

            # Valid compatibility / neutral fields succeed.
            valid_resp = await client.post(
                "/v1/completions",
                json={
                    "best_of": 1,
                    "frequency_penalty": 0.0,
                    "logit_bias": {},
                    "max_tokens": None,
                    "metadata": {"experiment": "alpha"},
                    "model": "test-model",
                    "n": 1,
                    "presence_penalty": 0.0,
                    "prompt": "Hello",
                    "repetition_penalty": 1.0,
                    "seed": -(1 << 63),
                    "stop": [],
                    "suffix": None,
                    "temperature": None,
                    "top_p": None,
                    "user": "eval-runner",
                },
            )
            assert valid_resp.status_code == 200

    asyncio.run(Run())


def test_completions_bos_token_ids_and_budget_bounds() -> None:
    bundle, core, transport = tests.openai.BuildHarness(max_model_len=24)

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. String prompt applies add_special_tokens=True by default (inserts BOS).
            r_bos = await client.post(
                "/v1/completions",
                json={"max_tokens": 2, "model": "test-model", "prompt": "Hello"},
            )
            assert r_bos.status_code == 200
            assert core.submitted_requests[-1].prompt_token_ids == [bundle.bos_id, bundle.hello_id]

            # 2. String prompt with add_special_tokens=False omits BOS.
            r_no_bos = await client.post(
                "/v1/completions",
                json={
                    "add_special_tokens": False,
                    "max_tokens": 2,
                    "model": "test-model",
                    "prompt": "Hello",
                },
            )
            assert r_no_bos.status_code == 200
            assert core.submitted_requests[-1].prompt_token_ids == [bundle.hello_id]

            # 3. Pre-tokenized integer prompt executes verbatim regardless of add_special_tokens=True,
            # and null max_tokens resolves to 16.
            r_ids = await client.post(
                "/v1/completions",
                json={
                    "add_special_tokens": True,
                    "max_tokens": None,
                    "model": "test-model",
                    "prompt": [bundle.hello_id, bundle.world_id],
                },
            )
            assert r_ids.status_code == 200
            last_req = core.submitted_requests[-1]
            assert last_req.prompt_token_ids == [bundle.hello_id, bundle.world_id]
            assert last_req.sampling_params.max_tokens == 16

            # 4. Requested budget > max_model_len rejected with HTTP 400.
            r_over_budget = await client.post(
                "/v1/completions",
                json={"max_tokens": 25, "model": "test-model", "prompt": [bundle.hello_id]},
            )
            assert r_over_budget.status_code == 400
            assert r_over_budget.json()["error"]["param"] == "max_tokens"

    asyncio.run(Run())


def test_completions_context_limits_and_batches() -> None:
    bundle, core, transport = tests.openai.BuildHarness(max_model_len=20)
    core.SetDefaultTokens(tokens=[bundle.world_id])

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Prompt of 12 tokens leaves remaining=8; omitted max_tokens (default 16) caps to 8.
            twelve_ids = [bundle.hello_id] * 12
            r_capped = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": twelve_ids},
            )
            assert r_capped.status_code == 200
            assert core.submitted_requests[-1].sampling_params.max_tokens == 8
            assert r_capped.json()["choices"][0]["finish_reason"] == "length"
            assert r_capped.json()["usage"] == {
                "completion_tokens": 8,
                "prompt_tokens": 12,
                "total_tokens": 20,
            }

            # Full-window prompt (20 tokens, remaining=0) fails for generative request, succeeds for max_tokens=0.
            full_ids = [bundle.hello_id] * 20
            r_full_gen = await client.post(
                "/v1/completions",
                json={"max_tokens": 1, "model": "test-model", "prompt": full_ids},
            )
            assert r_full_gen.status_code == 400

            r_full_prefill = await client.post(
                "/v1/completions",
                json={"echo": True, "max_tokens": 0, "model": "test-model", "prompt": full_ids},
            )
            assert r_full_prefill.status_code == 200
            assert r_full_prefill.json()["choices"][0]["finish_reason"] == "length"
            assert r_full_prefill.json()["usage"]["completion_tokens"] == 0

            # Overflow prompt (21 tokens > max_model_len=20) rejected even with max_tokens=0.
            r_overflow = await client.post(
                "/v1/completions",
                json={"max_tokens": 0, "model": "test-model", "prompt": [bundle.hello_id] * 21},
            )
            assert r_overflow.status_code == 400

            # Out-of-vocab token IDs rejected.
            for bad_id in [-1, bundle.vocab_size]:
                r_vocab = await client.post(
                    "/v1/completions",
                    json={"max_tokens": 1, "model": "test-model", "prompt": [bad_id]},
                )
                assert r_vocab.status_code == 400

            # Ordered batch completion and whole-batch rejection on single invalid item.
            prev_count = len(core.submitted_requests)
            r_bad_batch = await client.post(
                "/v1/completions",
                json={
                    "max_tokens": 2,
                    "model": "test-model",
                    "prompt": [[bundle.hello_id], [bundle.hello_id] * 21],
                },
            )
            assert r_bad_batch.status_code == 400
            assert len(core.submitted_requests) == prev_count

            r_batch = await client.post(
                "/v1/completions",
                json={
                    "echo": True,
                    "max_tokens": 1,
                    "model": "test-model",
                    "prompt": ["Hello", "Hello world"],
                },
            )
            assert r_batch.status_code == 200
            batch_data = r_batch.json()
            assert [c["index"] for c in batch_data["choices"]] == [0, 1]
            assert batch_data["choices"][0]["text"].startswith("Hello")
            assert batch_data["choices"][1]["text"].startswith("Hello world")
            # "Hello" -> 2 prompt tokens (BOS + Hello), "Hello world" -> 3 prompt tokens (BOS + Hello + world).
            assert batch_data["usage"] == {
                "completion_tokens": 2,
                "prompt_tokens": 5,
                "total_tokens": 7,
            }

    asyncio.run(Run())


def test_completions_logprobs_echo_clamping_collisions_and_eos() -> None:
    bundle, core, transport = tests.openai.BuildHarness(max_model_len=32)
    # Give alt_world_id (which also decodes to " world") a HIGHER logit than world_id,
    # and set token 0 to -inf to verify clamping at -9999.0.
    core.logit_overrides = {
        0: float("-inf"),
        bundle.alt_world_id: 8.0,
        bundle.world_id: 4.0,
    }

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. lm-eval likelihood scoring path: token-ID prompt, echo=True, logprobs=1, max_tokens=1.
            resp = await client.post(
                "/v1/completions",
                json={
                    "echo": True,
                    "logprobs": 1,
                    "max_tokens": 1,
                    "model": "test-model",
                    "prompt": [bundle.bos_id, bundle.hello_id],
                    "temperature": 0.0,
                },
            )
            assert resp.status_code == 200
            choice = resp.json()["choices"][0]
            lp = choice["logprobs"]
            # 2 prompt tokens + 1 generated token (world_id) = 3 positions.
            assert len(lp["tokens"]) == 3
            assert len(lp["token_logprobs"]) == 3
            assert len(lp["top_logprobs"]) == 3
            assert len(lp["text_offset"]) == 3
            assert lp["token_logprobs"][0] is None
            assert lp["top_logprobs"][0] is None
            # Leading space preserved on SentencePiece tokens.
            assert lp["tokens"][1] == " Hello"
            assert lp["tokens"][2] == " world"
            # Colliding candidate alt_world_id (" world", logit 8.0) has higher score than selected world_id (4.0):
            # token_logprobs[2] records actual selected world_id score, while top_logprobs[2][" world"] retains max!
            assert lp["top_logprobs"][2][" world"] > lp["token_logprobs"][2]

            # 2. Verify -inf logit clamps to -9999.0 and serializes as finite JSON number.
            core.SetDefaultTokens(tokens=[0])
            r_clamp = await client.post(
                "/v1/completions",
                json={
                    "logprobs": 0,
                    "max_tokens": 1,
                    "model": "test-model",
                    "prompt": [bundle.hello_id],
                },
            )
            assert r_clamp.status_code == 200
            clamp_lp = r_clamp.json()["choices"][0]["logprobs"]
            assert clamp_lp["token_logprobs"] == [-9999.0]
            assert clamp_lp["top_logprobs"] == [{"<unk>": -9999.0}]

            # 3. EOS accounting vs ignore_eos=True.
            core.SetDefaultTokens(tokens=[bundle.world_id, bundle.eos_id, bundle.hello_id])
            r_eos = await client.post(
                "/v1/completions",
                json={"max_tokens": 3, "model": "test-model", "prompt": [bundle.hello_id]},
            )
            assert r_eos.status_code == 200
            d_eos = r_eos.json()
            assert d_eos["choices"][0]["finish_reason"] == "stop"
            assert d_eos["choices"][0]["text"] == " world"
            # EOS token is counted in completion_tokens (world_id + eos_id = 2) and omitted from text.
            assert d_eos["usage"]["completion_tokens"] == 2

            r_ignore_eos = await client.post(
                "/v1/completions",
                json={
                    "ignore_eos": True,
                    "max_tokens": 3,
                    "model": "test-model",
                    "prompt": [bundle.hello_id],
                },
            )
            assert r_ignore_eos.status_code == 200
            d_ign = r_ignore_eos.json()
            assert d_ign["choices"][0]["finish_reason"] == "length"
            assert d_ign["choices"][0]["text"] == " world Hello"
            assert d_ign["usage"]["completion_tokens"] == 3

    asyncio.run(Run())


def test_chat_template_ast_analysis_and_dict_templates() -> None:
    # 1. Sliced / aliased loop_messages + {% generation %} + {% continue %} -> structured ("openai").
    sliced_structured_tpl = (
        "{%- set loop_messages = messages[1:] if messages[0]['role'] == 'system' else messages %}"
        "{%- for msg in loop_messages %}"
        "{%- if msg['role'] == 'system' %}{%- continue %}{%- endif %}"
        "{%- generation %}"
        "{%- for block in msg.content %}[{{ block.type }}:{{ block.text }}]{%- endfor %}"
        "{%- endgeneration %}"
        "{%- endfor %}"
    )
    assert fs_renderer.DetectTemplateContentFormat(chat_template=sliced_structured_tpl) == "openai"

    # 2. Macro forwarding message.content under a custom parameter name -> "openai".
    macro_structured_tpl = (
        "{%- macro render_blocks(items) %}"
        "{%- for entry in items %}{{ entry['text'] }}{%- endfor %}"
        "{%- endmacro %}"
        "{%- set ns = namespace(rest=messages) %}"
        "{%- for m in ns.rest %}{{ render_blocks(m['content']) }}{%- endfor %}"
    )
    assert fs_renderer.DetectTemplateContentFormat(chat_template=macro_structured_tpl) == "openai"

    # 3. Unrelated macro parameter named `content` not bound to message.content -> "string".
    unrelated_macro_tpl = (
        "{%- macro header(content) %}{%- for ch in content %}{{ ch }}{%- endfor %}{%- endmacro %}"
        "{{ header(['a', 'b']) }}"
        "{%- for m in messages %}{{ m['content'] }}{%- endfor %}"
    )
    assert fs_renderer.DetectTemplateContentFormat(chat_template=unrelated_macro_tpl) == "string"

    # 4. Multi-template dict checkpoint (`{"default": ...}`) and chunked-prefill echo+logprobs over HTTP.
    bundle, core, transport = tests.openai.BuildHarness(
        chat_template={"default": sliced_structured_tpl},
        max_model_len=64,
    )
    core.prefill_chunk_size = 2

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_dict_chat = await client.post(
                "/v1/chat/completions",
                json={
                    "max_tokens": 2,
                    "messages": [
                        {
                            "content": [
                                {"text": "Hello", "type": "text"},
                                {"text": "world", "type": "text"},
                            ],
                            "role": "user",
                        }
                    ],
                    "model": "test-model",
                },
            )
            assert r_dict_chat.status_code == 200
            rendered_ids = core.submitted_requests[-1].prompt_token_ids
            assert "[text:Hello][text:world]" in bundle.tokenizer.decode(rendered_ids)

            b0, b1, b2 = bundle.euro_byte_ids
            r_chunked_lp = await client.post(
                "/v1/completions",
                json={
                    "echo": True,
                    "logprobs": 1,
                    "max_tokens": 1,
                    "model": "test-model",
                    "prompt": [bundle.bos_id, b0, b1, b2, bundle.hello_id],
                },
            )
            assert r_chunked_lp.status_code == 200
            lp = r_chunked_lp.json()["choices"][0]["logprobs"]
            assert lp["tokens"] == ["<s>", "", "", "€", " Hello", " world"]

    asyncio.run(Run())


def test_async_rendering_does_not_block_event_loop() -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    gate = threading.Event()
    started = threading.Event()
    gated_tok = GatedFastTokenizer(
        bos_token="<s>",
        eos_token="</s>",
        gate=gate,
        started=started,
        tokenizer_object=bundle.tokenizer.backend_tokenizer,
        unk_token="<unk>",
    )
    gated_tok.chat_template = tests.helpers.DEFAULT_CHAT_TEMPLATE

    core = tests.backend.ScriptedBackend(
        default_tokens=[bundle.world_id, bundle.eos_id],
        eos_token_ids=frozenset({bundle.eos_id}),
        max_model_len=32,
        vocab_size=bundle.vocab_size,
    )
    engine = fs_generation.GenerationAdapter(backend=core, tokenizer=gated_tok)
    app = fs_server.BuildApp(engine=engine)
    transport = httpx.ASGITransport(app=app)

    async def WaitForWorkerStart() -> None:
        for _ in range(500):
            if started.is_set():
                return
            await asyncio.sleep(0.002)
        raise AssertionError("Renderer worker did not enter encode().")

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Blocked /v1/completions tokenization must not block /health, /v1/models, or SSE streams.
            cmpl_task = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={"max_tokens": 1, "model": "test-model", "prompt": "Hello"},
                )
            )
            await WaitForWorkerStart()
            assert not cmpl_task.done()

            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/v1/models")).status_code == 200
            r_sse = await client.post(
                "/v1/completions",
                json={
                    "max_tokens": 2,
                    "model": "test-model",
                    "prompt": [bundle.hello_id],
                    "stream": True,
                },
            )
            assert r_sse.status_code == 200
            _, saw_done = tests.openai.ParseSsePayloads(body=r_sse.text)
            assert saw_done is True

            gate.set()
            r_cmpl = await cmpl_task
            assert r_cmpl.status_code == 200

            # 2. Blocked /v1/chat/completions rendering + tokenization must also leave the event loop free.
            started.clear()
            gate.clear()
            chat_task = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "max_tokens": 1,
                        "messages": [{"content": "Hello", "role": "user"}],
                        "model": "test-model",
                    },
                )
            )
            await WaitForWorkerStart()
            assert not chat_task.done()

            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/v1/models")).status_code == 200

            gate.set()
            r_chat = await chat_task
            assert r_chat.status_code == 200

    asyncio.run(Run())
    fs_server.SetEngine(app=app, engine=None)


def test_failed_batch_cancels_and_aborts_sibling_generations() -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    core = tests.backend.ScriptedBackend(
        default_tokens=[bundle.world_id],
        eos_token_ids=frozenset({bundle.eos_id}),
        max_model_len=32,
        vocab_size=bundle.vocab_size,
    )
    engine = fs_generation.GenerationAdapter(backend=core, tokenizer=bundle.tokenizer)
    app = fs_server.BuildApp(engine=engine)
    transport = httpx.ASGITransport(app=app)

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Prompt 1 raises GenerationError at step 1 while Prompts 0 and 2 are held mid-generation.
            core.hold_gate = asyncio.Event()
            core.pause_after_steps = 1
            core.fail_after_steps = 1
            core.fail_prompt_token_ids = (bundle.world_id,)

            r_500 = await client.post(
                "/v1/completions",
                json={
                    "max_tokens": 4,
                    "model": "test-model",
                    "prompt": [
                        [bundle.hello_id],
                        [bundle.world_id],
                        [bundle.bos_id, bundle.hello_id],
                    ],
                },
            )
            assert r_500.status_code == 500
            assert r_500.json()["error"]["type"] == "server_error"
            assert core.active_request_ids == set()
            expected_ids = {req.request_id for req in core.submitted_requests[-3:]}
            assert len(expected_ids) == 3
            assert set(core.aborted_request_ids) == expected_ids

            # 2. BackendUnavailableError from one batch prompt preserves HTTP 503 while still aborting siblings.
            core.aborted_request_ids.clear()
            core.fail_exception = fs_backend_types.BackendUnavailableError("Worker disconnected.")
            r_503 = await client.post(
                "/v1/completions",
                json={
                    "max_tokens": 4,
                    "model": "test-model",
                    "prompt": [[bundle.hello_id], [bundle.world_id]],
                },
            )
            assert r_503.status_code == 503
            assert r_503.json()["error"]["type"] == "service_unavailable"
            assert core.active_request_ids == set()
            expected_503_ids = {req.request_id for req in core.submitted_requests[-2:]}
            assert set(core.aborted_request_ids) == expected_503_ids

    asyncio.run(Run())


def test_pyproject_dependency_floors_match_runtime_apis() -> None:
    pyproject_path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    project_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    reqs = {
        req.name: req
        for raw in project_data["dependencies"]
        if (req := packaging.requirements.Requirement(raw))
    }

    assert "transformers" in reqs
    # transformers.TokenizersBackend requires Transformers >= 5.0.0 (absent in 4.x).
    assert "4.57.6" not in reqs["transformers"].specifier
    assert "5.0.0" in reqs["transformers"].specifier

    for pkg_name in ("jinja2", "safetensors", "tokenizers", "transformers"):
        assert pkg_name in reqs
        installed_version = importlib.metadata.version(pkg_name)
        assert installed_version in reqs[pkg_name].specifier
