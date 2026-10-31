import asyncio
import fastapi
import httpx
import json
import typing

import tests.backend
import tests.helpers
import tests.openai

from frontiersplit.openai_api import generation as fs_generation
from frontiersplit.openai_api import server as fs_server


def _BuildStreamingHarness() -> tuple[
    tests.helpers.TokenizerBundle, tests.backend.ScriptedBackend, fastapi.FastAPI, httpx.ASGITransport
]:
    bundle = tests.helpers.BuildTestTokenizer()
    b0, b1, b2 = bundle.euro_byte_ids
    core = tests.backend.ScriptedBackend(
        default_tokens=[b0, b1, b2, bundle.world_id, bundle.eos_id],
        eos_token_ids=frozenset({bundle.eos_id}),
        max_model_len=32,
        vocab_size=bundle.vocab_size,
    )
    engine = fs_generation.GenerationAdapter(backend=core, tokenizer=bundle.tokenizer)
    app = fs_server.BuildApp(engine=engine)
    transport = httpx.ASGITransport(app=app)
    return bundle, core, app, transport


def test_backend_failure_before_headers() -> None:
    bundle, core, _, transport = _BuildStreamingHarness()

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            core.fail_before_first_step = True
            r_500 = await client.post(
                "/v1/completions",
                json={"max_tokens": 2, "model": "test-model", "prompt": [bundle.hello_id]},
            )
            assert r_500.status_code == 500
            assert r_500.json()["error"]["type"] == "server_error"

    asyncio.run(Run())


def test_backend_failure_during_streaming() -> None:
    _, core, _, transport = _BuildStreamingHarness()

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            core.fail_after_steps = 1
            r_mid_err = await client.post(
                "/v1/chat/completions",
                json={
                    "max_tokens": 5,
                    "messages": [{"content": "Hello", "role": "user"}],
                    "model": "test-model",
                    "stream": True,
                },
            )
            assert r_mid_err.status_code == 200
            err_chunks, err_done = tests.openai.ParseSsePayloads(body=r_mid_err.text)
            assert err_done is True
            err_obj = err_chunks[-1]["error"]
            assert err_obj["type"] == "server_error"
            assert err_obj["code"] == "stream_interrupted"

    asyncio.run(Run())


def test_chat_streaming_continuous_usage() -> None:
    _, _, _, transport = _BuildStreamingHarness()

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_chat_sse = await client.post(
                "/v1/chat/completions",
                json={
                    "max_tokens": 5,
                    "messages": [{"content": "Hello", "role": "user"}],
                    "model": "test-model",
                    "stream": True,
                    "stream_options": {"continuous_usage_stats": True, "include_usage": True},
                },
            )
            assert r_chat_sse.status_code == 200
            chat_chunks, chat_done = tests.openai.ParseSsePayloads(body=r_chat_sse.text)
            assert chat_done is True
            assert chat_chunks[0]["choices"] == [
                {"delta": {"content": "", "role": "assistant"}, "finish_reason": None, "index": 0}
            ]
            assert "usage" in chat_chunks[0]
            assert chat_chunks[-2]["choices"] == [
                {"delta": {}, "finish_reason": "stop", "index": 0}
            ]
            assert chat_chunks[-1]["choices"] == []
            assert chat_chunks[-1]["usage"]["completion_tokens"] == 5

    asyncio.run(Run())


def test_completion_streaming_utf8_and_final_usage() -> None:
    bundle, _, _, transport = _BuildStreamingHarness()

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_cmpl_sse = await client.post(
                "/v1/completions",
                json={
                    "max_tokens": 5,
                    "model": "test-model",
                    "prompt": [bundle.hello_id],
                    "stream": True,
                    "stream_options": {"continuous_usage_stats": False, "include_usage": True},
                },
            )
            assert r_cmpl_sse.status_code == 200
            assert r_cmpl_sse.headers["content-type"].startswith("text/event-stream")
            cmpl_chunks, saw_done = tests.openai.ParseSsePayloads(body=r_cmpl_sse.text)
            assert saw_done is True
            for ch in cmpl_chunks[:-1]:
                assert ch["object"] == "text_completion"
                assert "usage" not in ch
            streamed_cmpl_text = "".join(ch["choices"][0]["text"] for ch in cmpl_chunks[:-1])
            assert streamed_cmpl_text == "€ world"
            assert "\ufffd" not in streamed_cmpl_text
            assert cmpl_chunks[-1]["choices"] == []
            assert cmpl_chunks[-1]["usage"] == {
                "completion_tokens": 5,
                "prompt_tokens": 1,
                "total_tokens": 6,
            }

    asyncio.run(Run())


def test_pending_generation_does_not_block_model_discovery() -> None:
    bundle, core, _, transport = _BuildStreamingHarness()

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            gate = asyncio.Event()
            core.hold_gate = gate
            slow_task = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={"max_tokens": 1, "model": "test-model", "prompt": [bundle.hello_id]},
                )
            )
            await core.step_started.wait()
            fast_models = await client.get("/v1/models")
            assert fast_models.status_code == 200
            gate.set()
            slow_resp = await slow_task
            assert slow_resp.status_code == 200

    asyncio.run(Run())


def test_stream_disconnect_aborts_backend() -> None:
    _, core, app, _ = _BuildStreamingHarness()

    async def Run() -> None:
        core.pause_after_steps = 1
        core.hold_gate = asyncio.Event()
        stream_rx: asyncio.Queue[dict[str, typing.Any]] = asyncio.Queue()
        stream_tx: list[dict[str, typing.Any]] = []
        stream_body = json.dumps(
            {
                "max_tokens": 10,
                "messages": [{"content": "Hello", "role": "user"}],
                "model": "test-model",
                "stream": True,
            }
        ).encode("utf-8")
        await stream_rx.put({"body": stream_body, "more_body": False, "type": "http.request"})

        async def StreamReceive() -> dict[str, typing.Any]:
            return await stream_rx.get()

        async def StreamSend(message: dict[str, typing.Any]) -> None:
            stream_tx.append(message)

        stream_scope = {
            "client": ("127.0.0.1", 50001),
            "headers": [(b"content-type", b"application/json")],
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "server": ("127.0.0.1", 8000),
            "type": "http",
        }
        stream_task = asyncio.create_task(app(stream_scope, StreamReceive, StreamSend))
        await core.paused_event.wait()
        assert stream_tx[0]["status"] == 200
        await stream_rx.put({"type": "http.disconnect"})
        await stream_task
        assert len(core.aborted_request_ids) == 1

    asyncio.run(Run())


def test_streaming_rejects_incompatible_completion_requests() -> None:
    bundle, _, _, transport = _BuildStreamingHarness()

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for bad_stream in [
                {"model": "test-model", "prompt": ["a", "b"], "stream": True},
                {"max_tokens": 0, "model": "test-model", "prompt": "Hello", "stream": True},
                {"echo": True, "model": "test-model", "prompt": "Hello", "stream": True},
                {"logprobs": 0, "model": "test-model", "prompt": "Hello", "stream": True},
                {
                    "model": "test-model",
                    "prompt": [bundle.hello_id] * 32,
                    "stream": True,
                },
                {
                    "model": "test-model",
                    "prompt": "Hello",
                    "stream": True,
                    "stream_options": {"unknown": True},
                },
            ]:
                r_bad = await client.post("/v1/completions", json=bad_stream)
                assert r_bad.status_code == 400

    asyncio.run(Run())


def test_unary_disconnect_aborts_backend() -> None:
    bundle = tests.helpers.BuildTestTokenizer()
    core = tests.backend.ScriptedBackend(
        default_tokens=[bundle.world_id],
        eos_token_ids=frozenset({bundle.eos_id}),
        vocab_size=bundle.vocab_size,
    )
    core.hold_gate = asyncio.Event()
    engine = fs_generation.GenerationAdapter(backend=core, tokenizer=bundle.tokenizer)
    app = fs_server.BuildApp(engine=engine)

    async def Run() -> None:
        receive_queue: asyncio.Queue[dict[str, typing.Any]] = asyncio.Queue()
        sent_messages: list[dict[str, typing.Any]] = []
        raw_body = json.dumps(
            {"max_tokens": 4, "model": "test-model", "prompt": [bundle.hello_id]}
        ).encode("utf-8")
        await receive_queue.put({"body": raw_body, "more_body": False, "type": "http.request"})

        async def Receive() -> dict[str, typing.Any]:
            return await receive_queue.get()

        async def Send(message: dict[str, typing.Any]) -> None:
            sent_messages.append(message)

        scope = {
            "client": ("127.0.0.1", 50000),
            "headers": [(b"content-type", b"application/json")],
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/completions",
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "server": ("127.0.0.1", 8000),
            "type": "http",
        }
        asgi_task = asyncio.create_task(app(scope, Receive, Send))
        await core.step_started.wait()
        await receive_queue.put({"type": "http.disconnect"})
        await asgi_task

        assert len(core.aborted_request_ids) == 1
        assert sent_messages[0]["status"] == 499

    asyncio.run(Run())
