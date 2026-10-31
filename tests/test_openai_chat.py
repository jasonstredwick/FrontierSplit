import asyncio
import httpx

import tests.helpers
import tests.openai


def test_chat_rejects_invalid_turn_sequence() -> None:
    _, _, transport = tests.openai.BuildHarness(max_model_len=64)

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Invalid alternating roles rejected by template raise_exception -> 400.
            r_bad_turns = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"content": "One", "role": "user"},
                        {"content": "Two", "role": "user"},
                    ],
                    "model": "test-model",
                },
            )
            assert r_bad_turns.status_code == 400

    asyncio.run(Run())


def test_chat_rejects_missing_template() -> None:
    _, _, transport = tests.openai.BuildHarness(chat_template=None, max_model_len=64)

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_no_tpl = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"content": "Hi", "role": "user"}], "model": "test-model"},
            )
            assert r_no_tpl.status_code == 400

    asyncio.run(Run())


def test_chat_rejects_unsupported_fields_and_invalid_budgets() -> None:
    _, _, transport = tests.openai.BuildHarness(max_model_len=64)

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Conflicting budgets, zero budget, unsupported chat/completion fields rejected -> 400.
            for bad_chat in [
                {
                    "max_completion_tokens": 5,
                    "max_tokens": 4,
                    "messages": [{"content": "Hi", "role": "user"}],
                    "model": "test-model",
                },
                {
                    "max_tokens": 0,
                    "messages": [{"content": "Hi", "role": "user"}],
                    "model": "test-model",
                },
                {
                    "echo": True,
                    "messages": [{"content": "Hi", "role": "user"}],
                    "model": "test-model",
                },
                {
                    "logprobs": 1,
                    "messages": [{"content": "Hi", "role": "user"}],
                    "model": "test-model",
                },
                {
                    "messages": [{"content": [], "role": "user"}],
                    "model": "test-model",
                },
                {
                    "messages": [{"content": "Hi", "role": "tool"}],
                    "model": "test-model",
                },
                {
                    "messages": [{"content": "Hi", "role": "user"}],
                    "model": "test-model",
                    "response_format": {"type": "json_object"},
                },
                {
                    "messages": [{"content": "Hi", "role": "user"}],
                    "model": "test-model",
                    "tool_choice": "auto",
                },
            ]:
                resp = await client.post("/v1/chat/completions", json=bad_chat)
                assert resp.status_code == 400, f"Expected 400 for {bad_chat}"

    asyncio.run(Run())


def test_chat_string_content_and_default_budget() -> None:
    bundle, backend, transport = tests.openai.BuildHarness(max_model_len=64)

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # String-content template skips "" parts and joins remaining parts with "\n".
            r_chat = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"content": "Be concise.", "role": "system"},
                        {
                            "content": [
                                {"text": "Hello", "type": "text"},
                                {"text": "", "type": "text"},
                                {"text": "world", "type": "text"},
                            ],
                            "role": "user",
                        },
                    ],
                    "model": "test-model",
                    "response_format": {"type": "text"},
                    "tool_choice": "none",
                    "tools": None,
                },
            )
            assert r_chat.status_code == 200
            body = r_chat.json()
            assert body["object"] == "chat.completion"
            assert body["choices"][0]["message"] == {"content": " world", "role": "assistant"}
            assert body["choices"][0]["finish_reason"] == "stop"
            # Neither max_tokens nor max_completion_tokens was passed -> resolved to remaining capacity.
            submitted = backend.submitted_requests[-1]
            assert submitted.sampling_params.max_tokens == 64 - len(submitted.prompt_token_ids)
            decoded_prompt = bundle.tokenizer.decode(submitted.prompt_token_ids)
            assert "Hello\nworld" in decoded_prompt

    asyncio.run(Run())


def test_chat_structured_content_and_matching_budgets() -> None:
    bundle, backend, transport = tests.openai.BuildHarness(
        chat_template=tests.helpers.OPENAI_CONTENT_CHAT_TEMPLATE,
        max_model_len=64,
    )

    async def Run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_oai = await client.post(
                "/v1/chat/completions",
                json={
                    "max_completion_tokens": 4,
                    "max_tokens": 4,
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
            assert r_oai.status_code == 200
            oai_prompt_ids = backend.submitted_requests[-1].prompt_token_ids
            oai_decoded = bundle.tokenizer.decode(oai_prompt_ids)
            assert "[text:Hello][text:world]" in oai_decoded

    asyncio.run(Run())
