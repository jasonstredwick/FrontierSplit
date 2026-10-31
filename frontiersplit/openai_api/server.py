# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM; modified for FrontierSplit:
# completion route dispatch in a FrontierSplit application.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

import argparse
import collections.abc
import dataclasses
import fastapi
import os
import starlette.responses
import uvicorn

from frontiersplit.backend import types as fs_backend_types
from frontiersplit.openai_api import errors as fs_errors
from frontiersplit.openai_api import protocol as fs_protocol
from frontiersplit.openai_api import serving as fs_serving
from frontiersplit.openai_api import types as fs_types

API_KEY_ENV_VAR = "FRONTIERSPLIT_API_KEY"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
SSE_MEDIA_TYPE = "text/event-stream"


@dataclasses.dataclass(frozen=True, kw_only=True)
class ServerConfig:
    api_key: str | None = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


def BuildApp(
    *,
    api_key: str | None = None,
    engine: fs_types.GenerationClient | None = None,
) -> fastapi.FastAPI:
    app = fastapi.FastAPI(docs_url=None, openapi_url=None, redoc_url=None)
    serving = fs_serving.OpenAIServing(engine=engine)
    app.state.serving = serving

    fs_errors.RegisterExceptionHandlers(app=app)
    if api_key:
        app.add_middleware(fs_errors.AuthenticationMiddleware, api_key=api_key)

    async def _DispatchChat(
        *,
        request: fs_protocol.ChatCompletionRequest,
    ) -> starlette.responses.Response:
        result = await serving.CreateChatCompletion(request=request)
        if isinstance(result, fs_protocol.ChatCompletionResponse):
            return starlette.responses.JSONResponse(content=result.model_dump(mode="json"))
        return starlette.responses.StreamingResponse(result, media_type=SSE_MEDIA_TYPE)

    async def _DispatchCompletion(
        *,
        request: fs_protocol.CompletionRequest,
    ) -> starlette.responses.Response:
        result = await serving.CreateCompletion(request=request)
        if isinstance(result, fs_protocol.CompletionResponse):
            return starlette.responses.JSONResponse(content=result.model_dump(mode="json"))
        return starlette.responses.StreamingResponse(result, media_type=SSE_MEDIA_TYPE)

    @app.get("/health")
    async def Health() -> starlette.responses.Response:
        try:
            await serving.CheckReady()
        except (fs_errors.ServiceUnavailableError, fs_backend_types.BackendUnavailableError):
            return starlette.responses.Response(status_code=503)
        return starlette.responses.Response(status_code=200)

    @app.get("/v1/models")
    async def ListModels() -> starlette.responses.JSONResponse:
        models = await serving.ListModels()
        return starlette.responses.JSONResponse(content=models.model_dump(mode="json"))

    @app.post(
        "/v1/completions",
        dependencies=[fastapi.Depends(fs_errors.ValidateJsonRequest)],
    )
    async def CreateCompletion(
        *,
        raw_request: fastapi.Request,
        request: fs_protocol.CompletionRequest,
    ) -> starlette.responses.Response:
        return await fs_errors.RunWithCancellation(
            coro=_DispatchCompletion(request=request),
            raw_request=raw_request,
        )

    @app.post(
        "/v1/chat/completions",
        dependencies=[fastapi.Depends(fs_errors.ValidateJsonRequest)],
    )
    async def CreateChatCompletion(
        *,
        raw_request: fastapi.Request,
        request: fs_protocol.ChatCompletionRequest,
    ) -> starlette.responses.Response:
        return await fs_errors.RunWithCancellation(
            coro=_DispatchChat(request=request),
            raw_request=raw_request,
        )

    return app


def ParseServerConfig(
    *,
    argv: collections.abc.Sequence[str] | None = None,
    env: collections.abc.Mapping[str, str] | None = None,
) -> ServerConfig:
    parser = argparse.ArgumentParser(description="FrontierSplit OpenAI-compatible HTTP API server")
    parser.add_argument("--api-key", default=None, help="Optional Bearer token for /v1/* endpoints")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host address")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="Bind TCP port")
    args = parser.parse_args(args=argv)
    environ = os.environ if env is None else env
    raw_key = args.api_key if args.api_key is not None else environ.get(API_KEY_ENV_VAR)
    resolved_key = raw_key if raw_key else None
    return ServerConfig(api_key=resolved_key, host=args.host, port=args.port)


def RunServer(*, config: ServerConfig, engine: fs_types.GenerationClient | None = None) -> None:
    app = BuildApp(api_key=config.api_key, engine=engine)
    uvicorn.run(app, host=config.host, port=config.port)


def SetEngine(*, app: fastapi.FastAPI, engine: fs_types.GenerationClient | None) -> None:
    serving: fs_serving.OpenAIServing = app.state.serving
    serving.SetEngine(engine=engine)


if __name__ == "__main__":
    RunServer(config=ParseServerConfig())
