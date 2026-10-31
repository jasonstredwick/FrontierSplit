# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM; modified for FrontierSplit:
# authentication, validation errors, and request cancellation.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

import asyncio
import collections.abc
import fastapi
import fastapi.exceptions
import hashlib
import json
import re
import secrets
import starlette.datastructures
import starlette.exceptions
import starlette.responses
import starlette.types
import typing

from frontiersplit.backend import types as fs_backend_types


BRACKETED_LOC_RE = re.compile(r"[\[\]{}()]")
GUARDED_PREFIX = "/v1"
INTERNAL_LOC_MARKERS = frozenset(
    {
        "arguments",
        "body",
        "bool",
        "bytearray",
        "bytes",
        "call",
        "callable",
        "chain",
        "complex",
        "default",
        "dict",
        "float",
        "frozenset",
        "function-after",
        "function-before",
        "function-plain",
        "function-wrap",
        "int",
        "is-instance",
        "is-subclass",
        "json-or-python",
        "lax-or-strict",
        "list",
        "none",
        "nonetype",
        "nullable",
        "set",
        "str",
        "tagged-union",
        "tuple",
        "union",
    }
)
MAX_REPORTED_ERRORS = 10


class ApiError(Exception):
    def __init__(
        self,
        *,
        code: str | None = None,
        error_type: str,
        message: str,
        param: str | None = None,
        status_code: int,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.error_type = error_type
        self.message = message
        self.param = param
        self.status_code = status_code


class BadRequestError(ApiError):
    def __init__(self, *, code: str | None = None, message: str, param: str | None = None) -> None:
        super().__init__(
            code=code,
            error_type="invalid_request_error",
            message=message,
            param=param,
            status_code=400,
        )


class NotFoundError(ApiError):
    def __init__(self, *, message: str, param: str | None = None) -> None:
        super().__init__(error_type="not_found_error", message=message, param=param, status_code=404)


class ServiceUnavailableError(ApiError):
    def __init__(self, *, message: str = "Model backend is unavailable.") -> None:
        super().__init__(error_type="service_unavailable", message=message, status_code=503)


class AuthenticationMiddleware:
    def __init__(self, app: starlette.types.ASGIApp, *, api_key: str) -> None:
        self._app = app
        self._expected_digest = hashlib.sha256(api_key.encode("utf-8")).digest()

    def __call__(
        self,
        scope: starlette.types.Scope,
        receive: starlette.types.Receive,
        send: starlette.types.Send,
    ) -> collections.abc.Awaitable[None]:
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            return self._app(scope, receive, send)
        root_path = scope.get("root_path", "")
        path = scope["path"].removeprefix(root_path)
        if path == GUARDED_PREFIX or path.startswith(f"{GUARDED_PREFIX}/"):
            headers = starlette.datastructures.Headers(scope=scope)
            if not self._VerifyHeaders(headers=headers):
                payload = FormatErrorPayload(
                    error_type="authentication_error",
                    message="Invalid or missing Authorization Bearer token.",
                )
                resp = starlette.responses.JSONResponse(content=payload, status_code=401)
                return resp(scope, receive, send)
        return self._app(scope, receive, send)

    def _VerifyHeaders(self, *, headers: starlette.datastructures.Headers) -> bool:
        auth = headers.get("Authorization")
        if auth is None:
            return False
        scheme, sep, token = auth.partition(" ")
        if not sep or scheme.lower() != "bearer" or not token:
            return False
        candidate_digest = hashlib.sha256(token.encode("utf-8")).digest()
        return secrets.compare_digest(candidate_digest, self._expected_digest)


def FormatErrorPayload(
    *,
    code: str | None = None,
    error_type: str,
    message: str,
    param: str | None = None,
) -> dict[str, dict[str, str | None]]:
    return {
        "error": {
            "code": code,
            "message": message,
            "param": param,
            "type": error_type,
        }
    }


def CleanValidationParam(*, loc: collections.abc.Sequence[object]) -> str | None:
    parts = [
        str(part)
        for part in loc
        if not BRACKETED_LOC_RE.search(str(part)) and str(part).lower() not in INTERNAL_LOC_MARKERS
    ]
    return ".".join(parts) if parts else None


def _SummarizeValidationErrors(
    *,
    errors: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]],
) -> tuple[str, str | None]:
    param = CleanValidationParam(loc=errors[0]["loc"])
    lines: list[str] = []
    for err in errors[:MAX_REPORTED_ERRORS]:
        msg = str(err["msg"])
        clean_loc = CleanValidationParam(loc=err["loc"])
        lines.append(f"{clean_loc}: {msg}" if clean_loc is not None else msg)
    if len(errors) > MAX_REPORTED_ERRORS:
        lines.append(f"...and {len(errors) - MAX_REPORTED_ERRORS} more validation errors")
    return "; ".join(lines), param


def _RejectJsonConstant(token: str) -> typing.NoReturn:
    raise ValueError(f"Non-standard JSON numeric literal '{token}' is not allowed.")


async def ValidateJsonRequest(*, raw_request: fastapi.Request) -> None:
    content_type = raw_request.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type != "application/json":
        raise BadRequestError(message="Content-Type must be 'application/json'.")

    raw_body = await raw_request.body()
    try:
        decoded = raw_body.decode("utf-8")
        parsed = json.loads(decoded, parse_constant=_RejectJsonConstant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BadRequestError(message=f"Malformed JSON request body: {exc}") from exc

    if not isinstance(parsed, dict):
        raise BadRequestError(message="JSON request body must be an object.")


async def _ListenForDisconnect(*, request: fastapi.Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def RunWithCancellation(
    *,
    coro: collections.abc.Coroutine[typing.Any, typing.Any, starlette.responses.Response],
    raw_request: fastapi.Request,
) -> starlette.responses.Response:
    handler_task = asyncio.create_task(coro)
    disconnect_task = asyncio.create_task(_ListenForDisconnect(request=raw_request))

    done, pending = await asyncio.wait(
        (handler_task, disconnect_task),
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if handler_task in done:
        return handler_task.result()
    return starlette.responses.Response(status_code=499)


def _MapHttpStatusToType(*, status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 503:
        return "service_unavailable"
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "server_error"


def RegisterExceptionHandlers(*, app: fastapi.FastAPI) -> None:
    async def HandleApiError(
        _req: fastapi.Request,
        exc: ApiError,
    ) -> starlette.responses.JSONResponse:
        return starlette.responses.JSONResponse(
            content=FormatErrorPayload(
                code=exc.code,
                error_type=exc.error_type,
                message=exc.message,
                param=exc.param,
            ),
            status_code=exc.status_code,
        )

    async def HandleHttpException(
        _req: fastapi.Request,
        exc: starlette.exceptions.HTTPException,
    ) -> starlette.responses.JSONResponse:
        return starlette.responses.JSONResponse(
            content=FormatErrorPayload(
                error_type=_MapHttpStatusToType(status_code=exc.status_code),
                message=str(exc.detail),
            ),
            status_code=exc.status_code,
        )

    async def HandleUnavailableError(
        _req: fastapi.Request,
        exc: fs_backend_types.BackendUnavailableError,
    ) -> starlette.responses.JSONResponse:
        return starlette.responses.JSONResponse(
            content=FormatErrorPayload(
                error_type="service_unavailable",
                message=str(exc) or type(exc).__name__,
            ),
            status_code=503,
        )

    async def HandleUnhandledError(
        _req: fastapi.Request,
        _exc: Exception,
    ) -> starlette.responses.JSONResponse:
        return starlette.responses.JSONResponse(
            content=FormatErrorPayload(
                error_type="server_error",
                message="Internal server error.",
            ),
            status_code=500,
        )

    async def HandleValidationError(
        _req: fastapi.Request,
        exc: fastapi.exceptions.RequestValidationError,
    ) -> starlette.responses.JSONResponse:
        message, param = _SummarizeValidationErrors(errors=exc.errors())
        return starlette.responses.JSONResponse(
            content=FormatErrorPayload(
                error_type="invalid_request_error",
                message=message,
                param=param,
            ),
            status_code=400,
        )

    app.add_exception_handler(ApiError, HandleApiError)
    app.add_exception_handler(fastapi.exceptions.RequestValidationError, HandleValidationError)
    app.add_exception_handler(starlette.exceptions.HTTPException, HandleHttpException)
    app.add_exception_handler(fs_backend_types.BackendUnavailableError, HandleUnavailableError)
    app.add_exception_handler(fs_backend_types.GenerationError, HandleUnhandledError)
    app.add_exception_handler(Exception, HandleUnhandledError)
