# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Portions adapted from vLLM; modified for FrontierSplit:
# chat-template inspection and text-only prompt rendering.
# See THIRD_PARTY_NOTICES.md at the repository root for source attribution.

import asyncio
import collections
import concurrent.futures
import dataclasses
import functools
import jinja2
import jinja2.ext
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox
import typing
import weakref

from collections.abc import Sequence

from frontiersplit.openai_api import errors as fs_errors
from frontiersplit.openai_api import protocol as fs_protocol
from frontiersplit.openai_api import types as fs_types


# Adapted via vLLM's AssistantTracker from Hugging Face Transformers (Apache-2.0):
# https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/chat_template_utils.py
class _GenerationTagExtension(jinja2.ext.Extension):
    """Parse HuggingFace {% generation %}...{% endgeneration %} blocks in chat templates."""

    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)


DEFAULT_RENDERER_WORKERS = 4
JINJA_AST_ENV = jinja2.sandbox.ImmutableSandboxedEnvironment(
    extensions=[_GenerationTagExtension, jinja2.ext.loopcontrols],
    lstrip_blocks=True,
    trim_blocks=True,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RenderedPrompt:
    effective_max_tokens: int
    prompt_text: str | None
    prompt_token_ids: list[int]


def _AttrDottedName(*, node: jinja2.nodes.Node) -> str | None:
    if isinstance(node, jinja2.nodes.NSRef):
        return f"{node.name}.{node.attr}"
    parts: list[str] = []
    cur = node
    while isinstance(cur, jinja2.nodes.Getattr):
        parts.append(cur.attr)
        cur = cur.node
    if isinstance(cur, jinja2.nodes.Name):
        parts.append(cur.name)
        return ".".join(reversed(parts))
    return None


def _IsVarAccess(*, node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname
    if isinstance(node, jinja2.nodes.Getattr) and "." in varname:
        base, _, attr = varname.rpartition(".")
        return node.attr == attr and _IsVarAccess(node=node.node, varname=base)
    return False


def _IsAttrAccess(*, key: str, node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _IsVarAccess(node=node.node, varname=varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )
    if isinstance(node, jinja2.nodes.Getattr):
        return _IsVarAccess(node=node.node, varname=varname) and node.attr == key
    return False


def _IsVarOrElemsAccess(
    *,
    key: str | None = None,
    node: jinja2.nodes.Node,
    varname: str,
) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _IsVarOrElemsAccess(key=key, node=node.node, varname=varname)
    if isinstance(node, jinja2.nodes.Test):
        return _IsVarOrElemsAccess(key=key, node=node.node, varname=varname)
    if isinstance(node, jinja2.nodes.Getitem) and isinstance(node.arg, jinja2.nodes.Slice):
        return _IsVarOrElemsAccess(key=key, node=node.node, varname=varname)
    if isinstance(node, jinja2.nodes.CondExpr):
        return _IsVarOrElemsAccess(key=key, node=node.expr1, varname=varname) or (
            node.expr2 is not None and _IsVarOrElemsAccess(key=key, node=node.expr2, varname=varname)
        )
    if key is not None:
        return _IsAttrAccess(key=key, node=node, varname=varname)
    return _IsVarAccess(node=node, varname=varname)


def _ResolveAliases(*, root: jinja2.nodes.Node, seed_vars: Sequence[str]) -> list[str]:
    seen = set(seed_vars)
    ordered = list(seed_vars)
    queue = collections.deque(seed_vars)

    while queue:
        current = queue.popleft()
        for assign_node in root.find_all(jinja2.nodes.Assign):
            lhs = assign_node.target
            rhs = assign_node.node
            candidates: list[str] = []
            if isinstance(lhs, jinja2.nodes.Name) and _IsVarOrElemsAccess(node=rhs, varname=current):
                candidates.append(lhs.name)
            elif (
                isinstance(lhs, jinja2.nodes.Name)
                and isinstance(rhs, jinja2.nodes.Call)
                and isinstance(rhs.node, jinja2.nodes.Name)
                and rhs.node.name == "namespace"
            ):
                candidates.extend(
                    f"{lhs.name}.{kw.key}"
                    for kw in rhs.kwargs
                    if _IsVarOrElemsAccess(node=kw.value, varname=current)
                )
            elif isinstance(lhs, (jinja2.nodes.Getattr, jinja2.nodes.NSRef)) and _IsVarOrElemsAccess(
                node=rhs,
                varname=current,
            ):
                dotted = _AttrDottedName(node=lhs)
                if dotted is not None:
                    candidates.append(dotted)

            for alias in candidates:
                if alias not in seen:
                    seen.add(alias)
                    ordered.append(alias)
                    queue.append(alias)

    return ordered


def _IsMessageContentExpr(
    *,
    content_vars: Sequence[str],
    expr: jinja2.nodes.Node,
    message_vars: Sequence[str],
) -> bool:
    return any(_IsVarOrElemsAccess(key="content", node=expr, varname=mv) for mv in message_vars) or any(
        _IsVarOrElemsAccess(node=expr, varname=cv) for cv in content_vars
    )


@functools.lru_cache(maxsize=32)
def DetectTemplateContentFormat(*, chat_template: str) -> typing.Literal["string", "openai"]:
    ast = JINJA_AST_ENV.parse(chat_template)
    messages_vars = _ResolveAliases(root=ast, seed_vars=("messages",))

    message_vars: list[str] = []
    for loop_node in ast.find_all(jinja2.nodes.For):
        if isinstance(loop_node.target, jinja2.nodes.Name) and any(
            _IsVarOrElemsAccess(node=loop_node.iter, varname=mv) for mv in messages_vars
        ):
            message_vars.append(loop_node.target.name)

    direct_content_vars: list[str] = []
    for assign_node in ast.find_all(jinja2.nodes.Assign):
        if isinstance(assign_node.target, jinja2.nodes.Name) and any(
            _IsVarOrElemsAccess(key="content", node=assign_node.node, varname=mv) for mv in message_vars
        ):
            direct_content_vars.append(assign_node.target.name)
    content_vars = _ResolveAliases(root=ast, seed_vars=direct_content_vars) if direct_content_vars else []

    macro_content_params_by_loop: dict[int, set[str]] = {}
    loops_in_macros: set[int] = set()
    for macro_node in ast.find_all(jinja2.nodes.Macro):
        param_names = {arg.name for arg in macro_node.args}
        matched_params: set[str] = set()
        for call_node in ast.find_all(jinja2.nodes.Call):
            if not (isinstance(call_node.node, jinja2.nodes.Name) and call_node.node.name == macro_node.name):
                continue
            for idx, arg in enumerate(call_node.args):
                if idx < len(macro_node.args) and _IsMessageContentExpr(
                    content_vars=content_vars,
                    expr=arg,
                    message_vars=message_vars,
                ):
                    matched_params.add(macro_node.args[idx].name)
            for kwarg in call_node.kwargs:
                if kwarg.key in param_names and _IsMessageContentExpr(
                    content_vars=content_vars,
                    expr=kwarg.value,
                    message_vars=message_vars,
                ):
                    matched_params.add(kwarg.key)

        for loop_node in macro_node.find_all(jinja2.nodes.For):
            loops_in_macros.add(id(loop_node))
            if matched_params:
                macro_content_params_by_loop[id(loop_node)] = matched_params

    for loop_node in ast.find_all(jinja2.nodes.For):
        if _IsMessageContentExpr(
            content_vars=content_vars,
            expr=loop_node.iter,
            message_vars=message_vars,
        ):
            return "openai"
        macro_params = macro_content_params_by_loop.get(id(loop_node))
        if macro_params is not None and any(
            _IsVarOrElemsAccess(node=loop_node.iter, varname=param) for param in macro_params
        ):
            return "openai"
        if (
            id(loop_node) not in loops_in_macros
            and isinstance(loop_node.iter, jinja2.nodes.Name)
            and loop_node.iter.name == "content"
        ):
            return "openai"

    return "string"


def NormalizeChatContent(
    *,
    content: str | list[fs_protocol.ChatTextPart],
    content_format: typing.Literal["string", "openai"],
) -> str | list[dict[str, str]]:
    if content_format == "string":
        if isinstance(content, str):
            return content
        return "\n".join(part.text for part in content if part.text)
    if isinstance(content, str):
        return [{"text": content, "type": "text"}]
    return [{"text": part.text, "type": "text"} for part in content]


class OnlineRenderer:
    def __init__(
        self,
        *,
        engine: fs_types.GenerationClient,
        max_workers: int = DEFAULT_RENDERER_WORKERS,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self._engine = engine
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._finalizer = weakref.finalize(
            self,
            self._executor.shutdown,
            wait=False,
            cancel_futures=True,
        )

    def RenderChat(self, *, request: fs_protocol.ChatCompletionRequest) -> RenderedPrompt:
        model_cfg = self._engine.model_config
        max_model_len = model_cfg.max_model_len
        requested = request.RequestedMaxTokens()
        if requested is not None and requested > max_model_len:
            raise fs_errors.BadRequestError(
                message=f"Requested token budget ({requested}) exceeds max_model_len ({max_model_len}).",
                param="max_completion_tokens" if request.max_completion_tokens is not None else "max_tokens",
            )

        tokenizer = self._engine.tokenizer
        try:
            template = tokenizer.get_chat_template(chat_template=self._engine.chat_template)
        except ValueError as exc:
            raise fs_errors.BadRequestError(
                message="Served checkpoint does not configure a chat template.",
            ) from exc
        if not isinstance(template, str) or not template:
            raise fs_errors.BadRequestError(
                message="Served checkpoint does not configure a chat template.",
            )

        try:
            content_format = DetectTemplateContentFormat(chat_template=template)
            conversation: list[dict[str, typing.Any]] = [
                {
                    "content": NormalizeChatContent(content=msg.content, content_format=content_format),
                    "role": msg.role,
                }
                for msg in request.messages
            ]
            rendered_text = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=request.add_generation_prompt,
                chat_template=template,
                tokenize=False,
            )
        except (jinja2.TemplateError, TypeError, ValueError) as exc:
            raise fs_errors.BadRequestError(
                message=f"Chat template rendering failed: {exc}",
                param="messages",
            ) from exc

        token_ids: list[int] = list(
            tokenizer.encode(rendered_text, add_special_tokens=request.add_special_tokens)
        )
        if not token_ids:
            raise fs_errors.BadRequestError(message="Rendered chat prompt produced zero tokens.", param="messages")

        self._CheckPromptLength(max_model_len=max_model_len, param="messages", prompt_len=len(token_ids))
        remaining = max_model_len - len(token_ids)
        effective = remaining if requested is None else min(requested, remaining)
        if effective < 1:
            raise fs_errors.BadRequestError(
                message=f"Prompt fills max_model_len ({max_model_len}); zero tokens remain for generation.",
                param="messages",
            )
        return RenderedPrompt(
            effective_max_tokens=effective,
            prompt_text=rendered_text,
            prompt_token_ids=token_ids,
        )

    async def RenderChatAsync(
        self,
        *,
        request: fs_protocol.ChatCompletionRequest,
    ) -> RenderedPrompt:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(self.RenderChat, request=request),
        )

    def RenderCompletion(self, *, request: fs_protocol.CompletionRequest) -> list[RenderedPrompt]:
        model_cfg = self._engine.model_config
        max_model_len = model_cfg.max_model_len
        vocab_size = model_cfg.vocab_size
        requested = request.max_tokens
        if requested > max_model_len:
            raise fs_errors.BadRequestError(
                message=f"max_tokens ({requested}) exceeds max_model_len ({max_model_len}).",
                param="max_tokens",
            )

        raw_items = self._UnpackPromptBatch(prompt=request.prompt)
        tokenizer = self._engine.tokenizer
        rendered: list[RenderedPrompt] = []

        for item in raw_items:
            if isinstance(item, str):
                token_ids = list(tokenizer.encode(item, add_special_tokens=request.add_special_tokens))
                if not token_ids:
                    raise fs_errors.BadRequestError(
                        message="Tokenized prompt produced zero tokens.",
                        param="prompt",
                    )
                prompt_text: str | None = item if request.echo else None
            else:
                for tid in item:
                    if not (0 <= tid < vocab_size):
                        raise fs_errors.BadRequestError(
                            message=f"Token ID {tid} out of vocabulary range [0, {vocab_size}).",
                            param="prompt",
                        )
                token_ids = list(item)
                prompt_text = tokenizer.decode(token_ids) if request.echo else None

            self._CheckPromptLength(max_model_len=max_model_len, param="prompt", prompt_len=len(token_ids))
            remaining = max_model_len - len(token_ids)
            effective = min(requested, remaining)
            if requested > 0 and effective < 1:
                raise fs_errors.BadRequestError(
                    message=f"Prompt fills max_model_len ({max_model_len}); zero tokens remain for generation.",
                    param="prompt",
                )
            rendered.append(
                RenderedPrompt(
                    effective_max_tokens=effective,
                    prompt_text=prompt_text,
                    prompt_token_ids=token_ids,
                )
            )
        return rendered

    async def RenderCompletionAsync(
        self,
        *,
        request: fs_protocol.CompletionRequest,
    ) -> list[RenderedPrompt]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(self.RenderCompletion, request=request),
        )

    def Shutdown(self) -> None:
        self._finalizer()

    @staticmethod
    def _CheckPromptLength(*, max_model_len: int, param: str, prompt_len: int) -> None:
        if prompt_len > max_model_len:
            raise fs_errors.BadRequestError(
                message=f"Prompt length ({prompt_len}) exceeds max_model_len ({max_model_len}).",
                param=param,
            )

    @staticmethod
    def _UnpackPromptBatch(
        *,
        prompt: str | list[str] | list[int] | list[list[int]],
    ) -> Sequence[str | Sequence[int]]:
        if isinstance(prompt, str):
            return [prompt]
        if isinstance(prompt[0], int):
            return [prompt]
        return prompt
