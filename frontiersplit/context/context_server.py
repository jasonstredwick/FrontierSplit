"""Disaggregated Context Server for FrontierSplit.

Stores static, immutable prompt Key-Value (KV) caches and computes partial
attention on demand for decode workers via Online Softmax.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from frontiersplit.context.online_softmax import (
    PartialAttentionChunk,
    compute_partial_attention,
    merge_partial_attentions,
)
from frontiersplit.context.prefix_cache import RadixPrefixCache

logger = logging.getLogger("frontiersplit.context_server")

# Packet type identifiers for Context Server binary protocol
MSG_REGISTER_PROMPT_REQ = 0x20
MSG_REGISTER_PROMPT_RESP = 0x21
MSG_QUERY_PARTIAL_REQ = 0x22
MSG_QUERY_PARTIAL_RESP = 0x23
MSG_RELEASE_SESSION_REQ = 0x24
MSG_RELEASE_SESSION_RESP = 0x25
MSG_MATCH_PREFIX_REQ = 0x26
MSG_MATCH_PREFIX_RESP = 0x27
MSG_APPEND_KV_REQ = 0x28
MSG_APPEND_KV_RESP = 0x29

# 16-byte fixed binary header: MAGIC (2B), msg_type (1B), flags (1B), meta_len (4B), payload_len (8B)
CONTEXT_HEADER_FORMAT = "<2sBB I Q"
CONTEXT_HEADER_SIZE = struct.calcsize(CONTEXT_HEADER_FORMAT)
CONTEXT_MAGIC = b"FC"

# Raw binary tensor serialization dtype table (zero-pickle wire format)
_TORCH_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.float16: 1,
    torch.float32: 2,
    torch.bfloat16: 3,
    torch.float64: 4,
    torch.int64: 5,
    torch.int32: 6,
    torch.int16: 7,
    torch.int8: 8,
    torch.uint8: 9,
    torch.bool: 10,
}
_CODE_TO_TORCH_DTYPE: dict[int, torch.dtype] = {
    v: k for k, v in _TORCH_DTYPE_TO_CODE.items()
}


def pack_context_frame(msg_type: int, meta: dict[str, Any], payload: bytes) -> bytes:
    """Packs a structured metadata dict and binary payload into a binary frame."""
    meta_bytes = json.dumps(meta).encode("utf-8")
    header = struct.pack(
        CONTEXT_HEADER_FORMAT,
        CONTEXT_MAGIC,
        msg_type,
        0,  # flags
        len(meta_bytes),
        len(payload),
    )
    return header + meta_bytes + payload


async def read_context_frame_async(
    reader: asyncio.StreamReader,
) -> tuple[int, dict[str, Any], bytes]:
    """Asynchronously reads a full binary frame from an asyncio StreamReader."""
    header_bytes = await reader.readexactly(CONTEXT_HEADER_SIZE)
    magic, msg_type, _, meta_len, payload_len = struct.unpack(
        CONTEXT_HEADER_FORMAT, header_bytes
    )
    if magic != CONTEXT_MAGIC:
        raise ValueError(
            f"Invalid protocol magic: expected {CONTEXT_MAGIC!r}, got {magic!r}"
        )

    meta_bytes = await reader.readexactly(meta_len) if meta_len > 0 else b"{}"
    meta = json.loads(meta_bytes.decode("utf-8"))

    payload = await reader.readexactly(payload_len) if payload_len > 0 else b""
    return msg_type, meta, payload


def read_context_frame_sync(
    sock: socket.socket,
) -> tuple[int, dict[str, Any], bytes]:
    """Synchronously reads a full binary frame from a raw socket."""

    def recv_exact(n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        pos = 0
        while pos < n:
            nbytes = sock.recv_into(view[pos:])
            if nbytes == 0:
                raise ConnectionResetError("Socket closed while reading binary frame")
            pos += nbytes
        return bytes(buf)

    header_bytes = recv_exact(CONTEXT_HEADER_SIZE)
    magic, msg_type, _, meta_len, payload_len = struct.unpack(
        CONTEXT_HEADER_FORMAT, header_bytes
    )
    if magic != CONTEXT_MAGIC:
        raise ValueError(
            f"Invalid protocol magic: expected {CONTEXT_MAGIC!r}, got {magic!r}"
        )

    meta_bytes = recv_exact(meta_len) if meta_len > 0 else b"{}"
    meta = json.loads(meta_bytes.decode("utf-8"))
    payload = recv_exact(payload_len) if payload_len > 0 else b""
    return msg_type, meta, payload


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serializes a torch tensor to raw binary bytes without pickle/ZIP overhead."""
    cpu_t = tensor.detach().cpu().contiguous()
    dtype_code = _TORCH_DTYPE_TO_CODE.get(cpu_t.dtype)
    if dtype_code is None:
        raise ValueError(f"Unsupported tensor dtype for serialization: {cpu_t.dtype}")
    shape = cpu_t.shape
    ndim = len(shape)
    raw_bytes = cpu_t.view(torch.uint8).numpy().tobytes()
    header = struct.pack(f"<BB{ndim}IQ", dtype_code, ndim, *shape, len(raw_bytes))
    return header + raw_bytes


def _deserialize_tensor_from(
    mv: memoryview, offset: int = 0
) -> tuple[torch.Tensor, int]:
    """Deserializes a single tensor from a writable memoryview starting at offset."""
    dtype_code, ndim = struct.unpack_from("<BB", mv, offset)
    offset += 2
    dtype = _CODE_TO_TORCH_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported tensor dtype code: {dtype_code}")
    shape = struct.unpack_from(f"<{ndim}I", mv, offset) if ndim > 0 else ()
    offset += 4 * ndim
    (nbytes,) = struct.unpack_from("<Q", mv, offset)
    offset += 8
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype), offset
    sub = mv[offset : offset + nbytes]
    t = torch.frombuffer(sub, dtype=torch.uint8).view(dtype).reshape(shape)
    return t, offset + nbytes


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserializes raw binary bytes back into a torch tensor."""
    mv = memoryview(bytearray(data))
    tensor, _ = _deserialize_tensor_from(mv, 0)
    return tensor


def serialize_chunk(chunk: PartialAttentionChunk) -> bytes:
    """Serializes a PartialAttentionChunk to raw binary bytes."""
    dtype_code = (
        _TORCH_DTYPE_TO_CODE.get(chunk.dtype, 0) if chunk.dtype is not None else 0
    )
    return (
        struct.pack("<B", dtype_code)
        + serialize_tensor(chunk.accumulator)
        + serialize_tensor(chunk.max_score)
        + serialize_tensor(chunk.sum_exp)
    )


def deserialize_chunk(data: bytes) -> PartialAttentionChunk:
    """Deserializes raw binary bytes back into a PartialAttentionChunk."""
    mv = memoryview(bytearray(data))
    (dtype_code,) = struct.unpack_from("<B", mv, 0)
    target_dtype = _CODE_TO_TORCH_DTYPE.get(dtype_code) if dtype_code != 0 else None
    acc, offset = _deserialize_tensor_from(mv, 1)
    max_s, offset = _deserialize_tensor_from(mv, offset)
    sum_e, _ = _deserialize_tensor_from(mv, offset)
    return PartialAttentionChunk(
        accumulator=acc,
        max_score=max_s,
        sum_exp=sum_e,
        dtype=target_dtype,
    )


def _serialize_kv_pair(k: torch.Tensor, v: torch.Tensor) -> bytes:
    """Serializes a (Key, Value) tensor pair into a single contiguous binary payload."""
    return serialize_tensor(k) + serialize_tensor(v)


def _deserialize_kv_pair(data: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Deserializes a (Key, Value) tensor pair from a binary payload."""
    mv = memoryview(bytearray(data))
    k, offset = _deserialize_tensor_from(mv, 0)
    v, _ = _deserialize_tensor_from(mv, offset)
    return k, v


class ContextStore:
    """Thread-safe in-memory storage for static prompt Key-Value tensors and chunks."""

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        # Structure: {session_id: {layer_idx: list[tuple[torch.Tensor, torch.Tensor]]}}
        self._store: dict[str, dict[int, list[tuple[torch.Tensor, torch.Tensor]]]] = {}
        self._session_last_access: dict[str, float] = {}
        self._lock = threading.RLock()
        self.ttl_seconds = ttl_seconds

    def touch_session(self, session_id: str) -> None:
        """Refreshes the sliding TTL timestamp for a session."""
        with self._lock:
            self._session_last_access[session_id] = time.time()

    def get_last_access(self, session_id: str) -> float | None:
        """Returns the timestamp of last access for a session."""
        with self._lock:
            return self._session_last_access.get(session_id)

    def register_prompt(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Stores static prompt keys and values for a given session and layer as a single chunk."""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            self._store[session_id][layer_idx] = [(k.detach(), v.detach())]
            self.touch_session(session_id)

    def register_chunks(
        self,
        session_id: str,
        layer_idx: int,
        chunks: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Stores static prompt key-value chunks for a given session and layer without copying."""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            self._store[session_id][layer_idx] = [
                (k.detach(), v.detach()) for k, v in chunks if k.shape[-2] > 0
            ]
            self.touch_session(session_id)

    def append_kv(
        self,
        session_id: str,
        layer_idx: int,
        k_delta: torch.Tensor,
        v_delta: torch.Tensor,
    ) -> int:
        """Appends a delta Key-Value tensor slice to an existing session cache without reallocating.

        Returns:
            The new cumulative sequence length along the sequence dimension (-2).
        """
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}

            if layer_idx not in self._store[session_id]:
                self._store[session_id][layer_idx] = [
                    (k_delta.detach(), v_delta.detach())
                ]
            else:
                self._store[session_id][layer_idx].append(
                    (k_delta.detach(), v_delta.detach())
                )

            self.touch_session(session_id)
            return sum(k.shape[-2] for k, _ in self._store[session_id][layer_idx])

    def get_chunks(
        self,
        session_id: str,
        layer_idx: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Retrieves stored prompt key-value chunks, or None if not found."""
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            chunks = session.get(layer_idx)
            if chunks is None:
                return None
            self.touch_session(session_id)
            return list(chunks)

    def get_prompt(
        self,
        session_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Retrieves stored prompt keys and values (concatenating chunks if multiple exist), or None."""
        with self._lock:
            chunks = self.get_chunks(session_id, layer_idx)
            if chunks is None or len(chunks) == 0:
                return None
            if len(chunks) == 1:
                return chunks[0]
            # Multiple chunks: concatenate along sequence dimension for backward compatibility
            k_cat = torch.cat([c[0] for c in chunks], dim=-2)
            v_cat = torch.cat([c[1] for c in chunks], dim=-2)
            return (k_cat, v_cat)

    def has_session(self, session_id: str) -> bool:
        """Returns True if the session has registered prompt caches."""
        with self._lock:
            return session_id in self._store

    def release_session(self, session_id: str) -> bool:
        """Evicts the prompt KV cache for a completed session."""
        with self._lock:
            self._session_last_access.pop(session_id, None)
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    def evict_expired(self, max_idle_seconds: float | None = None) -> list[str]:
        """Evicts all sessions that have exceeded their sliding TTL window.

        Args:
            max_idle_seconds: Custom idle threshold in seconds, or self.ttl_seconds.

        Returns:
            List of evicted session IDs.
        """
        with self._lock:
            limit = (
                max_idle_seconds if max_idle_seconds is not None else self.ttl_seconds
            )
            now = time.time()
            expired = [
                sid
                for sid, last_t in list(self._session_last_access.items())
                if (now - last_t) > limit
            ]
            for sid in expired:
                self.release_session(sid)
            return expired

    def get_session_layers(
        self, session_id: str
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]] | None:
        """Retrieves all layer KV caches for a given session."""
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            res: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for l_idx in session:
                prompt = self.get_prompt(session_id, l_idx)
                if prompt is not None:
                    res[l_idx] = prompt
            return res

    def get_session_layer_chunks(
        self, session_id: str
    ) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]] | None:
        """Retrieves all layer chunk KV lists for a given session."""
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            return {l_idx: list(chunks) for l_idx, chunks in session.items()}

    @property
    def num_sessions(self) -> int:
        """Returns the number of active sessions stored in memory."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Clears all stored sessions."""
        with self._lock:
            self._store.clear()
            self._session_last_access.clear()


@dataclass
class _StagedUpload:
    """Tracks an in-progress multi-layer upload transaction for a session."""

    upload_id: str
    op: str  # "register" or "append"
    seq_len: int
    layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    token_ids: list[int] | None = None
    base_tokens: list[int] | None = None
    expected_layers: int | None = None


class ContextServer:
    """Server that computes partial attention against stored prompt KV caches."""

    def __init__(
        self,
        device: str = "cpu",
        max_cached_tokens: int = 1_000_000,
        session_ttl_seconds: float = 600.0,
        expected_num_layers: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.store = ContextStore(ttl_seconds=session_ttl_seconds)
        self.prefix_cache = RadixPrefixCache(max_cached_tokens=max_cached_tokens)
        self.expected_num_layers = expected_num_layers
        self._known_layers: set[int] = (
            set(range(expected_num_layers))
            if expected_num_layers is not None and expected_num_layers > 0
            else set()
        )
        self._session_expected_layers: dict[str, int] = {}
        self._session_tokens: dict[str, list[int]] = {}
        self._staged_uploads: dict[str, _StagedUpload] = {}
        self._pinned_sessions: dict[str, tuple[int, ...]] = {}
        self._server: asyncio.Server | None = None
        self._ttl_task: asyncio.Task[None] | None = None
        self._thread: threading.Thread | None = None
        self._thread_loop: asyncio.AbstractEventLoop | None = None
        self._active_tasks: set[asyncio.Task[Any]] = set()

    def _walk_prefix_nodes(self, seq: tuple[int, ...]) -> list[Any]:
        """Walks the Radix tree along `seq` without mutating or splitting nodes."""
        curr = self.prefix_cache.root
        pos = 0
        nodes = []
        while pos < len(seq):
            first_tok = seq[pos]
            child = curr.children.get(first_tok)
            if child is None:
                break
            nodes.append(child)
            pos += child.num_tokens
            curr = child
        return nodes

    def _set_session_pin(
        self, session_id: str, tokens: list[int] | tuple[int, ...]
    ) -> None:
        """Pins the RadixNode chain corresponding to `tokens` for `session_id`."""
        new_seq = tuple(tokens)
        if self._pinned_sessions.get(session_id) == new_seq:
            return
        if session_id in self._pinned_sessions:
            self._clear_session_pin(session_id)
        for node in self._walk_prefix_nodes(new_seq):
            node.session_count += 1
        self._pinned_sessions[session_id] = new_seq

    def _clear_session_pin(self, session_id: str) -> None:
        """Unpins the RadixNode chain previously bound to `session_id`."""
        old_seq = self._pinned_sessions.pop(session_id, None)
        if old_seq is None:
            return
        for node in self._walk_prefix_nodes(old_seq):
            node.session_count = max(0, node.session_count - 1)

    def _required_layers_for_session(
        self, session_id: str, current_layers: dict[int, Any]
    ) -> set[int]:
        """Computes the set of layer indices required before a session cache is complete."""
        required = set(self._known_layers) | set(current_layers.keys())
        if self.expected_num_layers is not None and self.expected_num_layers > 0:
            required |= set(range(self.expected_num_layers))
        exp = self._session_expected_layers.get(session_id)
        if exp is not None and exp > 0:
            required |= set(range(exp))
        return required

    def _try_publish_staged_upload(self, session_id: str, state: _StagedUpload) -> None:
        """Atomically publishes a staged upload to RadixPrefixCache once all expected layers and token_ids are present."""
        if state.token_ids is None or not state.token_ids:
            return
        if not state.layers:
            return

        if state.op == "register":
            full_tokens = list(state.token_ids)
            target_len = len(full_tokens)
            ready_layers = {
                l_i: kv
                for l_i, kv in state.layers.items()
                if kv[0].shape[-2] == target_len and kv[1].shape[-2] == target_len
            }
        else:
            if state.base_tokens is None:
                return
            full_tokens = list(state.base_tokens) + list(state.token_ids)
            delta_len = len(state.token_ids)
            ready_layers = {
                l_i: kv
                for l_i, kv in state.layers.items()
                if kv[0].shape[-2] == delta_len and kv[1].shape[-2] == delta_len
            }

        required = self._required_layers_for_session(session_id, state.layers)
        if state.expected_layers is not None and state.expected_layers > 0:
            required |= set(range(state.expected_layers))
        if not required or not required.issubset(ready_layers.keys()):
            return

        self._session_tokens[session_id] = full_tokens
        if state.op == "register":
            k_dict = {l_i: kv[0].clone() for l_i, kv in ready_layers.items()}
            v_dict = {l_i: kv[1].clone() for l_i, kv in ready_layers.items()}
            self.prefix_cache.insert(full_tokens, k_dict, v_dict)
        else:
            assert state.base_tokens is not None
            k_dict = {l_i: kv[0].clone() for l_i, kv in ready_layers.items()}
            v_dict = {l_i: kv[1].clone() for l_i, kv in ready_layers.items()}
            self.prefix_cache.insert_delta(
                base_tokens=state.base_tokens,
                suffix_tokens=state.token_ids,
                k_delta_by_layer=k_dict,
                v_delta_by_layer=v_dict,
            )
        self._set_session_pin(session_id, full_tokens)

        # Backfill any active prefix-bound sessions that matched a prefix of `full_tokens`
        # before later layers were discovered (applicable during register).
        if state.op == "register":
            for other_sid, other_toks in list(self._session_tokens.items()):
                if other_sid == session_id or not self.store.has_session(other_sid):
                    continue
                other_len = len(other_toks)
                if (
                    0 < other_len <= target_len
                    and other_toks == full_tokens[:other_len]
                ):
                    other_layers = self.store.get_session_layer_chunks(other_sid) or {}
                    for l_i, (k_full, v_full) in ready_layers.items():
                        if l_i not in other_layers:
                            self.store.register_chunks(
                                other_sid,
                                l_i,
                                [
                                    (
                                        k_full[:, :, :other_len, :],
                                        v_full[:, :, :other_len, :],
                                    )
                                ],
                            )

    def register_prompt(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        token_ids: list[int] | None = None,
        num_layers: int | None = None,
        upload_id: str | None = None,
    ) -> None:
        """Registers a static prompt KV cache in memory and indexes into prefix cache."""
        self._known_layers.add(layer_idx)
        if num_layers is not None and num_layers > 0:
            self._session_expected_layers[session_id] = num_layers
            self._known_layers.update(range(num_layers))

        k_dev = k.to(self.device)
        v_dev = v.to(self.device)
        seq_len = int(k_dev.shape[-2])

        state = self._staged_uploads.get(session_id)
        eff_upload_id = upload_id or "__implicit__"
        is_new_upload = (
            state is None
            or state.op != "register"
            or state.upload_id != eff_upload_id
            or layer_idx in state.layers
            or state.seq_len != seq_len
            or (
                token_ids is not None
                and state.token_ids is not None
                and list(token_ids) != state.token_ids
            )
        )
        if is_new_upload:
            self._clear_session_pin(session_id)
            self._session_tokens.pop(session_id, None)
            self.store.release_session(session_id)
            state = _StagedUpload(
                upload_id=eff_upload_id,
                op="register",
                seq_len=seq_len,
                expected_layers=num_layers,
            )
            self._staged_uploads[session_id] = state

        assert state is not None
        if num_layers is not None and num_layers > 0:
            state.expected_layers = num_layers
        if token_ids is not None:
            state.token_ids = list(token_ids)
        state.layers[layer_idx] = (k_dev, v_dev)

        self.store.register_prompt(
            session_id=session_id,
            layer_idx=layer_idx,
            k=k_dev,
            v=v_dev,
        )

        self._try_publish_staged_upload(session_id, state)

    def append_kv(
        self,
        session_id: str,
        layer_idx: int,
        k_delta: torch.Tensor,
        v_delta: torch.Tensor,
        token_ids: list[int] | None = None,
        num_layers: int | None = None,
        upload_id: str | None = None,
    ) -> int:
        """Appends a delta KV tensor slice to an active session cache."""
        self._known_layers.add(layer_idx)
        if num_layers is not None and num_layers > 0:
            self._session_expected_layers[session_id] = num_layers
            self._known_layers.update(range(num_layers))

        k_dev = k_delta.to(self.device)
        v_dev = v_delta.to(self.device)
        delta_len = int(k_dev.shape[-2])

        state = self._staged_uploads.get(session_id)
        eff_upload_id = upload_id or "__implicit__"
        is_new_upload = (
            state is None
            or state.op != "append"
            or state.upload_id != eff_upload_id
            or layer_idx in state.layers
            or state.seq_len != delta_len
            or (
                token_ids is not None
                and state.token_ids is not None
                and list(token_ids) != state.token_ids
            )
        )
        if is_new_upload:
            base_toks = (
                list(self._session_tokens[session_id])
                if session_id in self._session_tokens
                else None
            )
            state = _StagedUpload(
                upload_id=eff_upload_id,
                op="append",
                seq_len=delta_len,
                base_tokens=base_toks,
                expected_layers=num_layers,
            )
            self._staged_uploads[session_id] = state

        assert state is not None
        new_len = self.store.append_kv(
            session_id=session_id,
            layer_idx=layer_idx,
            k_delta=k_dev,
            v_delta=v_dev,
        )

        state.layers[layer_idx] = (k_dev, v_dev)
        if num_layers is not None and num_layers > 0:
            state.expected_layers = num_layers
        if token_ids is not None:
            state.token_ids = list(token_ids)

        self._try_publish_staged_upload(session_id, state)
        return new_len

    def evict_expired_sessions(
        self, max_idle_seconds: float | None = None
    ) -> list[str]:
        """Evicts sessions whose sliding TTL has expired."""
        evicted = self.store.evict_expired(max_idle_seconds=max_idle_seconds)
        for sid in evicted:
            self._clear_session_pin(sid)
            self._staged_uploads.pop(sid, None)
            self._session_tokens.pop(sid, None)
            self._session_expected_layers.pop(sid, None)
        return evicted

    def match_prefix(
        self,
        token_ids: list[int],
        target_session_id: str | None = None,
    ) -> tuple[int, str | None]:
        """Matches a token sequence against cached prefixes in the Radix tree.

        Returns:
            Tuple of (matched_length, bound_session_id).
        """
        required = set(self._known_layers)
        if self.expected_num_layers is not None and self.expected_num_layers > 0:
            required |= set(range(self.expected_num_layers))

        nodes, matched_len = self.prefix_cache.match(
            token_ids, required_layers=required if required else None
        )
        if matched_len == 0 or not nodes:
            return 0, None

        if required:
            for node in nodes:
                if not (
                    required.issubset(node.k_cache.keys())
                    and required.issubset(node.v_cache.keys())
                ):
                    return 0, None
            available_layers = set(required)
        else:
            available_layers = set(nodes[0].k_cache.keys()) & set(
                nodes[0].v_cache.keys()
            )
            for node in nodes[1:]:
                available_layers &= set(node.k_cache.keys()) & set(node.v_cache.keys())
            if not available_layers:
                return 0, None

        session_id = (
            target_session_id or f"prefix-{abs(hash(tuple(token_ids[:matched_len]))):x}"
        )
        matched_tokens = list(token_ids[:matched_len])
        self._staged_uploads.pop(session_id, None)
        self.store.release_session(session_id)
        self._session_tokens[session_id] = matched_tokens

        # Zero-copy registration: reference node KV caches directly as chunks
        for layer_idx in sorted(available_layers):
            chunks = [
                (node.k_cache[layer_idx], node.v_cache[layer_idx]) for node in nodes
            ]
            self.store.register_chunks(session_id, layer_idx, chunks)

        self._set_session_pin(session_id, matched_tokens)
        return matched_len, session_id

    def query_partial_attention(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,
        scale: float | None = None,
        context_len: int | None = None,
        sliding_window: int | None = None,
        past_local_len: int = 0,
    ) -> PartialAttentionChunk:
        """Computes partial attention for a given query against stored prompt KV chunks.

        Evaluates attention chunk-by-chunk using associative Online Softmax reduction,
        guaranteeing bit-for-bit exactness without contiguous memory copies.
        """
        chunks = self.store.get_chunks(session_id, layer_idx)
        if (chunks is None or len(chunks) == 0) and session_id in self._session_tokens:
            # Attempt lazy backfill from RadixPrefixCache if layer was published after match_prefix
            toks = self._session_tokens[session_id]
            nodes, m_len = self.prefix_cache.match(toks)
            if (
                m_len == len(toks)
                and nodes
                and all(
                    layer_idx in n.k_cache and layer_idx in n.v_cache for n in nodes
                )
            ):
                lazy_chunks = [
                    (n.k_cache[layer_idx], n.v_cache[layer_idx]) for n in nodes
                ]
                self.store.register_chunks(session_id, layer_idx, lazy_chunks)
                chunks = self.store.get_chunks(session_id, layer_idx)

        if chunks is None or len(chunks) == 0:
            raise KeyError(
                f"No prompt KV cache found for session '{session_id}', layer {layer_idx}"
            )

        q_dev = q.to(self.device)
        batch_size, num_heads, q_len, head_dim = q_dev.shape

        # Calculate total sequence length across all chunks
        total_p_len = sum(k_c.shape[-2] for k_c, _ in chunks)
        eff_len = (
            min(total_p_len, context_len) if context_len is not None else total_p_len
        )

        # Determine sliding window boundaries
        first_q_pos = eff_len + max(0, past_local_len - q_len)
        if sliding_window is not None and sliding_window > 0:
            if q_len == 1:
                if past_local_len <= 0:
                    start_bound = max(0, eff_len - sliding_window)
                else:
                    rem_window = sliding_window - past_local_len
                    if rem_window <= 0:
                        start_bound = eff_len
                    else:
                        start_bound = max(0, eff_len - rem_window)
            else:
                start_bound = max(0, first_q_pos - sliding_window + 1)
        else:
            start_bound = 0

        # If effective range is empty, return empty chunk immediately
        if start_bound >= eff_len or eff_len <= 0:
            return compute_partial_attention(
                q_dev,
                chunks[0][0][:, :, 0:0, :],
                chunks[0][1][:, :, 0:0, :],
                scale=scale,
            )

        # Single-chunk fast-path when only 1 chunk exists:
        if len(chunks) == 1:
            k_c, v_c = chunks[0]
            c_len = k_c.shape[-2]
            slice_end = min(c_len, eff_len)
            slice_start = max(0, start_bound)
            if slice_start >= slice_end:
                return compute_partial_attention(
                    q_dev, k_c[:, :, 0:0, :], v_c[:, :, 0:0, :], scale=scale
                )
            k_sub = k_c[:, :, slice_start:slice_end, :]
            v_sub = v_c[:, :, slice_start:slice_end, :]

            remote_mask: torch.Tensor | None = None
            if sliding_window is not None and sliding_window > 0 and q_len > 1:
                rem_len = k_sub.shape[-2]
                q_pos = torch.arange(
                    first_q_pos, first_q_pos + q_len, device=self.device
                ).unsqueeze(1)
                k_pos = torch.arange(
                    slice_start, slice_end, device=self.device
                ).unsqueeze(0)
                out_of_window = k_pos <= (q_pos - sliding_window)
                if out_of_window.any():
                    remote_mask = torch.where(
                        out_of_window,
                        torch.full(
                            (q_len, rem_len),
                            float("-inf"),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        torch.zeros(
                            (q_len, rem_len),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    ).view(1, 1, q_len, rem_len)

            return compute_partial_attention(
                q_dev, k_sub, v_sub, scale=scale, attention_mask=remote_mask
            )

        # Multi-chunk path: iterate through chunks, slice zero-copy views, compute partials & merge
        partial_chunks: list[PartialAttentionChunk] = []
        curr_offset = 0

        for k_c, v_c in chunks:
            c_len = k_c.shape[-2]
            chunk_start = curr_offset
            chunk_end = curr_offset + c_len
            curr_offset = chunk_end

            # Determine intersection with [start_bound, eff_len)
            v_start = max(chunk_start, start_bound)
            v_end = min(chunk_end, eff_len)

            if v_start >= v_end:
                continue

            rel_start = v_start - chunk_start
            rel_end = v_end - chunk_start
            k_sub = k_c[:, :, rel_start:rel_end, :]
            v_sub = v_c[:, :, rel_start:rel_end, :]
            sub_len = k_sub.shape[-2]

            remote_mask = None
            if sliding_window is not None and sliding_window > 0 and q_len > 1:
                q_pos = torch.arange(
                    first_q_pos, first_q_pos + q_len, device=self.device
                ).unsqueeze(1)
                k_pos = torch.arange(v_start, v_end, device=self.device).unsqueeze(0)
                out_of_window = k_pos <= (q_pos - sliding_window)
                if out_of_window.any():
                    remote_mask = torch.where(
                        out_of_window,
                        torch.full(
                            (q_len, sub_len),
                            float("-inf"),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        torch.zeros(
                            (q_len, sub_len),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    ).view(1, 1, q_len, sub_len)

            p_chunk = compute_partial_attention(
                q_dev, k_sub, v_sub, scale=scale, attention_mask=remote_mask
            )
            partial_chunks.append(p_chunk)

        if not partial_chunks:
            return compute_partial_attention(
                q_dev,
                chunks[0][0][:, :, 0:0, :],
                chunks[0][1][:, :, 0:0, :],
                scale=scale,
            )

        return merge_partial_attentions(partial_chunks)

    def release_session(self, session_id: str) -> bool:
        """Evicts a session's static prompt KV cache."""
        self._clear_session_pin(session_id)
        self._staged_uploads.pop(session_id, None)
        self._session_tokens.pop(session_id, None)
        self._session_expected_layers.pop(session_id, None)
        return self.store.release_session(session_id)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handles an incoming TCP client connection for partial attention requests."""
        # Enable TCP_NODELAY for sub-millisecond line-rate transmission
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_tasks.add(current_task)

        try:
            while not reader.at_eof():
                try:
                    msg_type, meta, payload = await read_context_frame_async(reader)
                except asyncio.IncompleteReadError:
                    break

                if msg_type == MSG_QUERY_PARTIAL_REQ:
                    session_id = meta["session_id"]
                    layer_idx = meta["layer_idx"]
                    scale = meta.get("scale")
                    context_len = meta.get("context_len")
                    sliding_window = meta.get("sliding_window")
                    past_local_len = int(meta.get("past_local_len", 0))
                    q = deserialize_tensor(payload)

                    try:
                        chunk = self.query_partial_attention(
                            session_id,
                            layer_idx,
                            q,
                            scale=scale,
                            context_len=context_len,
                            sliding_window=sliding_window,
                            past_local_len=past_local_len,
                        )
                        chunk_bytes = serialize_chunk(chunk)
                        resp = pack_context_frame(
                            MSG_QUERY_PARTIAL_RESP,
                            {
                                "status": "ok",
                                "session_id": session_id,
                                "layer_idx": layer_idx,
                            },
                            chunk_bytes,
                        )
                    except KeyError as e:
                        resp = pack_context_frame(
                            MSG_QUERY_PARTIAL_RESP,
                            {
                                "status": "error",
                                "error": str(e),
                                "session_id": session_id,
                                "layer_idx": layer_idx,
                            },
                            b"",
                        )
                    writer.write(resp)
                    await writer.drain()

                elif msg_type == MSG_APPEND_KV_REQ:
                    session_id = meta["session_id"]
                    layer_idx = meta["layer_idx"]
                    token_ids = meta.get("token_ids")
                    num_layers = meta.get("num_layers")
                    upload_id = meta.get("upload_id")
                    k_delta, v_delta = _deserialize_kv_pair(payload)
                    new_len = self.append_kv(
                        session_id=session_id,
                        layer_idx=layer_idx,
                        k_delta=k_delta,
                        v_delta=v_delta,
                        token_ids=token_ids,
                        num_layers=num_layers,
                        upload_id=upload_id,
                    )
                    resp = pack_context_frame(
                        MSG_APPEND_KV_RESP,
                        {
                            "status": "ok",
                            "session_id": session_id,
                            "layer_idx": layer_idx,
                            "new_len": new_len,
                        },
                        b"",
                    )
                    writer.write(resp)
                    await writer.drain()

                elif msg_type == MSG_REGISTER_PROMPT_REQ:
                    session_id = meta["session_id"]
                    layer_idx = meta["layer_idx"]
                    token_ids = meta.get("token_ids")
                    num_layers = meta.get("num_layers")
                    upload_id = meta.get("upload_id")
                    k_tensor, v_tensor = _deserialize_kv_pair(payload)
                    self.register_prompt(
                        session_id,
                        layer_idx,
                        k_tensor,
                        v_tensor,
                        token_ids=token_ids,
                        num_layers=num_layers,
                        upload_id=upload_id,
                    )

                    resp = pack_context_frame(
                        MSG_REGISTER_PROMPT_RESP,
                        {
                            "status": "ok",
                            "session_id": session_id,
                            "layer_idx": layer_idx,
                        },
                        b"",
                    )
                    writer.write(resp)
                    await writer.drain()

                elif msg_type == MSG_MATCH_PREFIX_REQ:
                    token_ids = meta["token_ids"]
                    target_session_id = meta.get("session_id")
                    matched_len, bound_session_id = self.match_prefix(
                        token_ids=token_ids,
                        target_session_id=target_session_id,
                    )
                    resp = pack_context_frame(
                        MSG_MATCH_PREFIX_RESP,
                        {
                            "status": "ok",
                            "matched_len": matched_len,
                            "session_id": bound_session_id,
                            "req_session_id": target_session_id,
                        },
                        b"",
                    )
                    writer.write(resp)
                    await writer.drain()

                elif msg_type == MSG_RELEASE_SESSION_REQ:
                    session_id = meta["session_id"]
                    evicted = self.release_session(session_id)
                    resp = pack_context_frame(
                        MSG_RELEASE_SESSION_RESP,
                        {
                            "status": "ok",
                            "evicted": evicted,
                            "session_id": session_id,
                        },
                        b"",
                    )
                    writer.write(resp)
                    await writer.drain()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Client connection terminated: %s", e)
        finally:
            if current_task is not None:
                self._active_tasks.discard(current_task)
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _ttl_eviction_loop(self) -> None:
        """Periodically sweeps expired multi-turn sessions in the background."""
        interval = max(0.5, min(self.store.ttl_seconds / 2.0, 30.0))
        try:
            while True:
                await asyncio.sleep(interval)
                self.evict_expired_sessions()
        except asyncio.CancelledError:
            pass

    async def start_server(
        self, host: str = "127.0.0.1", port: int = 50055
    ) -> asyncio.Server:
        """Starts the asynchronous TCP context service and background TTL task."""
        self._server = await asyncio.start_server(self._handle_connection, host, port)
        if self._ttl_task is None or self._ttl_task.done():
            self._ttl_task = asyncio.create_task(self._ttl_eviction_loop())
        return self._server

    async def stop_server(self) -> None:
        """Stops the TCP server and waits for client tasks to finish."""
        if self._ttl_task is not None:
            self._ttl_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._ttl_task
            self._ttl_task = None

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._active_tasks:
            for t in list(self._active_tasks):
                t.cancel()
            await asyncio.gather(*list(self._active_tasks), return_exceptions=True)
            self._active_tasks.clear()
        await asyncio.sleep(0)

    def start_in_thread(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Starts the Context Server in a dedicated background daemon thread.

        Useful for embedded testing and single-process demonstrations without
        blocking the main thread's synchronous PyTorch forward passes.

        Args:
            host: IP address to bind (defaults to "127.0.0.1").
            port: Port to bind (0 for automatic dynamic free port).

        Returns:
            The bound TCP port.
        """
        ready = threading.Event()
        server_port: list[int] = []

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._thread_loop = loop

            async def _start() -> None:
                srv = await self.start_server(host, port)
                sock = srv.sockets[0]
                server_port.append(sock.getsockname()[1])
                ready.set()

            loop.run_until_complete(_start())
            loop.run_forever()

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()
        ready.wait(timeout=10.0)
        return server_port[0]

    def stop_thread(self) -> None:
        """Stops the background server thread and closes active server sockets."""
        if self._thread_loop is not None and self._thread_loop.is_running():

            async def _cleanup() -> None:
                await self.stop_server()
                assert self._thread_loop is not None
                self._thread_loop.stop()

            fut = asyncio.run_coroutine_threadsafe(_cleanup(), self._thread_loop)
            with contextlib.suppress(Exception):
                fut.result(timeout=3.0)
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class ContextClient:
    """Client for querying a remote Context Server over persistent TCP."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 50055, timeout: float = 30.0
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._sync_sock: socket.socket | None = None
        self._sync_lock = threading.RLock()
        self._sync_in_flight: tuple[str, int, torch.device] | None = None
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Returns an asyncio.Lock bound to the currently running event loop."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _reset_async_transport(self) -> None:
        """Immediately closes and discards the async StreamWriter/StreamReader."""
        if self.writer is not None:
            with contextlib.suppress(Exception):
                self.writer.close()
        self.writer = None
        self.reader = None

    @staticmethod
    def _validate_response_meta(
        req_meta: dict[str, Any],
        resp_meta: dict[str, Any],
        expected_resp_type: int,
        actual_resp_type: int,
    ) -> None:
        """Validates that the response frame type and identity match the request."""
        if actual_resp_type != expected_resp_type:
            raise RuntimeError(
                f"Unexpected response frame type: expected {expected_resp_type:#x}, got {actual_resp_type:#x}"
            )
        if expected_resp_type == MSG_MATCH_PREFIX_RESP:
            req_sid = req_meta.get("session_id")
            if (
                req_sid is not None
                and "req_session_id" in resp_meta
                and resp_meta["req_session_id"] != req_sid
            ):
                raise RuntimeError(
                    f"Mismatched match_prefix response session_id: expected {req_sid!r}, got {resp_meta['req_session_id']!r}"
                )
            if (
                req_sid is not None
                and resp_meta.get("matched_len", 0) > 0
                and resp_meta.get("session_id") not in (None, req_sid)
            ):
                raise RuntimeError(
                    f"Mismatched bound session_id: expected {req_sid!r}, got {resp_meta.get('session_id')!r}"
                )
        else:
            if (
                "session_id" in req_meta
                and "session_id" in resp_meta
                and resp_meta["session_id"] != req_meta["session_id"]
            ):
                raise RuntimeError(
                    f"Mismatched response session_id: expected {req_meta['session_id']!r}, got {resp_meta['session_id']!r}"
                )
            if (
                "layer_idx" in req_meta
                and "layer_idx" in resp_meta
                and resp_meta["layer_idx"] != req_meta["layer_idx"]
            ):
                raise RuntimeError(
                    f"Mismatched response layer_idx: expected {req_meta['layer_idx']}, got {resp_meta['layer_idx']}"
                )

    async def _exchange_async(
        self,
        req_msg_type: int,
        expected_resp_type: int,
        meta: dict[str, Any],
        payload: bytes,
    ) -> tuple[dict[str, Any], bytes]:
        """Performs a cancellation- and timeout-safe async request/response exchange over TCP."""
        async with self._get_lock():
            try:
                async with asyncio.timeout(self.timeout):
                    await self.connect()
                    assert self.writer is not None and self.reader is not None
                    frame = pack_context_frame(req_msg_type, meta, payload)
                    self.writer.write(frame)
                    await self.writer.drain()
                    resp_type, resp_meta, resp_payload = await read_context_frame_async(
                        self.reader
                    )
                    self._validate_response_meta(
                        meta, resp_meta, expected_resp_type, resp_type
                    )
            except BaseException:
                self._reset_async_transport()
                raise
            return resp_meta, resp_payload

    def _exchange_sync(
        self,
        req_msg_type: int,
        expected_resp_type: int,
        meta: dict[str, Any],
        payload: bytes,
    ) -> tuple[dict[str, Any], bytes]:
        """Performs an exception-safe synchronous request/response exchange over TCP."""
        with self._sync_lock:
            if self._sync_in_flight is not None:
                self.close_sync()
            sock = self.connect_sync()
            frame = pack_context_frame(req_msg_type, meta, payload)
            try:
                sock.sendall(frame)
                resp_type, resp_meta, resp_payload = read_context_frame_sync(sock)
                self._validate_response_meta(
                    meta, resp_meta, expected_resp_type, resp_type
                )
            except BaseException:
                self.close_sync()
                raise
            return resp_meta, resp_payload

    def connect_sync(self) -> socket.socket:
        """Establishes or returns a synchronous persistent TCP socket with configured timeout."""
        with self._sync_lock:
            if self._sync_sock is None or self._sync_sock.fileno() == -1:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self.timeout)
                try:
                    sock.connect((self.host, self.port))
                except BaseException:
                    with contextlib.suppress(Exception):
                        sock.close()
                    raise
                self._sync_sock = sock
            else:
                self._sync_sock.settimeout(self.timeout)
            return self._sync_sock

    async def connect(self) -> None:
        """Establishes a persistent TCP connection to the Context Server."""
        current_loop = asyncio.get_running_loop()
        if (
            self.writer is not None
            and not self.writer.is_closing()
            and getattr(self.writer, "_loop", None) is current_loop
        ):
            return

        self._reset_async_transport()

        try:
            async with asyncio.timeout(self.timeout):
                self.reader, self.writer = await asyncio.open_connection(
                    self.host, self.port
                )
        except BaseException:
            self._reset_async_transport()
            raise
        sock = self.writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    async def close(self) -> None:
        """Closes both async and sync TCP connections."""
        self.close_sync()

        if self.writer is not None:
            with contextlib.suppress(Exception):
                self.writer.close()
                await self.writer.wait_closed()
            self.writer = None
            self.reader = None

    async def register_prompt(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        token_ids: list[int] | None = None,
        num_layers: int | None = None,
        upload_id: str | None = None,
    ) -> bool:
        """Registers a static prompt KV cache on the remote Context Server."""
        payload = _serialize_kv_pair(k, v)
        meta: dict[str, Any] = {"session_id": session_id, "layer_idx": layer_idx}
        if token_ids is not None:
            meta["token_ids"] = token_ids
        if num_layers is not None:
            meta["num_layers"] = num_layers
        if upload_id is not None:
            meta["upload_id"] = upload_id

        resp_meta, _ = await self._exchange_async(
            MSG_REGISTER_PROMPT_REQ, MSG_REGISTER_PROMPT_RESP, meta, payload
        )
        return resp_meta.get("status") == "ok"

    async def query_partial_attention(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,
        scale: float | None = None,
        context_len: int | None = None,
        sliding_window: int | None = None,
        past_local_len: int = 0,
    ) -> PartialAttentionChunk:
        """Queries the remote Context Server to compute partial attention over prompt."""
        payload = serialize_tensor(q)
        meta: dict[str, Any] = {
            "session_id": session_id,
            "layer_idx": layer_idx,
            "scale": scale,
        }
        if context_len is not None:
            meta["context_len"] = context_len
        if sliding_window is not None:
            meta["sliding_window"] = sliding_window
            meta["past_local_len"] = past_local_len

        resp_meta, resp_payload = await self._exchange_async(
            MSG_QUERY_PARTIAL_REQ, MSG_QUERY_PARTIAL_RESP, meta, payload
        )
        if resp_meta.get("status") != "ok":
            raise RuntimeError(
                f"Context Server error: {resp_meta.get('error', 'unknown error')}"
            )
        return deserialize_chunk(resp_payload).to(q.device)

    def send_query_partial_sync(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,
        scale: float | None = None,
        context_len: int | None = None,
        sliding_window: int | None = None,
        past_local_len: int = 0,
    ) -> None:
        """Sends a partial attention query frame over the synchronous TCP socket without blocking on response."""
        with self._sync_lock:
            if self._sync_in_flight is not None:
                self.close_sync()
            sock = self.connect_sync()
            payload = serialize_tensor(q)
            meta: dict[str, Any] = {
                "session_id": session_id,
                "layer_idx": layer_idx,
                "scale": scale,
            }
            if context_len is not None:
                meta["context_len"] = context_len
            if sliding_window is not None:
                meta["sliding_window"] = sliding_window
                meta["past_local_len"] = past_local_len

            frame = pack_context_frame(MSG_QUERY_PARTIAL_REQ, meta, payload)
            try:
                sock.sendall(frame)
                self._sync_in_flight = (session_id, layer_idx, q.device)
            except BaseException:
                self.close_sync()
                raise

    def recv_query_partial_sync(
        self, device: torch.device | str | None = None
    ) -> PartialAttentionChunk:
        """Receives and deserializes a partial attention response from the synchronous TCP socket."""
        with self._sync_lock:
            in_flight = self._sync_in_flight
            self._sync_in_flight = None
            sock = self.connect_sync()
            try:
                resp_type, meta, resp_payload = read_context_frame_sync(sock)
                req_meta: dict[str, Any] = {}
                if in_flight is not None:
                    req_meta["session_id"] = in_flight[0]
                    req_meta["layer_idx"] = in_flight[1]
                self._validate_response_meta(
                    req_meta, meta, MSG_QUERY_PARTIAL_RESP, resp_type
                )
            except BaseException:
                self.close_sync()
                raise

            if meta.get("status") != "ok":
                raise RuntimeError(
                    f"Context Server error: {meta.get('error', 'unknown error')}"
                )
            chunk = deserialize_chunk(resp_payload)
            target_dev = device or (in_flight[2] if in_flight is not None else None)
            if target_dev is not None:
                chunk = chunk.to(target_dev)
            return chunk

    def query_partial_attention_sync(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,
        scale: float | None = None,
        context_len: int | None = None,
        sliding_window: int | None = None,
        past_local_len: int = 0,
    ) -> PartialAttentionChunk:
        """Queries the remote Context Server synchronously over raw TCP socket."""
        with self._sync_lock:
            self.send_query_partial_sync(
                session_id=session_id,
                layer_idx=layer_idx,
                q=q,
                scale=scale,
                context_len=context_len,
                sliding_window=sliding_window,
                past_local_len=past_local_len,
            )
            return self.recv_query_partial_sync(device=q.device)

    async def append_kv(
        self,
        session_id: str,
        layer_idx: int,
        k_delta: torch.Tensor,
        v_delta: torch.Tensor,
        token_ids: list[int] | None = None,
        num_layers: int | None = None,
        upload_id: str | None = None,
    ) -> int:
        """Appends delta KV tensors to an active session cache on the Context Server."""
        payload = _serialize_kv_pair(k_delta, v_delta)
        meta: dict[str, Any] = {"session_id": session_id, "layer_idx": layer_idx}
        if token_ids is not None:
            meta["token_ids"] = token_ids
        if num_layers is not None:
            meta["num_layers"] = num_layers
        if upload_id is not None:
            meta["upload_id"] = upload_id

        resp_meta, _ = await self._exchange_async(
            MSG_APPEND_KV_REQ, MSG_APPEND_KV_RESP, meta, payload
        )
        if resp_meta.get("status") != "ok":
            raise RuntimeError(
                f"Context Server error: {resp_meta.get('error', 'unknown error')}"
            )
        return int(resp_meta.get("new_len", 0))

    def append_kv_sync(
        self,
        session_id: str,
        layer_idx: int,
        k_delta: torch.Tensor,
        v_delta: torch.Tensor,
        token_ids: list[int] | None = None,
        num_layers: int | None = None,
        upload_id: str | None = None,
    ) -> int:
        """Synchronously appends delta KV tensors over TCP."""
        payload = _serialize_kv_pair(k_delta, v_delta)
        meta: dict[str, Any] = {"session_id": session_id, "layer_idx": layer_idx}
        if token_ids is not None:
            meta["token_ids"] = token_ids
        if num_layers is not None:
            meta["num_layers"] = num_layers
        if upload_id is not None:
            meta["upload_id"] = upload_id

        resp_meta, _ = self._exchange_sync(
            MSG_APPEND_KV_REQ, MSG_APPEND_KV_RESP, meta, payload
        )
        if resp_meta.get("status") != "ok":
            raise RuntimeError(
                f"Context Server error: {resp_meta.get('error', 'unknown error')}"
            )
        return int(resp_meta.get("new_len", 0))

    def close_sync(self) -> None:
        """Synchronously closes the persistent TCP socket."""
        with self._sync_lock:
            self._sync_in_flight = None
            if self._sync_sock is not None:
                with contextlib.suppress(Exception):
                    self._sync_sock.close()
                self._sync_sock = None

    def register_prompt_sync(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        token_ids: list[int] | None = None,
        num_layers: int | None = None,
        upload_id: str | None = None,
    ) -> bool:
        """Synchronously registers static prompt KV cache for a layer over TCP."""
        payload = _serialize_kv_pair(k, v)
        meta: dict[str, Any] = {"session_id": session_id, "layer_idx": layer_idx}
        if token_ids is not None:
            meta["token_ids"] = token_ids
        if num_layers is not None:
            meta["num_layers"] = num_layers
        if upload_id is not None:
            meta["upload_id"] = upload_id

        resp_meta, _ = self._exchange_sync(
            MSG_REGISTER_PROMPT_REQ, MSG_REGISTER_PROMPT_RESP, meta, payload
        )
        return resp_meta.get("status") == "ok"

    async def match_prefix(
        self,
        token_ids: list[int],
        session_id: str | None = None,
    ) -> tuple[int, str | None]:
        """Queries the Context Server for the longest matching cached token prefix."""
        meta: dict[str, Any] = {"token_ids": token_ids}
        if session_id is not None:
            meta["session_id"] = session_id

        resp_meta, _ = await self._exchange_async(
            MSG_MATCH_PREFIX_REQ, MSG_MATCH_PREFIX_RESP, meta, b""
        )
        if resp_meta.get("status") != "ok":
            raise RuntimeError(
                f"Context Server error: {resp_meta.get('error', 'unknown error')}"
            )
        return resp_meta.get("matched_len", 0), resp_meta.get("session_id")

    def match_prefix_sync(
        self,
        token_ids: list[int],
        session_id: str | None = None,
    ) -> tuple[int, str | None]:
        """Synchronously queries the Context Server for matching cached token prefix."""
        meta: dict[str, Any] = {"token_ids": token_ids}
        if session_id is not None:
            meta["session_id"] = session_id

        resp_meta, _ = self._exchange_sync(
            MSG_MATCH_PREFIX_REQ, MSG_MATCH_PREFIX_RESP, meta, b""
        )
        if resp_meta.get("status") != "ok":
            raise RuntimeError(
                f"Context Server error: {resp_meta.get('error', 'unknown error')}"
            )
        return resp_meta.get("matched_len", 0), resp_meta.get("session_id")

    def release_session_sync(self, session_id: str) -> bool:
        """Synchronously releases the prompt KV cache on the remote Context Server."""
        resp_meta, _ = self._exchange_sync(
            MSG_RELEASE_SESSION_REQ,
            MSG_RELEASE_SESSION_RESP,
            {"session_id": session_id},
            b"",
        )
        return bool(resp_meta.get("evicted", False))

    async def release_session(self, session_id: str) -> bool:
        """Releases the prompt KV cache on the remote Context Server."""
        resp_meta, _ = await self._exchange_async(
            MSG_RELEASE_SESSION_REQ,
            MSG_RELEASE_SESSION_RESP,
            {"session_id": session_id},
            b"",
        )
        return bool(resp_meta.get("evicted", False))


def main() -> None:
    """CLI entrypoint for starting a standalone Disaggregated Context Server."""
    parser = argparse.ArgumentParser(
        description="FrontierSplit Disaggregated Context Server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host address")
    parser.add_argument("--port", type=int, default=50055, help="Bind TCP port")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--max-cached-tokens",
        type=int,
        default=1_000_000,
        help="Maximum tokens to cache in the Radix Prefix Tree",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=float,
        default=600.0,
        help="Sliding TTL in seconds for active sessions",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = ContextServer(
        device=args.device,
        max_cached_tokens=args.max_cached_tokens,
        session_ttl_seconds=args.ttl_seconds,
    )

    async def _serve() -> None:
        srv = await server.start_server(host=args.host, port=args.port)
        logger.info(
            "ContextServer listening on %s:%d (device=%s)",
            args.host,
            args.port,
            args.device,
        )
        async with srv:
            await srv.serve_forever()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_serve())


if __name__ == "__main__":
    main()


__all__ = [
    "ContextClient",
    "ContextServer",
    "ContextStore",
    "deserialize_chunk",
    "deserialize_tensor",
    "main",
    "pack_context_frame",
    "read_context_frame_async",
    "read_context_frame_sync",
    "serialize_chunk",
    "serialize_tensor",
]
