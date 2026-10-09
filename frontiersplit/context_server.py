"""Disaggregated Context Server for FrontierSplit.

Stores static, immutable prompt Key-Value (KV) caches and computes partial
attention on demand for decode workers via Online Softmax.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import socket
import struct
import threading
from typing import Any

import torch

from frontiersplit.online_softmax import (
    PartialAttentionChunk,
    compute_partial_attention,
)

logger = logging.getLogger("frontiersplit.context_server")

# Packet type identifiers for Context Server binary protocol
MSG_REGISTER_PROMPT_REQ = 0x20
MSG_REGISTER_PROMPT_RESP = 0x21
MSG_QUERY_PARTIAL_REQ = 0x22
MSG_QUERY_PARTIAL_RESP = 0x23
MSG_RELEASE_SESSION_REQ = 0x24
MSG_RELEASE_SESSION_RESP = 0x25

# 16-byte fixed binary header: MAGIC (2B), msg_type (1B), flags (1B), meta_len (4B), payload_len (8B)
CONTEXT_HEADER_FORMAT = "<2sBB I Q"
CONTEXT_HEADER_SIZE = struct.calcsize(CONTEXT_HEADER_FORMAT)
CONTEXT_MAGIC = b"FC"


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
    """Serializes a contiguous torch tensor to raw bytes using PyTorch."""
    buf = io.BytesIO()
    torch.save(tensor.contiguous(), buf)
    return buf.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserializes raw bytes back into a torch tensor."""
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)


def serialize_chunk(chunk: PartialAttentionChunk) -> bytes:
    """Serializes a PartialAttentionChunk to raw bytes."""
    buf = io.BytesIO()
    torch.save(
        {
            "acc": chunk.accumulator.contiguous(),
            "max": chunk.max_score.contiguous(),
            "sum": chunk.sum_exp.contiguous(),
        },
        buf,
    )
    return buf.getvalue()


def deserialize_chunk(data: bytes) -> PartialAttentionChunk:
    """Deserializes raw bytes back into a PartialAttentionChunk."""
    buf = io.BytesIO(data)
    d = torch.load(buf, weights_only=True)
    return PartialAttentionChunk(
        accumulator=d["acc"],
        max_score=d["max"],
        sum_exp=d["sum"],
    )


class ContextStore:
    """Thread-safe in-memory storage for static prompt Key-Value tensors."""

    def __init__(self) -> None:
        # Structure: {session_id: {layer_idx: (k_prompt, v_prompt)}}
        self._store: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    def register_prompt(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Stores static prompt keys and values for a given session and layer."""
        if session_id not in self._store:
            self._store[session_id] = {}
        self._store[session_id][layer_idx] = (k.detach(), v.detach())

    def get_prompt(
        self,
        session_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Retrieves stored prompt keys and values, or None if not found."""
        session = self._store.get(session_id)
        if session is None:
            return None
        return session.get(layer_idx)

    def has_session(self, session_id: str) -> bool:
        """Returns True if the session has registered prompt caches."""
        return session_id in self._store

    def release_session(self, session_id: str) -> bool:
        """Evicts the prompt KV cache for a completed session."""
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    @property
    def num_sessions(self) -> int:
        """Returns the number of active sessions stored in memory."""
        return len(self._store)

    def clear(self) -> None:
        """Clears all stored sessions."""
        self._store.clear()


class ContextServer:
    """Server that computes partial attention against stored prompt KV caches."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.store = ContextStore()
        self._server: asyncio.Server | None = None
        self._thread: threading.Thread | None = None
        self._thread_loop: asyncio.AbstractEventLoop | None = None
        self._active_tasks: set[asyncio.Task[Any]] = set()

    def register_prompt(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Registers a static prompt KV cache in memory."""
        self.store.register_prompt(
            session_id=session_id,
            layer_idx=layer_idx,
            k=k.to(self.device),
            v=v.to(self.device),
        )

    def query_partial_attention(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,
        scale: float | None = None,
    ) -> PartialAttentionChunk:
        """Computes partial attention for a given query against the stored prompt."""
        kv = self.store.get_prompt(session_id, layer_idx)
        if kv is None:
            raise KeyError(
                f"No prompt KV cache found for session '{session_id}', layer {layer_idx}"
            )
        k, v = kv
        q_dev = q.to(self.device)
        return compute_partial_attention(q_dev, k, v, scale=scale)

    def release_session(self, session_id: str) -> bool:
        """Evicts a session's static prompt KV cache."""
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
                    q = deserialize_tensor(payload)

                    try:
                        chunk = self.query_partial_attention(
                            session_id, layer_idx, q, scale=scale
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
                            {"status": "error", "error": str(e)},
                            b"",
                        )
                    writer.write(resp)
                    await writer.drain()

                elif msg_type == MSG_REGISTER_PROMPT_REQ:
                    session_id = meta["session_id"]
                    layer_idx = meta["layer_idx"]
                    tensors = torch.load(io.BytesIO(payload), weights_only=True)
                    self.register_prompt(
                        session_id, layer_idx, tensors["k"], tensors["v"]
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

                elif msg_type == MSG_RELEASE_SESSION_REQ:
                    session_id = meta["session_id"]
                    evicted = self.release_session(session_id)
                    resp = pack_context_frame(
                        MSG_RELEASE_SESSION_RESP,
                        {"status": "ok", "evicted": evicted},
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

    async def start_server(
        self, host: str = "127.0.0.1", port: int = 50055
    ) -> asyncio.Server:
        """Starts the asynchronous TCP context service."""
        self._server = await asyncio.start_server(self._handle_connection, host, port)
        return self._server

    async def stop_server(self) -> None:
        """Stops the TCP server and waits for client tasks to finish."""
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
        self._lock = asyncio.Lock()

    def connect_sync(self) -> socket.socket:
        """Establishes or returns a synchronous persistent TCP socket."""
        if self._sync_sock is None or self._sync_sock.fileno() == -1:
            self._sync_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sync_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sync_sock.connect((self.host, self.port))
        return self._sync_sock

    async def connect(self) -> None:
        """Establishes a persistent TCP connection to the Context Server."""
        if self.writer is not None and not self.writer.is_closing():
            return
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        sock = self.writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    async def close(self) -> None:
        """Closes both async and sync TCP connections."""
        if self._sync_sock is not None:
            self._sync_sock.close()
            self._sync_sock = None

        if self.writer is not None:
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
    ) -> bool:
        """Registers a static prompt KV cache on the remote Context Server."""
        async with self._lock:
            await self.connect()
            assert self.writer is not None and self.reader is not None

            buf = io.BytesIO()
            torch.save({"k": k.contiguous(), "v": v.contiguous()}, buf)
            payload = buf.getvalue()

            frame = pack_context_frame(
                MSG_REGISTER_PROMPT_REQ,
                {"session_id": session_id, "layer_idx": layer_idx},
                payload,
            )
            self.writer.write(frame)
            await self.writer.drain()

            msg_type, meta, _ = await read_context_frame_async(self.reader)
            return meta.get("status") == "ok"

    async def query_partial_attention(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,
        scale: float | None = None,
    ) -> PartialAttentionChunk:
        """Queries the remote Context Server to compute partial attention over prompt."""
        async with self._lock:
            await self.connect()
            assert self.writer is not None and self.reader is not None

            payload = serialize_tensor(q)
            frame = pack_context_frame(
                MSG_QUERY_PARTIAL_REQ,
                {"session_id": session_id, "layer_idx": layer_idx, "scale": scale},
                payload,
            )
            self.writer.write(frame)
            await self.writer.drain()

            msg_type, meta, resp_payload = await read_context_frame_async(self.reader)
            if meta.get("status") != "ok":
                raise RuntimeError(
                    f"Context Server error: {meta.get('error', 'unknown error')}"
                )
            return deserialize_chunk(resp_payload)

    def query_partial_attention_sync(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,
        scale: float | None = None,
    ) -> PartialAttentionChunk:
        """Queries the remote Context Server synchronously over raw TCP socket."""
        sock = self.connect_sync()
        payload = serialize_tensor(q)
        frame = pack_context_frame(
            MSG_QUERY_PARTIAL_REQ,
            {"session_id": session_id, "layer_idx": layer_idx, "scale": scale},
            payload,
        )
        sock.sendall(frame)
        msg_type, meta, resp_payload = read_context_frame_sync(sock)
        if meta.get("status") != "ok":
            raise RuntimeError(
                f"Context Server error: {meta.get('error', 'unknown error')}"
            )
        return deserialize_chunk(resp_payload)

    def close_sync(self) -> None:
        """Synchronously closes the persistent TCP socket."""
        if self._sync_sock is not None:
            self._sync_sock.close()
            self._sync_sock = None

    def register_prompt_sync(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        """Synchronously registers static prompt KV cache for a layer over TCP."""
        sock = self.connect_sync()
        buf = io.BytesIO()
        torch.save({"k": k.contiguous(), "v": v.contiguous()}, buf)
        payload = buf.getvalue()

        frame = pack_context_frame(
            MSG_REGISTER_PROMPT_REQ,
            {"session_id": session_id, "layer_idx": layer_idx},
            payload,
        )
        sock.sendall(frame)
        msg_type, meta, _ = read_context_frame_sync(sock)
        return meta.get("status") == "ok"

    def release_session_sync(self, session_id: str) -> bool:
        """Synchronously releases the prompt KV cache on the remote Context Server."""
        sock = self.connect_sync()
        frame = pack_context_frame(
            MSG_RELEASE_SESSION_REQ,
            {"session_id": session_id},
            b"",
        )
        sock.sendall(frame)
        msg_type, meta, _ = read_context_frame_sync(sock)
        return bool(meta.get("evicted", False))

    async def release_session(self, session_id: str) -> bool:
        """Releases the prompt KV cache on the remote Context Server."""
        async with self._lock:
            await self.connect()
            assert self.writer is not None and self.reader is not None

            frame = pack_context_frame(
                MSG_RELEASE_SESSION_REQ,
                {"session_id": session_id},
                b"",
            )
            self.writer.write(frame)
            await self.writer.drain()

            msg_type, meta, _ = await read_context_frame_async(self.reader)
            return bool(meta.get("evicted", False))


__all__ = [
    "ContextClient",
    "ContextServer",
    "ContextStore",
    "deserialize_chunk",
    "deserialize_tensor",
    "pack_context_frame",
    "read_context_frame_async",
    "read_context_frame_sync",
    "serialize_chunk",
    "serialize_tensor",
]
