"""FrontierSplit Persistent Binary TCP Transport Layer.

Provides high-performance persistent TCP streaming for inter-node tensor handoffs,
eliminating HTTP/JSON/Base64 overhead, disabling Nagle's algorithm (TCP_NODELAY),
and offering automatic reconnection with peer failover.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from frontiersplit.protocol import (
    HEADER_SIZE,
    MAGIC,
    MSG_FORWARD_BATCHED_REQ,
    MSG_FORWARD_BATCHED_RESP,
    MSG_PING,
    MSG_PONG,
    MSG_FORWARD_ASYNC_REQ,
    MSG_FORWARD_ACK,
    MSG_RELEASE_SESSION_REQ,
    MSG_RELEASE_SESSION_RESP,
    BatchedActivationPacket,
    BatchedGenerationResponse,
    ReleaseSessionPacket,
    ReleaseSessionResponse,
    pack_header,
    read_binary_frame_async,
)

logger = logging.getLogger("frontiersplit.transport")


class BinaryTransportClient:
    """Persistent TCP client for transmitting binary activation frames to downstream nodes."""

    def __init__(
        self,
        peer_addresses: Union[str, List[str]],
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_ms: float = 50.0,
    ):
        if isinstance(peer_addresses, str):
            # Parse comma-separated or single peer address
            clean = peer_addresses.replace("tcp://", "").strip()
            self.peers = [p.strip() for p in clean.split(",") if p.strip()]
        else:
            self.peers = [p.replace("tcp://", "").strip() for p in peer_addresses if p.strip()]

        if not self.peers:
            raise ValueError("BinaryTransportClient requires at least one peer address")

        self.current_peer_idx: int = 0
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        self.retry_backoff_ms: float = retry_backoff_ms

        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def current_peer(self) -> str:
        return self.peers[self.current_peer_idx]

    def _parse_peer(self, peer: str) -> Tuple[str, int]:
        parts = peer.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 50052
        return host, port

    async def connect(self) -> None:
        """Establishes or verifies a persistent TCP connection to the active peer."""
        if self.writer is not None and not self.writer.is_closing():
            return

        last_error = None
        # Try active peer first, then cycle through available replica peers
        for attempt in range(len(self.peers)):
            peer = self.peers[self.current_peer_idx]
            host, port = self._parse_peer(peer)
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=min(self.timeout, 5.0),
                )
                # Disable Nagle's algorithm for immediate line-rate transmission
                sock = writer.get_extra_info("socket")
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                self.reader = reader
                self.writer = writer
                logger.debug(f"[BinaryClient] Connected persistent TCP socket to {host}:{port}")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"[BinaryClient] Failed to connect to peer {peer}: {e}. Trying failover...")
                self.current_peer_idx = (self.current_peer_idx + 1) % len(self.peers)

        raise ConnectionError(f"Could not connect to any peer in {self.peers}: {last_error}")

    async def close(self) -> None:
        """Closes the active persistent socket connection."""
        if self.writer is not None:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            finally:
                self.writer = None
                self.reader = None

    async def send_batched_forward(
        self,
        packet: BatchedActivationPacket,
        raw_tensor_bytes: Optional[bytes] = None,
    ) -> BatchedGenerationResponse:
        """Transmits a batched forward activation packet over persistent TCP and awaits the response."""
        frame = packet.encode_binary(raw_tensor_bytes=raw_tensor_bytes)

        async with self._lock:
            for attempt in range(self.max_retries):
                try:
                    await self.connect()
                    assert self.writer is not None and self.reader is not None

                    # Write frame and flush directly to socket
                    self.writer.write(frame)
                    await self.writer.drain()

                    # Await response frame
                    msg_type, flags, meta, payload, dtype_str, shape = await asyncio.wait_for(
                        read_binary_frame_async(self.reader),
                        timeout=self.timeout,
                    )

                    if msg_type == MSG_FORWARD_BATCHED_RESP:
                        return BatchedGenerationResponse.decode_binary(json.dumps(meta).encode("utf-8"))
                    else:
                        raise RuntimeError(f"Unexpected response message type from peer: {msg_type}")

                except (ConnectionError, asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError) as e:
                    logger.warning(
                        f"[BinaryClient] Socket error on attempt {attempt + 1}/{self.max_retries} to {self.current_peer}: {e}"
                    )
                    await self.close()
                    # Rotate peer in case primary crashed
                    if len(self.peers) > 1:
                        self.current_peer_idx = (self.current_peer_idx + 1) % len(self.peers)

                    if attempt < self.max_retries - 1:
                        backoff = (self.retry_backoff_ms / 1000.0) * (2 ** attempt)
                        await asyncio.sleep(backoff)
                    else:
                        raise ConnectionError(f"Persistent binary transmission failed after {self.max_retries} attempts: {e}")

        raise RuntimeError("Unreachable send_batched_forward completion")

    async def send_async_forward(
        self,
        packet: BatchedActivationPacket,
        raw_tensor_bytes: Optional[bytes] = None,
    ) -> None:
        """Transmits a batched forward packet asynchronously, awaits immediate ACK, and unblocks instantly without waiting for downstream pipeline unwinding."""
        frame = packet.encode_binary(raw_tensor_bytes=raw_tensor_bytes, msg_type=MSG_FORWARD_ASYNC_REQ)

        async with self._lock:
            for attempt in range(self.max_retries):
                try:
                    await self.connect()
                    assert self.writer is not None and self.reader is not None

                    # Write frame and flush directly to socket
                    self.writer.write(frame)
                    await self.writer.drain()

                    # Await immediate 32-byte ACK frame (<0.1 ms)
                    msg_type, flags, meta, payload, dtype_str, shape = await asyncio.wait_for(
                        read_binary_frame_async(self.reader),
                        timeout=min(self.timeout, 10.0),
                    )

                    if msg_type == MSG_FORWARD_ACK:
                        return
                    else:
                        raise RuntimeError(f"Expected MSG_FORWARD_ACK (8), got {msg_type}")

                except (ConnectionError, asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError) as e:
                    logger.warning(
                        f"[BinaryClient] Socket error on send_async_forward attempt {attempt + 1}/{self.max_retries} to {self.current_peer}: {e}"
                    )
                    await self.close()
                    if len(self.peers) > 1:
                        self.current_peer_idx = (self.current_peer_idx + 1) % len(self.peers)

                    if attempt < self.max_retries - 1:
                        backoff = (self.retry_backoff_ms / 1000.0) * (2 ** attempt)
                        await asyncio.sleep(backoff)
                    else:
                        raise ConnectionError(f"Persistent binary async transmission failed after {self.max_retries} attempts: {e}")

        raise RuntimeError("Unreachable send_async_forward completion")

    async def send_response_frame(self, response: BatchedGenerationResponse) -> None:
        """Transmits a completed batch generation response frame directly to the gateway / reply listener."""
        frame = response.encode_binary()
        async with self._lock:
            for attempt in range(self.max_retries):
                try:
                    await self.connect()
                    assert self.writer is not None
                    self.writer.write(frame)
                    await self.writer.drain()
                    return
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    await self.close()
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep((self.retry_backoff_ms / 1000.0) * (2 ** attempt))
                    else:
                        raise ConnectionError(f"Failed to transmit generation response to reply peer {self.current_peer}: {e}")

    async def send_release_sessions(self, request_ids: List[str]) -> ReleaseSessionResponse:
        """Sends session KV cache release command down the pipeline over persistent TCP."""
        packet = ReleaseSessionPacket(request_ids=request_ids)
        frame = packet.encode_binary()

        async with self._lock:
            for attempt in range(self.max_retries):
                try:
                    await self.connect()
                    assert self.writer is not None and self.reader is not None

                    self.writer.write(frame)
                    await self.writer.drain()

                    msg_type, flags, meta, payload, dtype_str, shape = await asyncio.wait_for(
                        read_binary_frame_async(self.reader),
                        timeout=self.timeout,
                    )

                    if msg_type == MSG_RELEASE_SESSION_RESP:
                        return ReleaseSessionResponse.decode_binary(json.dumps(meta).encode("utf-8"))
                    else:
                        raise RuntimeError(f"Unexpected response to release session: {msg_type}")

                except (ConnectionError, asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError) as e:
                    await self.close()
                    if attempt < self.max_retries - 1:
                        backoff = (self.retry_backoff_ms / 1000.0) * (2 ** attempt)
                        await asyncio.sleep(backoff)
                    else:
                        raise ConnectionError(f"Persistent binary session release failed after {self.max_retries} attempts: {e}")

        raise RuntimeError("Unreachable send_release_sessions completion")


class BinaryTransportServer:
    """High-throughput asynchronous TCP server handling incoming binary activation frames."""

    def __init__(
        self,
        host: str,
        port: int,
        forward_handler: Optional[Callable[[BatchedActivationPacket], Awaitable[Any]]] = None,
        release_handler: Optional[Callable[[ReleaseSessionPacket], Awaitable[ReleaseSessionResponse]]] = None,
        response_handler: Optional[Callable[[BatchedGenerationResponse], Awaitable[None]]] = None,
    ):
        self.host = host
        self.port = port
        self.forward_handler = forward_handler
        self.release_handler = release_handler
        self.response_handler = response_handler
        self.server: Optional[asyncio.Server] = None
        self.is_running: bool = False
        self._active_writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        """Starts the persistent TCP server."""
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            reuse_address=True,
        )
        self.is_running = True
        if self.server.sockets:
            self.port = self.server.sockets[0].getsockname()[1]
        logger.info(f"[BinaryServer] Persistent TCP server listening on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stops the TCP server and closes all listening sockets and active client connections."""
        self.is_running = False
        for writer in list(self._active_writers):
            try:
                writer.close()
            except Exception:
                pass
        self._active_writers.clear()

        if self.server is not None:
            if hasattr(self.server, "close_clients"):
                self.server.close_clients()
            self.server.close()
            try:
                await asyncio.wait_for(self.server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            self.server = None
            logger.info(f"[BinaryServer] Persistent TCP server stopped on {self.host}:{self.port}")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handles an ongoing persistent client connection stream."""
        self._active_writers.add(writer)
        # Enable TCP_NODELAY on the incoming connection
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        peer_name = writer.get_extra_info("peername")
        logger.debug(f"[BinaryServer] Client connected: {peer_name}")

        try:
            while self.is_running and not reader.at_eof():
                try:
                    msg_type, flags, meta, payload, dtype_str, shape = await read_binary_frame_async(reader)
                except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
                    # Client disconnected normally or socket closed
                    break

                if msg_type == MSG_FORWARD_BATCHED_REQ:
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    packet = BatchedActivationPacket.decode_binary(
                        meta_bytes=meta_bytes,
                        payload_bytes=payload,
                        dtype_str=dtype_str,
                        shape=shape,
                        flags=flags,
                    )

                    # Execute pipeline stage processing
                    if self.forward_handler is not None:
                        response = await self.forward_handler(packet)
                        if isinstance(response, BatchedGenerationResponse):
                            resp_frame = response.encode_binary()
                            writer.write(resp_frame)
                            await writer.drain()

                elif msg_type == MSG_FORWARD_ASYNC_REQ:
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    packet = BatchedActivationPacket.decode_binary(
                        meta_bytes=meta_bytes,
                        payload_bytes=payload,
                        dtype_str=dtype_str,
                        shape=shape,
                        flags=flags,
                    )
                    # Immediate acknowledgment back to sender so sender unblocks in <0.1 ms!
                    ack_header = pack_header(
                        msg_type=MSG_FORWARD_ACK,
                        flags=0,
                        meta_len=0,
                        payload_len=0,
                    )
                    writer.write(ack_header)
                    await writer.drain()

                    # Execute forward pass asynchronously in the background
                    if self.forward_handler is not None:
                        asyncio.create_task(self.forward_handler(packet))

                elif msg_type == MSG_FORWARD_BATCHED_RESP:
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    resp = BatchedGenerationResponse.decode_binary(meta_bytes)
                    if self.response_handler is not None:
                        asyncio.create_task(self.response_handler(resp))

                elif msg_type == MSG_RELEASE_SESSION_REQ:
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    packet = ReleaseSessionPacket.decode_binary(meta_bytes=meta_bytes)
                    if self.release_handler is not None:
                        resp = await self.release_handler(packet)
                    else:
                        resp = ReleaseSessionResponse(status="ok", released_count=len(packet.request_ids), active_sessions=0)
                    resp_frame = resp.encode_binary()
                    writer.write(resp_frame)
                    await writer.drain()

                elif msg_type == MSG_PING:
                    pong = pack_header(MSG_PONG, flags=0, meta_len=0, payload_len=0)
                    writer.write(pong)
                    await writer.drain()

        except Exception as e:
            logger.error(f"[BinaryServer] Error in client handler ({peer_name}): {e}", exc_info=True)
        finally:
            self._active_writers.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug(f"[BinaryServer] Client disconnected: {peer_name}")
