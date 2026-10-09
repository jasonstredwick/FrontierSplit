"""FrontierSplit Network Protocol: Wire formats and tensor serialization."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

# Binary Protocol Constants
HEADER_SIZE = 32
MAGIC = b"FS"
HEADER_STRUCT = "<2sBB I Q B B 4H 6x"

MSG_FORWARD_BATCHED_REQ = 1
MSG_FORWARD_BATCHED_RESP = 2
MSG_RELEASE_SESSION_REQ = 3
MSG_RELEASE_SESSION_RESP = 4
MSG_PING = 5
MSG_PONG = 6
MSG_FORWARD_ASYNC_REQ = 7
MSG_FORWARD_ACK = 8
MSG_FORWARD_CHUNK_REQ = 9
MSG_FORWARD_CHUNK_RESP = 10

FLAG_IS_PREFILL = 0x01
FLAG_USE_KV_CACHE = 0x02
FLAG_QUANTIZED = 0x04

DTYPE_TO_CODE = {
    "float16": 1,
    "float32": 2,
    "bfloat16": 3,
    "int64": 4,
    "int32": 5,
    "int8": 6,
    "float8_e4m3fn": 7,
}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}


def pack_header(
    msg_type: int,
    flags: int,
    meta_len: int,
    payload_len: int,
    dtype_code: int = 0,
    shape: Optional[List[int]] = None,
) -> bytes:
    """Packs a 32-byte binary frame header with little-endian encoding."""
    shape_4 = [1, 1, 1, 1]
    ndim = 0
    if shape:
        ndim = min(len(shape), 4)
        for i in range(ndim):
            shape_4[i] = min(shape[i], 65535)
    return struct.pack(
        HEADER_STRUCT,
        MAGIC,
        msg_type,
        flags,
        meta_len,
        payload_len,
        dtype_code,
        ndim,
        shape_4[0],
        shape_4[1],
        shape_4[2],
        shape_4[3],
    )


def unpack_header(header_bytes: bytes) -> Dict[str, Any]:
    """Unpacks a 32-byte binary frame header."""
    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"Invalid header size: expected {HEADER_SIZE}, got {len(header_bytes)}")
    magic, msg_type, flags, meta_len, payload_len, dtype_code, ndim, s0, s1, s2, s3 = struct.unpack(
        HEADER_STRUCT, header_bytes
    )
    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes: expected {MAGIC!r}, got {magic!r}")
    shape = [s0, s1, s2, s3][:ndim] if ndim > 0 else []
    return {
        "msg_type": msg_type,
        "flags": flags,
        "meta_len": meta_len,
        "payload_len": payload_len,
        "dtype_code": dtype_code,
        "ndim": ndim,
        "shape": shape,
    }


async def read_binary_frame_async(reader: asyncio.StreamReader) -> Tuple[int, int, Dict[str, Any], bytes, str, List[int]]:
    """Asynchronously reads a full binary frame from an asyncio StreamReader."""
    header_bytes = await reader.readexactly(HEADER_SIZE)
    hdr = unpack_header(header_bytes)
    meta_bytes = await reader.readexactly(hdr["meta_len"]) if hdr["meta_len"] > 0 else b"{}"
    meta = json.loads(meta_bytes.decode("utf-8"))
    payload_bytes = await reader.readexactly(hdr["payload_len"]) if hdr["payload_len"] > 0 else b""
    dtype_str = CODE_TO_DTYPE.get(hdr["dtype_code"], "float32")
    shape = meta.get("tensor_shape", hdr["shape"])
    return hdr["msg_type"], hdr["flags"], meta, payload_bytes, dtype_str, shape


def read_binary_frame_sync(sock: socket.socket) -> Tuple[int, int, Dict[str, Any], bytes, str, List[int]]:
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

    header_bytes = recv_exact(HEADER_SIZE)
    hdr = unpack_header(header_bytes)
    meta_bytes = recv_exact(hdr["meta_len"]) if hdr["meta_len"] > 0 else b"{}"
    meta = json.loads(meta_bytes.decode("utf-8"))
    payload_bytes = recv_exact(hdr["payload_len"]) if hdr["payload_len"] > 0 else b""
    dtype_str = CODE_TO_DTYPE.get(hdr["dtype_code"], "float32")
    shape = meta.get("tensor_shape", hdr["shape"])
    return hdr["msg_type"], hdr["flags"], meta, payload_bytes, dtype_str, shape


class ActivationPacket(BaseModel):
    """The inter-node payload passed along the pipeline chain."""
    request_id: str
    sequence_step: int
    stage_id: int
    is_prefill: bool = False
    use_kv_cache: bool = True
    max_tokens: Optional[int] = None
    tokens: Optional[List[int]] = None
    tensor_shape: List[int] = Field(default_factory=list)
    tensor_dtype: str = "float32"
    tensor_bytes_b64: str = ""
    timestamp_sent_ms: float = Field(default_factory=lambda: time.time() * 1000)
    stage_timings: Dict[str, float] = Field(default_factory=dict)

    def set_tensor(self, arr: np.ndarray) -> None:
        """Serialize a numpy tensor into base64 raw bytes."""
        self.tensor_shape = list(arr.shape)
        self.tensor_dtype = str(arr.dtype)
        raw_bytes = arr.tobytes()
        self.tensor_bytes_b64 = base64.b64encode(raw_bytes).decode("ascii")

    def get_tensor(self) -> np.ndarray:
        """Deserialize raw base64 bytes back into a numpy array."""
        if not self.tensor_bytes_b64:
            return np.zeros(self.tensor_shape, dtype=self.tensor_dtype)
        raw_bytes = base64.b64decode(self.tensor_bytes_b64.encode("ascii"))
        return np.frombuffer(raw_bytes, dtype=self.tensor_dtype).reshape(self.tensor_shape)


class GenerationResponse(BaseModel):
    """Output packet generated by the final node in the pipeline."""
    request_id: str
    token_id: int
    text: str
    is_finished: bool
    latency_ms: float
    stage_timings: Dict[str, float] = Field(default_factory=dict)


class BatchedActivationPacket(BaseModel):
    """Inter-node payload containing multiple requests batched into a single forward pass."""
    request_ids: List[str]
    sequence_steps: List[int]
    stage_id: int
    is_prefill: bool = False
    use_kv_cache: bool = True
    max_tokens_list: Optional[List[int]] = None
    tokens_batch: Optional[List[List[int]]] = None  # Padded tokens per request [B, max_len]
    attention_mask: Optional[List[List[int]]] = None # Attention mask per request [B, max_len]
    tensor_shape: List[int] = Field(default_factory=list) # [B, max_len, hidden_size]
    tensor_dtype: str = "float32"
    tensor_bytes_b64: str = ""
    timestamp_sent_ms: float = Field(default_factory=lambda: time.time() * 1000)
    stage_timings: Dict[str, float] = Field(default_factory=dict)
    reply_to: Optional[str] = None
    tensor_scale: Optional[float] = None

    # Internal cached raw bytes to bypass Base64 encoding/decoding during binary transport
    _raw_tensor_bytes: Optional[bytes] = None

    def set_raw_tensor(self, raw_bytes: bytes, shape: List[int], dtype: str = "float16") -> None:
        """Directly store raw tensor bytes without base64 or type conversion."""
        self._raw_tensor_bytes = raw_bytes
        self.tensor_shape = list(shape)
        self.tensor_dtype = str(dtype)
        self.tensor_bytes_b64 = ""

    def get_raw_bytes(self) -> bytes:
        """Retrieves raw unencoded tensor bytes."""
        if self._raw_tensor_bytes is not None:
            return self._raw_tensor_bytes
        if self.tensor_bytes_b64:
            return base64.b64decode(self.tensor_bytes_b64.encode("ascii"))
        return b""

    def set_tensor(self, arr: Any) -> None:
        """Serialize a numpy array or PyTorch tensor into raw bytes and base64."""
        if hasattr(arr, "contiguous") and hasattr(arr, "cpu"):
            # PyTorch tensor
            self.tensor_shape = list(arr.shape)
            self.tensor_dtype = str(arr.dtype).replace("torch.", "")
            cpu_t = arr.contiguous().cpu()
            # Direct view to avoid type promotion
            try:
                import torch
                raw_bytes = cpu_t.view(torch.uint8).numpy().tobytes()
            except Exception:
                raw_bytes = cpu_t.numpy().tobytes()
        else:
            self.tensor_shape = list(arr.shape)
            self.tensor_dtype = str(arr.dtype)
            raw_bytes = arr.tobytes()

        self._raw_tensor_bytes = raw_bytes
        self.tensor_bytes_b64 = base64.b64encode(raw_bytes).decode("ascii")

    def get_tensor(self) -> np.ndarray:
        """Deserialize raw tensor bytes into a numpy array, dequantizing if scale is present."""
        raw_bytes = self.get_raw_bytes()
        if not raw_bytes:
            return np.zeros(self.tensor_shape, dtype=self.tensor_dtype)
        if self.tensor_scale is not None and self.tensor_dtype in ("int8", "float8_e4m3fn"):
            q_arr = np.frombuffer(raw_bytes, dtype=np.int8).reshape(self.tensor_shape)
            return (q_arr.astype(np.float32) * self.tensor_scale).astype(np.float32)
        return np.frombuffer(raw_bytes, dtype=self.tensor_dtype).reshape(self.tensor_shape)

    def encode_binary(self, raw_tensor_bytes: Optional[bytes] = None, msg_type: Optional[int] = None) -> bytes:
        """Encodes this packet into a 32-byte framed binary packet with raw FP16/FP32/INT8 payload."""
        payload = raw_tensor_bytes if raw_tensor_bytes is not None else self.get_raw_bytes()
        meta = {
            "request_ids": self.request_ids,
            "sequence_steps": self.sequence_steps,
            "stage_id": self.stage_id,
            "is_prefill": self.is_prefill,
            "use_kv_cache": self.use_kv_cache,
            "max_tokens_list": self.max_tokens_list,
            "tokens_batch": self.tokens_batch,
            "attention_mask": self.attention_mask,
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": self.tensor_dtype,
            "timestamp_sent_ms": self.timestamp_sent_ms,
            "stage_timings": self.stage_timings,
            "reply_to": self.reply_to,
            "tensor_scale": self.tensor_scale,
        }
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        flags = 0
        if self.is_prefill:
            flags |= FLAG_IS_PREFILL
        if self.use_kv_cache:
            flags |= FLAG_USE_KV_CACHE
        if self.tensor_scale is not None:
            flags |= FLAG_QUANTIZED

        dtype_code = DTYPE_TO_CODE.get(self.tensor_dtype, 0)
        resolved_msg_type = msg_type or (MSG_FORWARD_ASYNC_REQ if self.reply_to else MSG_FORWARD_BATCHED_REQ)
        header = pack_header(
            msg_type=resolved_msg_type,
            flags=flags,
            meta_len=len(meta_bytes),
            payload_len=len(payload),
            dtype_code=dtype_code,
            shape=self.tensor_shape,
        )
        return header + meta_bytes + payload

    @classmethod
    def decode_binary(
        cls,
        meta_bytes: bytes,
        payload_bytes: bytes,
        dtype_str: str,
        shape: List[int],
        flags: int = 0,
    ) -> BatchedActivationPacket:
        """Decodes a binary frame into a BatchedActivationPacket."""
        meta = json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {}
        packet = cls(
            request_ids=meta.get("request_ids", []),
            sequence_steps=meta.get("sequence_steps", []),
            stage_id=meta.get("stage_id", 0),
            is_prefill=bool(flags & FLAG_IS_PREFILL) or meta.get("is_prefill", False),
            use_kv_cache=bool(flags & FLAG_USE_KV_CACHE) or meta.get("use_kv_cache", True),
            max_tokens_list=meta.get("max_tokens_list"),
            tokens_batch=meta.get("tokens_batch"),
            attention_mask=meta.get("attention_mask"),
            tensor_shape=meta.get("tensor_shape", shape),
            tensor_dtype=meta.get("tensor_dtype", dtype_str),
            timestamp_sent_ms=meta.get("timestamp_sent_ms", time.time() * 1000),
            stage_timings=meta.get("stage_timings", {}),
            reply_to=meta.get("reply_to"),
            tensor_scale=meta.get("tensor_scale"),
        )
        packet.set_raw_tensor(payload_bytes, packet.tensor_shape, packet.tensor_dtype)
        return packet

    @classmethod
    def from_binary_frame(cls, frame_bytes: bytes) -> BatchedActivationPacket:
        """Parses a complete framed binary message into a BatchedActivationPacket."""
        if len(frame_bytes) < HEADER_SIZE:
            raise ValueError(f"Frame length {len(frame_bytes)} is less than header size {HEADER_SIZE}")
        hdr = unpack_header(frame_bytes[:HEADER_SIZE])
        meta_end = HEADER_SIZE + hdr["meta_len"]
        meta_bytes = frame_bytes[HEADER_SIZE:meta_end]
        payload_bytes = frame_bytes[meta_end : meta_end + hdr["payload_len"]]
        dtype_str = CODE_TO_DTYPE.get(hdr["dtype_code"], "float32")
        shape = hdr["shape"]
        return cls.decode_binary(meta_bytes, payload_bytes, dtype_str, shape, flags=hdr["flags"])


class BatchedGenerationResponse(BaseModel):
    """Output packet generated by the final node containing responses for all requests in batch."""
    responses: List[GenerationResponse]
    batch_size: int
    stage_timings: Dict[str, float] = Field(default_factory=dict)

    def encode_binary(self) -> bytes:
        """Encodes this response into a binary frame."""
        meta = {
            "responses": [r.model_dump() for r in self.responses],
            "batch_size": self.batch_size,
            "stage_timings": self.stage_timings,
        }
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        header = pack_header(
            msg_type=MSG_FORWARD_BATCHED_RESP,
            flags=0,
            meta_len=len(meta_bytes),
            payload_len=0,
            dtype_code=0,
            shape=None,
        )
        return header + meta_bytes

    @classmethod
    def decode_binary(cls, meta_bytes: bytes) -> BatchedGenerationResponse:
        """Decodes binary frame meta into BatchedGenerationResponse."""
        meta = json.loads(meta_bytes.decode("utf-8"))
        responses = [GenerationResponse(**r) for r in meta.get("responses", [])]
        return cls(
            responses=responses,
            batch_size=meta.get("batch_size", len(responses)),
            stage_timings=meta.get("stage_timings", {}),
        )


class ReleaseSessionPacket(BaseModel):
    """Inter-node command packet sent to workers to immediately reclaim session KV caches."""
    request_ids: List[str]

    def encode_binary(self) -> bytes:
        meta = {"request_ids": self.request_ids}
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        header = pack_header(
            msg_type=MSG_RELEASE_SESSION_REQ,
            flags=0,
            meta_len=len(meta_bytes),
            payload_len=0,
        )
        return header + meta_bytes

    @classmethod
    def decode_binary(cls, meta_bytes: bytes) -> ReleaseSessionPacket:
        meta = json.loads(meta_bytes.decode("utf-8"))
        return cls(request_ids=meta.get("request_ids", []))


class ReleaseSessionResponse(BaseModel):
    """Response acknowledging session KV cache reclamation across pipeline stages."""
    status: str = "ok"
    released_count: int
    active_sessions: int

    def encode_binary(self) -> bytes:
        meta = {
            "status": self.status,
            "released_count": self.released_count,
            "active_sessions": self.active_sessions,
        }
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        header = pack_header(
            msg_type=MSG_RELEASE_SESSION_RESP,
            flags=0,
            meta_len=len(meta_bytes),
            payload_len=0,
        )
        return header + meta_bytes

    @classmethod
    def decode_binary(cls, meta_bytes: bytes) -> ReleaseSessionResponse:
        meta = json.loads(meta_bytes.decode("utf-8"))
        return cls(
            status=meta.get("status", "ok"),
            released_count=meta.get("released_count", 0),
            active_sessions=meta.get("active_sessions", 0),
        )


class ChunkActivationPacket(BaseModel):
    """Network packet representing an atomic chunk of tokens of fixed size C (default 16).
    Shape of hidden_states is always strictly [chunk_size, hidden_size] with ZERO variable batch dimension B.
    """
    request_id: str
    chunk_idx: int = 0          # Chunk index d in [0 .. D-1]
    total_chunks: int = 1       # D = ceil(S / C)
    chunk_size: int = 16        # C = 16
    valid_tokens: int = 16      # Number of valid tokens in chunk (1..C)
    is_prefill: bool = True
    stage_id: int = 0
    tokens: Optional[List[int]] = None   # Token IDs of length C (passed to Stage 0 for embedding)
    tensor_shape: List[int] = Field(default_factory=lambda: [16, 4096])
    tensor_dtype: str = "float16"
    tensor_bytes_b64: Optional[str] = None
    timestamp_sent_ms: float = Field(default_factory=lambda: time.time() * 1000)
    stage_timings: Dict[str, float] = Field(default_factory=dict)
    reply_to: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 128
    _raw_tensor_bytes: Optional[bytes] = None

    def set_raw_tensor(self, raw_bytes: bytes, shape: List[int], dtype_str: str = "float16") -> None:
        self._raw_tensor_bytes = raw_bytes
        self.tensor_shape = shape
        self.tensor_dtype = dtype_str

    def get_raw_bytes(self) -> bytes:
        if self._raw_tensor_bytes is not None:
            return self._raw_tensor_bytes
        if self.tensor_bytes_b64:
            return base64.b64decode(self.tensor_bytes_b64.encode("ascii"))
        return b""

    def set_tensor(self, arr: Any) -> None:
        if hasattr(arr, "contiguous") and hasattr(arr, "cpu"):
            self.tensor_shape = list(arr.shape)
            self.tensor_dtype = str(arr.dtype).replace("torch.", "")
            cpu_t = arr.contiguous().cpu()
            try:
                import torch
                raw_bytes = cpu_t.view(torch.uint8).numpy().tobytes()
            except Exception:
                raw_bytes = cpu_t.numpy().tobytes()
        else:
            self.tensor_shape = list(arr.shape)
            self.tensor_dtype = str(arr.dtype)
            raw_bytes = arr.tobytes()

        self._raw_tensor_bytes = raw_bytes
        self.tensor_bytes_b64 = base64.b64encode(raw_bytes).decode("ascii")

    def get_tensor(self) -> np.ndarray:
        raw_bytes = self.get_raw_bytes()
        if not raw_bytes:
            return np.zeros(self.tensor_shape, dtype=self.tensor_dtype)
        return np.frombuffer(raw_bytes, dtype=self.tensor_dtype).reshape(self.tensor_shape)

    def encode_binary(self, raw_tensor_bytes: Optional[bytes] = None, msg_type: Optional[int] = None) -> bytes:
        payload = raw_tensor_bytes if raw_tensor_bytes is not None else self.get_raw_bytes()
        meta = {
            "request_id": self.request_id,
            "chunk_idx": self.chunk_idx,
            "total_chunks": self.total_chunks,
            "chunk_size": self.chunk_size,
            "valid_tokens": self.valid_tokens,
            "is_prefill": self.is_prefill,
            "stage_id": self.stage_id,
            "tokens": self.tokens,
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": self.tensor_dtype,
            "timestamp_sent_ms": self.timestamp_sent_ms,
            "stage_timings": self.stage_timings,
            "reply_to": self.reply_to,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        flags = 0
        if self.is_prefill:
            flags |= FLAG_IS_PREFILL
        flags |= FLAG_USE_KV_CACHE

        dtype_code = DTYPE_TO_CODE.get(self.tensor_dtype, 1)
        resolved_msg_type = msg_type or MSG_FORWARD_CHUNK_REQ
        header = pack_header(
            msg_type=resolved_msg_type,
            flags=flags,
            meta_len=len(meta_bytes),
            payload_len=len(payload),
            dtype_code=dtype_code,
            shape=self.tensor_shape,
        )
        return header + meta_bytes + payload

    @classmethod
    def decode_binary(
        cls,
        meta_bytes: bytes,
        payload_bytes: bytes,
        dtype_str: str,
        shape: List[int],
        flags: int = 0,
    ) -> ChunkActivationPacket:
        meta = json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {}
        packet = cls(
            request_id=meta.get("request_id", ""),
            chunk_idx=meta.get("chunk_idx", 0),
            total_chunks=meta.get("total_chunks", 1),
            chunk_size=meta.get("chunk_size", 16),
            valid_tokens=meta.get("valid_tokens", 16),
            is_prefill=bool(flags & FLAG_IS_PREFILL) or meta.get("is_prefill", True),
            stage_id=meta.get("stage_id", 0),
            tokens=meta.get("tokens"),
            tensor_shape=meta.get("tensor_shape", shape),
            tensor_dtype=meta.get("tensor_dtype", dtype_str),
            timestamp_sent_ms=meta.get("timestamp_sent_ms", time.time() * 1000),
            stage_timings=meta.get("stage_timings", {}),
            reply_to=meta.get("reply_to"),
            temperature=meta.get("temperature", 0.0),
            max_tokens=meta.get("max_tokens", 128),
        )
        packet.set_raw_tensor(payload_bytes, packet.tensor_shape, packet.tensor_dtype)
        return packet

    @classmethod
    def from_binary_frame(cls, frame_bytes: bytes) -> ChunkActivationPacket:
        if len(frame_bytes) < HEADER_SIZE:
            raise ValueError(f"Frame length {len(frame_bytes)} is less than header size {HEADER_SIZE}")
        hdr = unpack_header(frame_bytes[:HEADER_SIZE])
        meta_end = HEADER_SIZE + hdr["meta_len"]
        meta_bytes = frame_bytes[HEADER_SIZE:meta_end]
        payload_bytes = frame_bytes[meta_end : meta_end + hdr["payload_len"]]
        dtype_str = CODE_TO_DTYPE.get(hdr["dtype_code"], "float16")
        shape = hdr["shape"]
        return cls.decode_binary(meta_bytes, payload_bytes, dtype_str, shape, flags=hdr["flags"])


class ChunkGenerationResponse(BaseModel):
    """Output packet generated by the final stage for a chunk step."""
    request_id: str
    chunk_idx: int
    is_final_chunk: bool
    next_token_id: Optional[int] = None
    is_finished: bool = False
    finish_reason: Optional[str] = None
    stage_timings: Dict[str, float] = Field(default_factory=dict)

    def encode_binary(self) -> bytes:
        meta = {
            "request_id": self.request_id,
            "chunk_idx": self.chunk_idx,
            "is_final_chunk": self.is_final_chunk,
            "next_token_id": self.next_token_id,
            "is_finished": self.is_finished,
            "finish_reason": self.finish_reason,
            "stage_timings": self.stage_timings,
        }
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        header = pack_header(
            msg_type=MSG_FORWARD_CHUNK_RESP,
            flags=0,
            meta_len=len(meta_bytes),
            payload_len=0,
            dtype_code=0,
            shape=None,
        )
        return header + meta_bytes

    @classmethod
    def decode_binary(cls, meta_bytes: bytes) -> ChunkGenerationResponse:
        meta = json.loads(meta_bytes.decode("utf-8"))
        return cls(
            request_id=meta.get("request_id", ""),
            chunk_idx=meta.get("chunk_idx", 0),
            is_final_chunk=meta.get("is_final_chunk", False),
            next_token_id=meta.get("next_token_id"),
            is_finished=meta.get("is_finished", False),
            finish_reason=meta.get("finish_reason"),
            stage_timings=meta.get("stage_timings", {}),
        )

