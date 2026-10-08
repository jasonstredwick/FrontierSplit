"""Unit tests for FrontierSplit Persistent Binary TCP Transport and Framing."""

import asyncio
import json
import socket
import unittest
import numpy as np

from frontiersplit.protocol import (
    HEADER_SIZE,
    MAGIC,
    MSG_FORWARD_BATCHED_REQ,
    MSG_FORWARD_BATCHED_RESP,
    BatchedActivationPacket,
    BatchedGenerationResponse,
    GenerationResponse,
    ReleaseSessionPacket,
    ReleaseSessionResponse,
    pack_header,
    unpack_header,
)
from frontiersplit.transport import BinaryTransportClient, BinaryTransportServer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestBinaryFraming(unittest.TestCase):
    def test_header_pack_unpack_exact(self):
        """Verify 32-byte header pack and unpack round-trip."""
        shape = [2, 1, 4096]
        hdr = pack_header(
            msg_type=MSG_FORWARD_BATCHED_REQ,
            flags=3,
            meta_len=145,
            payload_len=16384,
            dtype_code=1,  # float16
            shape=shape,
        )
        self.assertEqual(len(hdr), HEADER_SIZE)

        parsed = unpack_header(hdr)
        self.assertEqual(parsed["msg_type"], MSG_FORWARD_BATCHED_REQ)
        self.assertEqual(parsed["flags"], 3)
        self.assertEqual(parsed["meta_len"], 145)
        self.assertEqual(parsed["payload_len"], 16384)
        self.assertEqual(parsed["dtype_code"], 1)
        self.assertEqual(parsed["shape"], shape)

    def test_packet_fp16_tensor_preservation(self):
        """Verify that native FP16 tensor bytes are preserved without type conversion."""
        raw_arr = np.random.randn(2, 4, 128).astype(np.float16)
        raw_bytes = raw_arr.tobytes()

        packet = BatchedActivationPacket(
            request_ids=["req-1", "req-2"],
            sequence_steps=[1, 1],
            stage_id=1,
            is_prefill=True,
            use_kv_cache=True,
            tensor_shape=list(raw_arr.shape),
            tensor_dtype="float16",
        )
        packet.set_raw_tensor(raw_bytes, shape=list(raw_arr.shape), dtype="float16")

        # Encode to binary frame
        frame = packet.encode_binary()
        self.assertGreater(len(frame), len(raw_bytes))

        # Decode from binary frame
        hdr = unpack_header(frame[:HEADER_SIZE])
        meta_bytes = frame[HEADER_SIZE : HEADER_SIZE + hdr["meta_len"]]
        payload_bytes = frame[HEADER_SIZE + hdr["meta_len"] : HEADER_SIZE + hdr["meta_len"] + hdr["payload_len"]]

        decoded = BatchedActivationPacket.decode_binary(
            meta_bytes=meta_bytes,
            payload_bytes=payload_bytes,
            dtype_str="float16",
            shape=hdr["shape"],
            flags=hdr["flags"],
        )

        self.assertEqual(decoded.request_ids, ["req-1", "req-2"])
        self.assertTrue(decoded.is_prefill)
        self.assertTrue(decoded.use_kv_cache)
        self.assertEqual(decoded.tensor_shape, [2, 4, 128])
        self.assertEqual(decoded.tensor_dtype, "float16")

        # Check exact byte-level tensor preservation
        recovered_arr = decoded.get_tensor()
        self.assertEqual(recovered_arr.dtype, np.float16)
        self.assertTrue(np.array_equal(raw_arr, recovered_arr))


class TestPersistentTCPTransport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.port = get_free_port()
        self.received_packets = []

        async def forward_handler(packet: BatchedActivationPacket) -> BatchedGenerationResponse:
            self.received_packets.append(packet)
            responses = [
                GenerationResponse(
                    request_id=req_id,
                    token_id=101,
                    text="world",
                    is_finished=False,
                    latency_ms=1.5,
                )
                for req_id in packet.request_ids
            ]
            return BatchedGenerationResponse(
                responses=responses,
                batch_size=len(responses),
                stage_timings={"stage_0_compute_ms": 1.2},
            )

        async def release_handler(packet: ReleaseSessionPacket) -> ReleaseSessionResponse:
            return ReleaseSessionResponse(
                status="ok",
                released_count=len(packet.request_ids),
                active_sessions=0,
            )

        self.server = BinaryTransportServer(
            host="127.0.0.1",
            port=self.port,
            forward_handler=forward_handler,
            release_handler=release_handler,
        )
        await self.server.start()

    async def asyncTearDown(self):
        if self.server:
            await self.server.stop()

    async def test_persistent_streaming_multiple_requests(self):
        """Verify multiple requests stream over the same persistent TCP connection."""
        client = BinaryTransportClient(f"127.0.0.1:{self.port}", timeout=5.0)

        for step in range(3):
            arr = np.ones((1, 1, 64), dtype=np.float16) * (step + 1)
            packet = BatchedActivationPacket(
                request_ids=[f"stream-{step}"],
                sequence_steps=[step],
                stage_id=0,
                is_prefill=False,
                tensor_shape=[1, 1, 64],
                tensor_dtype="float16",
            )
            packet.set_raw_tensor(arr.tobytes(), shape=[1, 1, 64], dtype="float16")

            resp = await client.send_batched_forward(packet)
            self.assertEqual(resp.batch_size, 1)
            self.assertEqual(resp.responses[0].request_id, f"stream-{step}")
            self.assertEqual(resp.responses[0].token_id, 101)

        self.assertEqual(len(self.received_packets), 3)
        # Verify the received data inside server
        last_arr = self.received_packets[-1].get_tensor()
        self.assertTrue(np.allclose(last_arr, 3.0))

        await client.close()

    async def test_session_release_over_tcp(self):
        """Verify session KV cache release works over persistent TCP."""
        client = BinaryTransportClient(f"127.0.0.1:{self.port}", timeout=5.0)
        rel_resp = await client.send_release_sessions(["req-1", "req-2"])
        self.assertEqual(rel_resp.status, "ok")
        self.assertEqual(rel_resp.released_count, 2)
        await client.close()

    async def test_auto_reconnect_on_server_restart(self):
        """Verify client automatically recovers and reconnects if server drops connection."""
        client = BinaryTransportClient(f"127.0.0.1:{self.port}", max_retries=4, retry_backoff_ms=25.0)

        # 1. First request succeeds
        p1 = BatchedActivationPacket(
            request_ids=["first-req"],
            sequence_steps=[0],
            stage_id=0,
            tensor_shape=[1, 1, 16],
            tensor_dtype="float16",
        )
        p1.set_raw_tensor(np.zeros((1, 1, 16), dtype=np.float16).tobytes(), shape=[1, 1, 16], dtype="float16")
        r1 = await client.send_batched_forward(p1)
        self.assertEqual(r1.responses[0].request_id, "first-req")

        # 2. Simulate server restart: stop server, re-bind to same port
        await self.server.stop()

        # Restart server on the same port
        await self.server.start()

        # 3. Second request through client should detect disconnect, reconnect, and succeed!
        p2 = BatchedActivationPacket(
            request_ids=["reconnect-req"],
            sequence_steps=[1],
            stage_id=0,
            tensor_shape=[1, 1, 16],
            tensor_dtype="float16",
        )
        p2.set_raw_tensor(np.zeros((1, 1, 16), dtype=np.float16).tobytes(), shape=[1, 1, 16], dtype="float16")
        r2 = await client.send_batched_forward(p2)
        self.assertEqual(r2.responses[0].request_id, "reconnect-req")

        await client.close()

    async def test_candidate_peer_failover(self):
        """Verify client fails over to secondary replica if primary is dead."""
        dead_port = get_free_port()
        live_port = self.port

        # Dead peer is first in the list
        client = BinaryTransportClient([f"127.0.0.1:{dead_port}", f"127.0.0.1:{live_port}"], timeout=2.0)

        p = BatchedActivationPacket(
            request_ids=["failover-req"],
            sequence_steps=[0],
            stage_id=0,
            tensor_shape=[1, 1, 16],
            tensor_dtype="float16",
        )
        p.set_raw_tensor(np.zeros((1, 1, 16), dtype=np.float16).tobytes(), shape=[1, 1, 16], dtype="float16")

        resp = await client.send_batched_forward(p)
        self.assertEqual(resp.responses[0].request_id, "failover-req")
        # Ensure client rotated to the live peer
        self.assertIn(str(live_port), client.current_peer)

        await client.close()


if __name__ == "__main__":
    unittest.main()
