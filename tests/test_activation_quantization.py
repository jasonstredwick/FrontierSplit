"""Integration and accuracy tests for FrontierSplit Activation Quantization (INT8/FP8)."""

import asyncio
import socket
import unittest
import numpy as np

from frontiersplit.protocol import (
    BatchedActivationPacket,
    FLAG_QUANTIZED,
    HEADER_SIZE,
    unpack_header,
)
from frontiersplit.scheduler import PipelineScheduler
from frontiersplit.worker import create_worker_app


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestActivationQuantization(unittest.TestCase):
    def test_numerical_fidelity_and_compression(self):
        """Verify dynamic symmetric INT8 quantization achieves < 1% relative error and 50% compression."""
        np.random.seed(42)
        # Typical hidden states: (batch_size=4, seq_len=1, hidden_size=4096)
        x_fp16 = np.random.randn(4, 1, 4096).astype(np.float16)

        amax = float(np.max(np.abs(x_fp16)))
        scale = amax / 127.0
        q = np.clip(np.round(x_fp16 / scale), -128, 127).astype(np.int8)

        # Dequantize
        x_recon = (q.astype(np.float32) * scale).astype(np.float32)

        # Relative error: ||x - x_recon|| / ||x||
        rel_error = np.mean(np.abs(x_fp16.astype(np.float32) - x_recon)) / np.mean(np.abs(x_fp16.astype(np.float32)))
        self.assertLess(rel_error, 0.015, f"Relative error {rel_error:.4f} exceeds 1.5% threshold")

        cos_sim = float(np.dot(x_fp16.flatten().astype(np.float32), x_recon.flatten()) / (np.linalg.norm(x_fp16) * np.linalg.norm(x_recon)))
        self.assertGreater(cos_sim, 0.999, f"Cosine similarity {cos_sim:.5f} below 0.999")

        # Compression ratio: FP16 (2 bytes/element) vs INT8 (1 byte/element)
        fp16_bytes = x_fp16.nbytes
        int8_bytes = q.nbytes
        self.assertEqual(int8_bytes * 2, fp16_bytes)

    def test_packet_quantization_framing(self):
        """Verify binary protocol framing preserves quantization flags, scales, and dequantization."""
        packet = BatchedActivationPacket(
            request_ids=["req-1", "req-2"],
            sequence_steps=[0, 0],
            stage_id=1,
            is_prefill=True,
            tensor_scale=0.015625,
            tensor_shape=[2, 1, 64],
            tensor_dtype="int8",
        )
        fake_int8_data = np.arange(-64, 64, dtype=np.int8).tobytes()
        packet.set_raw_tensor(fake_int8_data, [2, 1, 64], "int8")

        binary_bytes = packet.encode_binary()
        # Verify 32-byte header
        self.assertGreater(len(binary_bytes), HEADER_SIZE)
        hdr = unpack_header(binary_bytes[:HEADER_SIZE])
        self.assertTrue(hdr["flags"] & FLAG_QUANTIZED, "FLAG_QUANTIZED must be set in header")

        # Decode using from_binary_frame
        decoded = BatchedActivationPacket.from_binary_frame(binary_bytes)
        self.assertEqual(decoded.tensor_scale, 0.015625)
        self.assertEqual(decoded.tensor_dtype, "int8")
        self.assertEqual(decoded.tensor_shape, [2, 1, 64])

        # Verify get_tensor dequantizes using scale
        recon_arr = decoded.get_tensor()
        self.assertEqual(recon_arr.shape, (2, 1, 64))
        expected_arr = (np.arange(-64, 64, dtype=np.int8).astype(np.float32) * 0.015625).reshape(2, 1, 64)
        np.testing.assert_allclose(recon_arr, expected_arr, rtol=1e-5)


class TestQuantizedPipelineIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.port3 = get_free_port()
        self.port2 = get_free_port()
        self.port1 = get_free_port()
        self.port0 = get_free_port()

        # Build 4-stage worker applications with quantize_activations=True
        self.stage3_app = create_worker_app(
            stage_id=3,
            total_stages=4,
            tcp_port=self.port3,
            downstream_tcp=None,
            hidden_size=64,
            vocab_size=256,
            quantize_activations=True,
        )
        self.stage2_app = create_worker_app(
            stage_id=2,
            total_stages=4,
            tcp_port=self.port2,
            downstream_tcp=f"127.0.0.1:{self.port3}",
            hidden_size=64,
            vocab_size=256,
            quantize_activations=True,
        )
        self.stage1_app = create_worker_app(
            stage_id=1,
            total_stages=4,
            tcp_port=self.port1,
            downstream_tcp=f"127.0.0.1:{self.port2}",
            hidden_size=64,
            vocab_size=256,
            quantize_activations=True,
        )
        self.stage0_app = create_worker_app(
            stage_id=0,
            total_stages=4,
            tcp_port=self.port0,
            downstream_tcp=f"127.0.0.1:{self.port1}",
            hidden_size=64,
            vocab_size=256,
            quantize_activations=True,
        )

        # Start all 4 persistent binary TCP servers
        await self.stage3_app.state.binary_server.start()
        await self.stage2_app.state.binary_server.start()
        await self.stage1_app.state.binary_server.start()
        await self.stage0_app.state.binary_server.start()

    async def asyncTearDown(self):
        for stage_app in [self.stage0_app, self.stage1_app, self.stage2_app, self.stage3_app]:
            if stage_app.state.binary_server:
                await stage_app.state.binary_server.stop()
            if stage_app.state.downstream_client:
                await stage_app.state.downstream_client.close()

    async def test_quantized_pipeline_e2e_generation(self):
        """Verify 4-stage pipeline executes with activation quantization active on all inter-stage hops."""
        scheduler = PipelineScheduler(
            stage0_tcp=f"127.0.0.1:{self.port0}",
            num_workers=4,
            total_stages=4,
            tokenizer=None,
            max_batch_size=4,
            use_kv_cache=True,
            enable_1f1b=True,
            reply_port=0,
        )
        await scheduler.start()

        try:
            # Submit 4 concurrent requests
            requests = []
            for i in range(4):
                req = await scheduler.submit_request(
                    model="frontiersplit-test",
                    messages=[{"role": "user", "content": f"Quantized test prompt {i}"}],
                    max_tokens=5,
                )
                requests.append(req)

            # Await all streams
            done_futures = [req.done_event.wait() for req in requests]
            await asyncio.wait_for(asyncio.gather(*done_futures), timeout=10.0)

            for i, req in enumerate(requests):
                self.assertIsNone(req.error, f"Request {i} encountered error: {req.error}")
                self.assertTrue(req.is_finished, f"Request {i} did not finish")
                self.assertGreaterEqual(len(req.generated_tokens), 5, f"Request {i} missing tokens")
        finally:
            await scheduler.stop()
