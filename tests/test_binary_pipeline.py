"""End-to-end integration tests for FrontierSplit Live Persistent Binary TCP Pipeline."""

import asyncio
import socket
import unittest
import numpy as np

from frontiersplit.gateway import create_gateway_app
from frontiersplit.protocol import BatchedActivationPacket, ReleaseSessionPacket
from frontiersplit.scheduler import PipelineScheduler, ScheduledRequest
from frontiersplit.transport import BinaryTransportClient
from frontiersplit.worker import create_worker_app


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestLiveBinaryTCPPipeline(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.port3 = get_free_port()
        self.port2 = get_free_port()
        self.port1 = get_free_port()
        self.port0 = get_free_port()

        # Build 4-stage worker applications with persistent binary TCP transport
        self.stage3_app = create_worker_app(
            stage_id=3,
            total_stages=4,
            tcp_port=self.port3,
            downstream_tcp=None,
            hidden_size=64,
            vocab_size=256,
        )
        self.stage2_app = create_worker_app(
            stage_id=2,
            total_stages=4,
            tcp_port=self.port2,
            downstream_tcp=f"127.0.0.1:{self.port3}",
            hidden_size=64,
            vocab_size=256,
        )
        self.stage1_app = create_worker_app(
            stage_id=1,
            total_stages=4,
            tcp_port=self.port1,
            downstream_tcp=f"127.0.0.1:{self.port2}",
            hidden_size=64,
            vocab_size=256,
        )
        self.stage0_app = create_worker_app(
            stage_id=0,
            total_stages=4,
            tcp_port=self.port0,
            downstream_tcp=f"127.0.0.1:{self.port1}",
            hidden_size=64,
            vocab_size=256,
        )

        # Start all 4 persistent binary TCP servers
        await self.stage3_app.state.binary_server.start()
        await self.stage2_app.state.binary_server.start()
        await self.stage1_app.state.binary_server.start()
        await self.stage0_app.state.binary_server.start()

        # Create scheduler connected directly via persistent TCP to Stage 0
        self.scheduler = PipelineScheduler(
            stage0_tcp=f"127.0.0.1:{self.port0}",
            num_workers=2,
            total_stages=4,
            tokenizer=None,
            max_batch_size=4,
            use_kv_cache=True,
        )
        await self.scheduler.start()

    async def asyncTearDown(self):
        await self.scheduler.stop()
        for stage_app in [self.stage0_app, self.stage1_app, self.stage2_app, self.stage3_app]:
            if stage_app.state.binary_server:
                await stage_app.state.binary_server.stop()
            if stage_app.state.downstream_client:
                await stage_app.state.downstream_client.close()

    async def test_direct_4stage_binary_tcp_forward(self):
        """Verify direct BatchedActivationPacket hops through all 4 stages over live TCP."""
        client = BinaryTransportClient(f"127.0.0.1:{self.port0}", timeout=5.0)

        # Stage 0 token input packet
        packet = BatchedActivationPacket(
            request_ids=["binary-req-1", "binary-req-2"],
            sequence_steps=[0, 0],
            stage_id=0,
            is_prefill=True,
            use_kv_cache=True,
            tokens_batch=[[10, 20, 30], [40, 50, 60]],
            attention_mask=[[1, 1, 1], [1, 1, 1]],
        )

        response = await client.send_batched_forward(packet)
        self.assertEqual(response.batch_size, 2)
        self.assertEqual(len(response.responses), 2)
        self.assertEqual(response.responses[0].request_id, "binary-req-1")
        self.assertEqual(response.responses[1].request_id, "binary-req-2")
        self.assertIsInstance(response.responses[0].token_id, int)
        self.assertTrue(response.responses[0].text.startswith("tok_"))

        # Verify all stages recorded compute timings
        timings = response.stage_timings
        self.assertIn("stage_0_compute_ms", timings)
        self.assertIn("stage_1_compute_ms", timings)
        self.assertIn("stage_2_compute_ms", timings)
        self.assertIn("stage_3_compute_ms", timings)

        await client.close()

    async def test_scheduler_live_tcp_generation(self):
        """Verify PipelineScheduler orchestrates concurrent multi-step decode over live TCP."""
        # Submit 2 concurrent requests to the live binary TCP pipeline
        req1 = await self.scheduler.submit_request(
            model="frontiersplit-test",
            messages=[{"role": "user", "content": "Prompt 1"}],
            max_tokens=4,
        )
        req2 = await self.scheduler.submit_request(
            model="frontiersplit-test",
            messages=[{"role": "user", "content": "Prompt 2"}],
            max_tokens=4,
        )

        # Await completion
        await asyncio.wait_for(
            asyncio.gather(req1.done_event.wait(), req2.done_event.wait()),
            timeout=10.0,
        )

        self.assertTrue(req1.is_finished)
        self.assertTrue(req2.is_finished)
        self.assertEqual(len(req1.generated_tokens), 4)
        self.assertEqual(len(req2.generated_tokens), 4)
        self.assertIsNone(req1.error)
        self.assertIsNone(req2.error)

        # Check telemetry
        telemetry = self.scheduler.get_telemetry()
        self.assertGreaterEqual(telemetry["total_tokens_generated"], 8)
        self.assertEqual(telemetry["total_completed"], 2)


if __name__ == "__main__":
    unittest.main()
