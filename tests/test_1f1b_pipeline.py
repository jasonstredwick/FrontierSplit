"""Dedicated integration tests for FrontierSplit Asynchronous 1F1B Micro-batch Pipelining."""

import asyncio
import socket
import unittest

from frontiersplit.protocol import BatchedActivationPacket
from frontiersplit.scheduler import PipelineScheduler
from frontiersplit.transport import BinaryTransportClient
from frontiersplit.worker import create_worker_app


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Test1F1BPipeline(unittest.IsolatedAsyncioTestCase):
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

    async def asyncTearDown(self):
        for stage_app in [self.stage0_app, self.stage1_app, self.stage2_app, self.stage3_app]:
            if stage_app.state.binary_server:
                await stage_app.state.binary_server.stop()
            if stage_app.state.downstream_client:
                await stage_app.state.downstream_client.close()

    async def test_1f1b_concurrent_generation(self):
        """Verify 1F1B asynchronous pipelining handles 8 concurrent streams cleanly."""
        scheduler = PipelineScheduler(
            stage0_tcp=f"127.0.0.1:{self.port0}",
            num_workers=4,
            total_stages=4,
            tokenizer=None,
            max_batch_size=4,
            use_kv_cache=True,
            enable_1f1b=True,
            reply_port=0,  # Ephemeral port
        )
        await scheduler.start()

        try:
            # Submit 8 concurrent requests
            requests = []
            for i in range(8):
                req = await scheduler.submit_request(
                    model="frontiersplit-test",
                    messages=[{"role": "user", "content": f"Task {i}"}],
                    max_tokens=6,
                )
                requests.append(req)

            # Await all 8 streams to complete
            done_futures = [req.done_event.wait() for req in requests]
            await asyncio.wait_for(asyncio.gather(*done_futures), timeout=10.0)

            for idx, req in enumerate(requests):
                self.assertTrue(req.is_finished, f"Request {idx} failed to finish")
                self.assertIsNone(req.error, f"Request {idx} error: {req.error}")
                self.assertEqual(len(req.generated_tokens), 6, f"Request {idx} token count mismatch")
                self.assertEqual(len(req.generated_chunks), 6)

            telemetry = scheduler.get_telemetry()
            self.assertEqual(telemetry["total_completed"], 8)
            self.assertEqual(telemetry["total_tokens_generated"], 48)
            self.assertGreater(telemetry["throughput_tokens_per_sec"], 0)

        finally:
            await scheduler.stop()

    async def test_1f1b_vs_sync_switch(self):
        """Verify PipelineScheduler can toggle cleanly between 1F1B and synchronous modes."""
        # Run with 1F1B disabled (synchronous mode)
        scheduler_sync = PipelineScheduler(
            stage0_tcp=f"127.0.0.1:{self.port0}",
            num_workers=2,
            total_stages=4,
            tokenizer=None,
            max_batch_size=2,
            use_kv_cache=True,
            enable_1f1b=False,
        )
        await scheduler_sync.start()
        try:
            req_sync = await scheduler_sync.submit_request(
                model="frontiersplit-test",
                messages=[{"role": "user", "content": "Sync test"}],
                max_tokens=4,
            )
            await asyncio.wait_for(req_sync.done_event.wait(), timeout=5.0)
            self.assertTrue(req_sync.is_finished)
            self.assertEqual(len(req_sync.generated_tokens), 4)
        finally:
            await scheduler_sync.stop()

        # Run with 1F1B enabled
        scheduler_1f1b = PipelineScheduler(
            stage0_tcp=f"127.0.0.1:{self.port0}",
            num_workers=2,
            total_stages=4,
            tokenizer=None,
            max_batch_size=2,
            use_kv_cache=True,
            enable_1f1b=True,
            reply_port=0,
        )
        await scheduler_1f1b.start()
        try:
            req_1f1b = await scheduler_1f1b.submit_request(
                model="frontiersplit-test",
                messages=[{"role": "user", "content": "1F1B test"}],
                max_tokens=4,
            )
            await asyncio.wait_for(req_1f1b.done_event.wait(), timeout=5.0)
            self.assertTrue(req_1f1b.is_finished)
            self.assertEqual(len(req_1f1b.generated_tokens), 4)
        finally:
            await scheduler_1f1b.stop()


if __name__ == "__main__":
    unittest.main()
