"""Unit tests for FrontierSplit Chunked Sequence Pipeline Architecture (B removed, S chunked into C=16)."""

import unittest
import numpy as np
from frontiersplit.protocol import (
    ChunkActivationPacket,
    ChunkGenerationResponse,
    MSG_FORWARD_CHUNK_REQ,
    MSG_FORWARD_CHUNK_RESP,
)
from frontiersplit.scheduler import ScheduledRequest


class TestChunkPipeline(unittest.TestCase):
    """Verifies chunked sequence creation, padding, and packet wire serialization."""

    def test_scheduled_request_chunking(self):
        # Prompt of length S = 37 tokens, chunk size C = 16
        prompt_tokens = list(range(100, 137))  # 37 tokens
        req = ScheduledRequest(
            request_id="test-req-1",
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            prompt_text="hello",
            prompt_tokens=prompt_tokens,
            max_tokens=32,
        )

        # Expected D = ceil(37 / 16) = 3
        self.assertEqual(req.total_prefill_chunks, 3)
        self.assertEqual(len(req.prefill_chunks), 3)
        self.assertEqual(len(req.chunk_valid_lens), 3)

        # Chunk 0: 16 tokens
        self.assertEqual(len(req.prefill_chunks[0]), 16)
        self.assertEqual(req.chunk_valid_lens[0], 16)
        self.assertEqual(req.prefill_chunks[0], prompt_tokens[0:16])

        # Chunk 1: 16 tokens
        self.assertEqual(len(req.prefill_chunks[1]), 16)
        self.assertEqual(req.chunk_valid_lens[1], 16)
        self.assertEqual(req.prefill_chunks[1], prompt_tokens[16:32])

        # Chunk 2: 5 valid tokens + 11 padding tokens
        self.assertEqual(len(req.prefill_chunks[2]), 16)
        self.assertEqual(req.chunk_valid_lens[2], 5)
        self.assertEqual(req.prefill_chunks[2][:5], prompt_tokens[32:37])
        self.assertEqual(req.prefill_chunks[2][5:], [0] * 11)

    def test_chunk_packet_binary_serialization(self):
        # Exactly [16, 4096] in float16
        arr = np.random.randn(16, 4096).astype(np.float16)
        packet = ChunkActivationPacket(
            request_id="req-chunk-abc",
            chunk_idx=2,
            total_chunks=5,
            chunk_size=16,
            valid_tokens=7,
            is_prefill=True,
            stage_id=3,
            tensor_shape=[16, 4096],
            tensor_dtype="float16",
        )
        packet.set_tensor(arr)

        raw_bytes = packet.get_raw_bytes()
        self.assertEqual(len(raw_bytes), 16 * 4096 * 2)  # 131,072 bytes

        frame = packet.encode_binary()
        decoded = ChunkActivationPacket.from_binary_frame(frame)

        self.assertEqual(decoded.request_id, "req-chunk-abc")
        self.assertEqual(decoded.chunk_idx, 2)
        self.assertEqual(decoded.total_chunks, 5)
        self.assertEqual(decoded.chunk_size, 16)
        self.assertEqual(decoded.valid_tokens, 7)
        self.assertEqual(decoded.is_prefill, True)
        self.assertEqual(decoded.stage_id, 3)
        self.assertEqual(decoded.tensor_shape, [16, 4096])
        self.assertEqual(decoded.tensor_dtype, "float16")

        reconstructed_arr = decoded.get_tensor()
        np.testing.assert_array_equal(arr, reconstructed_arr)

    def test_chunk_response_binary_serialization(self):
        resp = ChunkGenerationResponse(
            request_id="req-chunk-xyz",
            chunk_idx=4,
            is_final_chunk=True,
            next_token_id=28705,
            is_finished=False,
            stage_timings={"stage_0_compute_ms": 12.5, "stage_7_compute_ms": 14.1},
        )
        frame = resp.encode_binary()

        # Unpack header + meta
        from frontiersplit.protocol import unpack_header, HEADER_SIZE
        import json
        hdr = unpack_header(frame[:HEADER_SIZE])
        self.assertEqual(hdr["msg_type"], MSG_FORWARD_CHUNK_RESP)
        meta = json.loads(frame[HEADER_SIZE : HEADER_SIZE + hdr["meta_len"]].decode("utf-8"))
        decoded = ChunkGenerationResponse.model_validate(meta)

        self.assertEqual(decoded.request_id, "req-chunk-xyz")
        self.assertEqual(decoded.chunk_idx, 4)
        self.assertEqual(decoded.is_final_chunk, True)
        self.assertEqual(decoded.next_token_id, 28705)
        self.assertEqual(decoded.is_finished, False)
        self.assertEqual(decoded.stage_timings["stage_0_compute_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()
