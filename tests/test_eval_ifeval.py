"""Tests for FrontierSplit IFEval Benchmark Runner."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import httpx

from benchmarks.eval_ifeval import (
    IFEvalInstruction,
    IFEvalPrompt,
    IFEvalRunner,
    check_bullet_points,
    check_casing,
    check_forbidden_words,
    check_json_format,
    check_letter_frequency,
    check_number_paragraphs,
    check_number_words,
    check_start_end,
    load_dataset_from_jsonl,
)


class TestIFEvalRules(unittest.TestCase):
    def test_word_count(self):
        text = "This is a sentence containing exactly seven words."
        self.assertTrue(check_number_words(text, {"min_words": 5, "max_words": 10}))
        self.assertFalse(check_number_words(text, {"min_words": 10}))
        self.assertFalse(check_number_words(text, {"max_words": 4}))

    def test_paragraphs(self):
        text = "Paragraph one is here.\n\nParagraph two is here."
        self.assertTrue(check_number_paragraphs(text, {"num_paragraphs": 2}))
        self.assertFalse(check_number_paragraphs(text, {"num_paragraphs": 3}))

    def test_json_format(self):
        strict_valid = '{"gpus": 4, "vram_per_gpu_gb": 24, "interconnect": "vpc"}'
        self.assertTrue(check_json_format(strict_valid, {"required_keys": ["gpus"]}))
        self.assertFalse(check_json_format(strict_valid, {"required_keys": ["nonexistent"]}))

        # Markdown wrapped JSON
        loose_valid = "```json\n" + strict_valid + "\n```"
        self.assertFalse(check_json_format(loose_valid, {}, loose=False))
        self.assertTrue(check_json_format(loose_valid, {}, loose=True))

        invalid = "Here is JSON: {gpus: 4}"
        self.assertFalse(check_json_format(invalid, {}, loose=True))

    def test_forbidden_words(self):
        text = "This solution uses pipeline parallelism over standard network links."
        self.assertTrue(check_forbidden_words(text, {"forbidden_words": ["nvlink", "infiniband"]}))
        self.assertFalse(check_forbidden_words(text, {"forbidden_words": ["pipeline"]}))
        # Case insensitivity
        self.assertFalse(check_forbidden_words(text, {"forbidden_words": ["NETWORK"]}))

    def test_letter_frequency(self):
        text = "A quick brown fox jumps."
        # Contains 'e'? No 'e' in "A quick brown fox jumps."
        self.assertTrue(check_letter_frequency(text, {"letter": "e", "max_count": 0}))
        text_with_e = "The quick brown fox."
        self.assertFalse(check_letter_frequency(text_with_e, {"letter": "e", "max_count": 0}))

    def test_casing(self):
        text_upper = "FRONTIERSPLIT IS RUNNING!"
        text_lower = "frontiersplit is running!"
        self.assertTrue(check_casing(text_upper, {"case": "uppercase"}))
        self.assertFalse(check_casing(text_upper, {"case": "lowercase"}))
        self.assertTrue(check_casing(text_lower, {"case": "lowercase"}))
        self.assertFalse(check_casing(text_lower, {"case": "uppercase"}))

    def test_start_end(self):
        text = "Hello world! Done."
        self.assertTrue(check_start_end(text, {"start_phrase": "Hello", "end_phrase": "Done."}))
        self.assertFalse(check_start_end(text, {"end_phrase": "Finished."}))

    def test_bullet_points(self):
        text = "* Item 1\n* Item 2\n* Item 3\n* Item 4"
        self.assertTrue(check_bullet_points(text, {"num_bullets": 4}))
        self.assertFalse(check_bullet_points(text, {"num_bullets": 3}))


class TestIFEvalRunner(unittest.TestCase):
    def setUp(self):
        self.runner = IFEvalRunner(
            base_url="http://mock-gateway/v1",
            model="frontiersplit-test",
            concurrency=2,
        )

    def test_run_evaluation_mocked(self):
        mock_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-ifeval",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "All lowercase words in this short response.",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
            },
            request=httpx.Request("POST", "http://mock-gateway/v1/chat/completions"),
        )

        test_dataset = [
            IFEvalPrompt(
                prompt_id="test-1",
                prompt="Say something with at least 5 words.",
                instructions=[
                    IFEvalInstruction("inst-1", "number_words", {"min_words": 5}),
                ],
            ),
            IFEvalPrompt(
                prompt_id="test-2",
                prompt="Say something without the letter 'z'.",
                instructions=[
                    IFEvalInstruction("inst-2", "letter_frequency", {"letter": "z", "max_count": 0}),
                ],
            ),
        ]

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            summary = asyncio.run(self.runner.run_evaluation(dataset=test_dataset, max_tokens=16))

            self.assertEqual(summary["total_prompts"], 2)
            self.assertEqual(summary["total_instructions"], 2)
            self.assertEqual(summary["strict_prompt_accuracy"], 100.0)
            self.assertEqual(summary["loose_prompt_accuracy"], 100.0)
            self.assertEqual(mock_post.call_count, 2)
            self.assertIn("bubble_fraction", summary)
            self.assertIn("bubble_elimination_pct", summary)

    def test_load_dataset_from_jsonl(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as tf:
            tf.write(
                json.dumps({
                    "prompt_id": "file-1",
                    "prompt": "Test prompt",
                    "instructions": [{"rule_type": "number_words", "kwargs": {"min_words": 2}}],
                }) + "\n"
            )
            temp_path = tf.name

        try:
            loaded = load_dataset_from_jsonl(temp_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].prompt_id, "file-1")
            self.assertEqual(len(loaded[0].instructions), 1)
            self.assertEqual(loaded[0].instructions[0].rule_type, "number_words")
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
