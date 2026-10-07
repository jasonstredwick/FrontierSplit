"""FrontierSplit IFEval (Instruction-Following Evaluation) Benchmark Runner.

Evaluates verifiable instruction-following accuracy (word counts, formatting,
letter frequencies, JSON constraints, casing) while driving concurrent multi-stream
traffic through the FrontierSplit pipeline to eliminate idle bubbles.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import httpx


# ---------------------------------------------------------------------------
# Verifiable Instruction Rule Definitions & Checkers
# ---------------------------------------------------------------------------

def check_number_words(text: str, kwargs: Dict[str, Any], loose: bool = False) -> bool:
    """Check minimum and/or maximum word count."""
    words = re.findall(r"\b\w+\b", text)
    count = len(words)
    min_words = kwargs.get("min_words")
    max_words = kwargs.get("max_words")
    if min_words is not None and count < min_words:
        return False
    if max_words is not None and count > max_words:
        return False
    return True


def check_number_paragraphs(text: str, kwargs: Dict[str, Any], loose: bool = False) -> bool:
    """Check exact number of paragraphs separated by double newlines."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    expected = kwargs.get("num_paragraphs", 1)
    return len(paragraphs) == expected


def check_json_format(text: str, kwargs: Dict[str, Any], loose: bool = False) -> bool:
    """Check if the text is valid JSON."""
    clean = text.strip()
    if loose:
        # Strip potential markdown ```json ... ``` blocks
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean).strip()
    try:
        data = json.loads(clean)
        required_keys = kwargs.get("required_keys")
        if required_keys and isinstance(data, dict):
            return all(k in data for k in required_keys)
        return True
    except Exception:
        return False


def check_forbidden_words(text: str, kwargs: Dict[str, Any], loose: bool = False) -> bool:
    """Check that forbidden words do not appear in the text (case-insensitive)."""
    forbidden = kwargs.get("forbidden_words", [])
    lower_text = text.lower()
    for word in forbidden:
        pattern = r"\b" + re.escape(word.lower()) + r"\b"
        if re.search(pattern, lower_text):
            return False
    return True


def check_letter_frequency(text: str, kwargs: Dict[str, Any], loose: bool = False) -> bool:
    """Check frequency of a specific character (e.g. without letter 'e')."""
    letter = kwargs.get("letter", "").lower()
    if not letter:
        return True
    count = text.lower().count(letter)
    max_count = kwargs.get("max_count")
    min_count = kwargs.get("min_count")
    if max_count is not None and count > max_count:
        return False
    if min_count is not None and count < min_count:
        return False
    return True


def check_casing(text: str, kwargs: Dict[str, Any], loose: bool = False) -> bool:
    """Check if text is all uppercase or all lowercase."""
    mode = kwargs.get("case", "lowercase")
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    if mode == "lowercase":
        return all(c.islower() for c in alpha_chars)
    elif mode == "uppercase":
        return all(c.isupper() for c in alpha_chars)
    return True


def check_start_end(text: str, kwargs: Dict[str, Any], loose: bool = False) -> bool:
    """Check if text starts or ends with a specific phrase."""
    target_start = kwargs.get("start_phrase")
    target_end = kwargs.get("end_phrase")
    stripped = text.strip() if loose else text
    if target_start and not stripped.startswith(target_start):
        return False
    if target_end and not stripped.endswith(target_end):
        return False
    return True


def check_bullet_points(text: str, kwargs: Dict[str, Any], loose: bool = False) -> bool:
    """Check bullet point count (lines starting with *, -, or numbered 1.)."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    bullet_lines = [
        line for line in lines
        if re.match(r"^(\*|-|\+|\d+\.)\s+", line)
    ]
    expected = kwargs.get("num_bullets")
    min_bullets = kwargs.get("min_bullets")
    if expected is not None:
        return len(bullet_lines) == expected
    if min_bullets is not None:
        return len(bullet_lines) >= min_bullets
    return len(bullet_lines) > 0


RULE_REGISTRY: Dict[str, Callable[[str, Dict[str, Any], bool], bool]] = {
    "number_words": check_number_words,
    "number_paragraphs": check_number_paragraphs,
    "json_format": check_json_format,
    "forbidden_words": check_forbidden_words,
    "letter_frequency": check_letter_frequency,
    "casing": check_casing,
    "start_end": check_start_end,
    "bullet_points": check_bullet_points,
}


@dataclass
class IFEvalInstruction:
    instruction_id: str
    rule_type: str
    kwargs: Dict[str, Any]

    def verify(self, text: str, loose: bool = False) -> bool:
        checker = RULE_REGISTRY.get(self.rule_type)
        if not checker:
            return False
        return checker(text, self.kwargs, loose=loose)


@dataclass
class IFEvalPrompt:
    prompt_id: str
    prompt: str
    instructions: List[IFEvalInstruction]


# ---------------------------------------------------------------------------
# Standard Benchmark Suite (Curated Representative Dataset)
# ---------------------------------------------------------------------------

DEFAULT_IFEVAL_DATASET: List[IFEvalPrompt] = [
    IFEvalPrompt(
        prompt_id="ifeval-001",
        prompt="Write a short summary of how Mixture-of-Experts routing works. Your response must be at least 30 words and at most 60 words.",
        instructions=[
            IFEvalInstruction("inst-1a", "number_words", {"min_words": 30, "max_words": 60}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-002",
        prompt="Explain the purpose of pipeline parallelism. Do NOT use the word 'latency' or the word 'network' in your response.",
        instructions=[
            IFEvalInstruction("inst-2a", "forbidden_words", {"forbidden_words": ["latency", "network"]}),
            IFEvalInstruction("inst-2b", "number_words", {"min_words": 15}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-003",
        prompt="Output a valid JSON object describing a GPU cluster with keys 'gpus', 'vram_per_gpu_gb', and 'interconnect'. Do not include extra text.",
        instructions=[
            IFEvalInstruction("inst-3a", "json_format", {"required_keys": ["gpus", "vram_per_gpu_gb", "interconnect"]}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-004",
        prompt="Write a 2-paragraph essay comparing NVLink to VPC Ethernet. Separate the paragraphs with a blank line.",
        instructions=[
            IFEvalInstruction("inst-4a", "number_paragraphs", {"num_paragraphs": 2}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-005",
        prompt="Describe the concept of an activation vector in ALL UPPERCASE letters.",
        instructions=[
            IFEvalInstruction("inst-5a", "casing", {"case": "uppercase"}),
            IFEvalInstruction("inst-5b", "number_words", {"min_words": 10}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-006",
        prompt="Write a list of exactly 4 bullet points explaining why pipeline bubbles occur during single-stream inference.",
        instructions=[
            IFEvalInstruction("inst-6a", "bullet_points", {"num_bullets": 4}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-007",
        prompt="Write a short sentence about AI hardware without using the letter 'e'.",
        instructions=[
            IFEvalInstruction("inst-7a", "letter_frequency", {"letter": "e", "max_count": 0}),
            IFEvalInstruction("inst-7b", "number_words", {"min_words": 4}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-008",
        prompt="Explain continuous micro-batch interleaving. Your entire response must be written in lowercase letters only.",
        instructions=[
            IFEvalInstruction("inst-8a", "casing", {"case": "lowercase"}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-009",
        prompt="Summarize the benefits of FP8 quantization. End your response with the exact phrase: 'Efficiency unlocked.'",
        instructions=[
            IFEvalInstruction("inst-9a", "start_end", {"end_phrase": "Efficiency unlocked."}),
        ],
    ),
    IFEvalPrompt(
        prompt_id="ifeval-010",
        prompt="Provide a JSON response with keys 'project' and 'target_model'.",
        instructions=[
            IFEvalInstruction("inst-10a", "json_format", {"required_keys": ["project", "target_model"]}),
        ],
    ),
]


# ---------------------------------------------------------------------------
# IFEval Benchmark Runner
# ---------------------------------------------------------------------------

class IFEvalRunner:
    """Executes IFEval benchmark against the FrontierSplit API Gateway."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "frontiersplit-mixtral-8x7b",
        concurrency: int = 4,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout

    async def evaluate_single_prompt(
        self,
        client: httpx.AsyncClient,
        prompt_item: IFEvalPrompt,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Send prompt to gateway, measure latency, and verify all constraints."""
        start_time = time.time()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_item.prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        try:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
            tokens_count = data.get("usage", {}).get("completion_tokens", max_tokens)
            error = None
        except Exception as e:
            response_text = ""
            tokens_count = 0
            error = str(e)

        elapsed_s = time.time() - start_time

        # Evaluate instruction constraints
        inst_strict_results: List[bool] = []
        inst_loose_results: List[bool] = []
        instruction_breakdown: List[Dict[str, Any]] = []

        if error is None:
            for inst in prompt_item.instructions:
                strict_pass = inst.verify(response_text, loose=False)
                loose_pass = inst.verify(response_text, loose=True)
                inst_strict_results.append(strict_pass)
                inst_loose_results.append(loose_pass)
                instruction_breakdown.append({
                    "instruction_id": inst.instruction_id,
                    "rule_type": inst.rule_type,
                    "strict_pass": strict_pass,
                    "loose_pass": loose_pass,
                })
        else:
            for inst in prompt_item.instructions:
                inst_strict_results.append(False)
                inst_loose_results.append(False)
                instruction_breakdown.append({
                    "instruction_id": inst.instruction_id,
                    "rule_type": inst.rule_type,
                    "strict_pass": False,
                    "loose_pass": False,
                })

        prompt_strict_pass = bool(inst_strict_results and all(inst_strict_results))
        prompt_loose_pass = bool(inst_loose_results and all(inst_loose_results))

        return {
            "prompt_id": prompt_item.prompt_id,
            "prompt": prompt_item.prompt,
            "response": response_text,
            "tokens_count": tokens_count,
            "latency_s": round(elapsed_s, 3),
            "prompt_strict_pass": prompt_strict_pass,
            "prompt_loose_pass": prompt_loose_pass,
            "instruction_breakdown": instruction_breakdown,
            "error": error,
        }

    async def run_evaluation(
        self,
        dataset: Optional[List[IFEvalPrompt]] = None,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Execute concurrent evaluation over the benchmark dataset."""
        items = dataset or DEFAULT_IFEVAL_DATASET
        print("\n" + "=" * 76)
        print(f" FrontierSplit IFEval Benchmark: {len(items)} Prompts (Concurrency N={self.concurrency})")
        print("=" * 76)

        start_time = time.time()
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient() as client:
            async def bounded_eval(p: IFEvalPrompt):
                async with semaphore:
                    return await self.evaluate_single_prompt(client, p, max_tokens=max_tokens)

            results = await asyncio.gather(*[bounded_eval(p) for p in items])

        total_elapsed = time.time() - start_time
        total_tokens = sum(r["tokens_count"] for r in results)
        total_prompts = len(results)

        total_instructions = sum(len(r["instruction_breakdown"]) for r in results)
        strict_prompts_passed = sum(1 for r in results if r["prompt_strict_pass"])
        loose_prompts_passed = sum(1 for r in results if r["prompt_loose_pass"])

        strict_inst_passed = sum(
            sum(1 for inst in r["instruction_breakdown"] if inst["strict_pass"])
            for r in results
        )
        loose_inst_passed = sum(
            sum(1 for inst in r["instruction_breakdown"] if inst["loose_pass"])
            for r in results
        )

        strict_prompt_acc = (strict_prompts_passed / total_prompts * 100.0) if total_prompts else 0.0
        loose_prompt_acc = (loose_prompts_passed / total_prompts * 100.0) if total_prompts else 0.0
        strict_inst_acc = (strict_inst_passed / total_instructions * 100.0) if total_instructions else 0.0
        loose_inst_acc = (loose_inst_passed / total_instructions * 100.0) if total_instructions else 0.0

        throughput = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
        avg_latency = sum(r["latency_s"] for r in results) / total_prompts if total_prompts else 0.0

        # Theoretical bubble elimination with K=4 stages
        k = 4
        m = self.concurrency
        bubble_fraction = (k - 1) / (m + k - 1)
        bubble_elimination = (1.0 - bubble_fraction) * 100.0

        summary = {
            "benchmark": "IFEval",
            "model": self.model,
            "concurrency": self.concurrency,
            "total_prompts": total_prompts,
            "total_instructions": total_instructions,
            "strict_prompt_accuracy": round(strict_prompt_acc, 2),
            "loose_prompt_accuracy": round(loose_prompt_acc, 2),
            "strict_instruction_accuracy": round(strict_inst_acc, 2),
            "loose_instruction_accuracy": round(loose_inst_acc, 2),
            "total_tokens": total_tokens,
            "elapsed_s": round(total_elapsed, 3),
            "throughput_tok_per_sec": round(throughput, 2),
            "avg_latency_s": round(avg_latency, 3),
            "bubble_fraction": round(bubble_fraction, 4),
            "bubble_elimination_pct": round(bubble_elimination, 1),
            "results": results,
        }

        self.print_summary_table(summary)
        return summary

    @staticmethod
    def print_summary_table(summary: Dict[str, Any]) -> None:
        """Render a clean summary of IFEval accuracy and pipeline hardware metrics."""
        print("\n" + "=" * 76)
        print(" IFEval Benchmark Results & Pipeline Telemetry")
        print("=" * 76)
        print(f"  Target Model:                   {summary['model']}")
        print(f"  Concurrency Level (M):          {summary['concurrency']} concurrent streams")
        print(f"  Total Prompts Evaluated:        {summary['total_prompts']}")
        print(f"  Total Verifiable Constraints:   {summary['total_instructions']}")
        print("-" * 76)
        print("  ACCURACY METRICS:")
        print(f"    Strict Prompt Accuracy:       {summary['strict_prompt_accuracy']}%")
        print(f"    Loose Prompt Accuracy:        {summary['loose_prompt_accuracy']}%")
        print(f"    Strict Instruction Accuracy:  {summary['strict_instruction_accuracy']}%")
        print(f"    Loose Instruction Accuracy:   {summary['loose_instruction_accuracy']}%")
        print("-" * 76)
        print("  PIPELINE PERFORMANCE & BUBBLE ELIMINATION:")
        print(f"    Aggregate Throughput:         {summary['throughput_tok_per_sec']} tokens/s")
        print(f"    Average Request Latency:      {summary['avg_latency_s']} s")
        print(f"    Theoretical Idle Bubble (F):  {round(summary['bubble_fraction'] * 100, 1)}%")
        print(f"    Hardware Saturation Rate:     {summary['bubble_elimination_pct']}%")
        print("=" * 76)


def load_dataset_from_jsonl(filepath: str) -> List[IFEvalPrompt]:
    """Load IFEval prompts and instructions from a JSONL file."""
    prompts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            instructions = [
                IFEvalInstruction(
                    instruction_id=inst.get("instruction_id", "inst"),
                    rule_type=inst["rule_type"],
                    kwargs=inst.get("kwargs", {}),
                )
                for inst in item.get("instructions", [])
            ]
            prompts.append(
                IFEvalPrompt(
                    prompt_id=item.get("prompt_id", "prompt"),
                    prompt=item["prompt"],
                    instructions=instructions,
                )
            )
    return prompts


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit IFEval Benchmark Runner")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="Gateway URL")
    parser.add_argument("--model", type=str, default="frontiersplit-mixtral-8x7b", help="Model name")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent request streams")
    parser.add_argument("--dataset", type=str, default=None, help="Path to external JSONL dataset")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON results")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per prompt")
    args = parser.parse_args()

    dataset = load_dataset_from_jsonl(args.dataset) if args.dataset else None
    runner = IFEvalRunner(base_url=args.base_url, model=args.model, concurrency=args.concurrency)
    res = asyncio.run(runner.run_evaluation(dataset=dataset, max_tokens=args.max_tokens))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
