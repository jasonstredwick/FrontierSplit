"""FrontierSplit Multi-Agent Concurrency Harness.

Generates synthetic agentic concurrency (Tree of Thoughts, Decomposed Agent Swarms,
and concurrency sweeps N=1, 4, 8, 16) against the Ingress Gateway to eliminate
pipeline idle bubbles and saturate distributed cluster hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List, Optional
import httpx


class MultiAgentHarness:
    """Multi-Agent concurrency driver and pipeline bubble benchmark."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "frontiersplit-mixtral-8x7b",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def _send_completion(
        self,
        client: httpx.AsyncClient,
        messages: List[Dict[str, str]],
        max_tokens: int = 32,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a single chat completion request."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        start = time.time()
        resp = await client.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        latency_s = time.time() - start

        if stream:
            tokens = []
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            tokens.append(delta)
                    except json.JSONDecodeError:
                        pass
            return {
                "content": "".join(tokens),
                "latency_s": latency_s,
                "tokens_count": len(tokens),
            }
        else:
            data = resp.json()
            usage = data.get("usage", {})
            return {
                "content": data["choices"][0]["message"]["content"],
                "latency_s": latency_s,
                "tokens_count": usage.get("completion_tokens", max_tokens),
            }

    async def run_tree_of_thoughts(
        self,
        prompt: str,
        branches: int = 4,
        tokens_per_thought: int = 24,
    ) -> Dict[str, Any]:
        """Execute parallel Tree-of-Thoughts reasoning branches simultaneously."""
        print(f"\n[ToT] Spawning {branches} parallel reasoning branches for prompt: {prompt!r}")
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            async def explore_branch(branch_id: int):
                messages = [
                    {
                        "role": "user",
                        "content": f"Problem: {prompt}\nExplore approach #{branch_id + 1} step-by-step:",
                    }
                ]
                return await self._send_completion(client, messages, max_tokens=tokens_per_thought)

            # Fire all branches concurrently into the pipeline
            branch_results = await asyncio.gather(*[explore_branch(i) for i in range(branches)])

            # Synthesis step: Consolidate parallel thoughts into final verdict
            synthesis_prompt = f"Problem: {prompt}\nSynthesize the {branches} approaches into a verified solution."
            verdict = await self._send_completion(
                client,
                [{"role": "user", "content": synthesis_prompt}],
                max_tokens=tokens_per_thought * 2,
            )

        total_elapsed = time.time() - start_time
        total_tokens = sum(r["tokens_count"] for r in branch_results) + verdict["tokens_count"]

        return {
            "mode": "tree_of_thoughts",
            "branches": branches,
            "total_tokens": total_tokens,
            "elapsed_s": round(total_elapsed, 3),
            "aggregate_tokens_per_sec": round(total_tokens / total_elapsed, 2),
            "branch_results": branch_results,
            "verdict": verdict,
        }

    async def run_agent_swarm(
        self,
        tasks: List[str],
        tokens_per_task: int = 32,
    ) -> Dict[str, Any]:
        """Execute a swarm of specialized sub-agents concurrently."""
        print(f"\n[Swarm] Dispatching {len(tasks)} sub-agent tasks concurrently...")
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            async def run_subagent(task_id: int, task_desc: str):
                messages = [
                    {"role": "system", "content": "You are a specialized autonomous engineering sub-agent."},
                    {"role": "user", "content": f"Task #{task_id + 1}: {task_desc}"},
                ]
                return await self._send_completion(client, messages, max_tokens=tokens_per_task)

            results = await asyncio.gather(*[run_subagent(i, t) for i, t in enumerate(tasks)])

        total_elapsed = time.time() - start_time
        total_tokens = sum(r["tokens_count"] for r in results)

        return {
            "mode": "agent_swarm",
            "agent_count": len(tasks),
            "total_tokens": total_tokens,
            "elapsed_s": round(total_elapsed, 3),
            "aggregate_tokens_per_sec": round(total_tokens / total_elapsed, 2),
            "results": results,
        }

    async def run_concurrency_sweep(
        self,
        concurrency_levels: List[int] = [1, 4, 8, 16],
        tokens_per_request: int = 16,
    ) -> List[Dict[str, Any]]:
        """Systematically benchmark pipeline bubble reduction under N=1, 4, 8, 16 concurrency."""
        print("\n" + "=" * 70)
        print(" FrontierSplit Pipeline Concurrency Sweep: Bubble Elimination Study")
        print("=" * 70)

        sweep_results = []

        async with httpx.AsyncClient() as client:
            for n in concurrency_levels:
                print(f"\n--- Testing Concurrency N = {n} streams ---")
                start_time = time.time()

                async def worker(stream_id: int):
                    messages = [
                        {"role": "user", "content": f"Stream {stream_id}: Analyze algorithmic complexity of MoE layers."}
                    ]
                    return await self._send_completion(client, messages, max_tokens=tokens_per_request)

                stream_results = await asyncio.gather(*[worker(i) for i in range(n)])
                elapsed_s = time.time() - start_time
                total_tokens = sum(r["tokens_count"] for r in stream_results)
                tok_per_sec = total_tokens / elapsed_s
                avg_latency = sum(r["latency_s"] for r in stream_results) / len(stream_results)

                # Theoretical bubble calculation (K=4 stages)
                k = 4
                bubble_fraction = (k - 1) / (n + k - 1)
                bubble_elimination = (1.0 - bubble_fraction) * 100.0

                result = {
                    "concurrency_n": n,
                    "total_tokens": total_tokens,
                    "elapsed_s": round(elapsed_s, 3),
                    "throughput_tok_s": round(tok_per_sec, 2),
                    "avg_latency_s": round(avg_latency, 3),
                    "bubble_fraction": round(bubble_fraction, 4),
                    "bubble_elimination_pct": round(bubble_elimination, 1),
                }
                sweep_results.append(result)

                print(f"  Throughput: {result['throughput_tok_s']} tokens/s | Latency: {result['avg_latency_s']}s")
                print(f"  Pipeline Bubble: {round(bubble_fraction * 100, 1)}% idle | Elimination: {result['bubble_elimination_pct']}% saturated")

        self.print_sweep_table(sweep_results)
        return sweep_results

    @staticmethod
    def print_sweep_table(results: List[Dict[str, Any]]) -> None:
        """Render a formatted performance comparison table."""
        print("\n" + "=" * 76)
        print(f"{'Concurrency (N)':<16} | {'Throughput (tok/s)':<18} | {'Avg Latency':<12} | {'Bubble Idle %':<14} | {'Saturated %'}")
        print("-" * 76)
        for r in results:
            bubble_pct = f"{round(r['bubble_fraction'] * 100, 1)}%"
            sat_pct = f"{r['bubble_elimination_pct']}%"
            print(f"{r['concurrency_n']:<16} | {r['throughput_tok_s']:<18} | {r['avg_latency_s']:<10}s | {bubble_pct:<14} | {sat_pct}")
        print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Multi-Agent Concurrency Harness")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="Gateway API URL")
    parser.add_argument("--model", type=str, default="frontiersplit-mixtral-8x7b", help="Model name")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sweep", "tot", "swarm"],
        default="sweep",
        help="Benchmark mode: sweep, tot (Tree of Thoughts), or swarm",
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16], help="Concurrency levels for sweep")
    parser.add_argument("--tokens", type=int, default=16, help="Tokens per request")
    args = parser.parse_args()

    harness = MultiAgentHarness(base_url=args.base_url, model=args.model)

    if args.mode == "sweep":
        asyncio.run(harness.run_concurrency_sweep(concurrency_levels=args.concurrency, tokens_per_request=args.tokens))
    elif args.mode == "tot":
        res = asyncio.run(
            harness.run_tree_of_thoughts(
                prompt="Design an optimal zero-copy ring-buffer for inter-GPU activation vectors.",
                branches=4,
                tokens_per_thought=args.tokens,
            )
        )
        print("\nTree-of-Thoughts Result:")
        print(json.dumps({k: v for k, v in res.items() if k != "branch_results"}, indent=2))
    elif args.mode == "swarm":
        tasks = [
            "Analyze AST syntax tree for Python MoE forward pass",
            "Generate pytest test cases for pipeline tensor serialization",
            "Profile memory bandwidth consumption of expert routing weights",
            "Verify VPC firewall egress rules for node-to-node TCP mesh",
        ]
        res = asyncio.run(harness.run_agent_swarm(tasks=tasks, tokens_per_task=args.tokens))
        print("\nAgent Swarm Result:")
        print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
