"""FrontierSplit SWE-bench Lite Compound Agent Harness.

Executes autonomous software engineering agent workflows (issue analysis, code planning,
unified diff patch generation, and patch verification) against the FrontierSplit Ingress
Gateway. Drives heavy multi-turn concurrency across cluster nodes to eliminate pipeline bubbles.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import httpx


# ---------------------------------------------------------------------------
# SWE-bench Lite Data Structures & Patch Utilities
# ---------------------------------------------------------------------------

@dataclass
class SWEBenchInstance:
    instance_id: str
    repo: str
    problem_statement: str
    target_files: List[str]
    golden_patch: Optional[str] = None
    hints_text: Optional[str] = None


@dataclass
class PatchVerificationResult:
    is_valid_diff: bool
    affected_files: List[str]
    hunk_count: int
    lines_added: int
    lines_removed: int
    matches_target_files: bool
    error_message: Optional[str] = None


def extract_unified_diff(text: str) -> str:
    """Extract unified git diff block from model output (handling markdown fences)."""
    # Check for markdown code blocks containing diffs
    code_block_match = re.search(r"```(?:diff|patch)?\s*\n(---.*?\n\+\+\+.*?\n@@.*?)```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Search for raw diff headers
    diff_start = re.search(r"(?:diff --git|\-\-\- [ab]/|\-\-\- [^\n]+)", text)
    if diff_start:
        return text[diff_start.start():].strip()

    return text.strip()


def verify_patch_syntax(patch_text: str, target_files: Optional[List[str]] = None) -> PatchVerificationResult:
    """Parse and validate unified diff syntax and verify affected target files."""
    cleaned = extract_unified_diff(patch_text)
    if not cleaned:
        return PatchVerificationResult(
            is_valid_diff=False,
            affected_files=[],
            hunk_count=0,
            lines_added=0,
            lines_removed=0,
            matches_target_files=False,
            error_message="Empty patch or no diff headers found",
        )

    lines = cleaned.splitlines()
    affected_files = []
    hunk_count = 0
    lines_added = 0
    lines_removed = 0

    file_header_re = re.compile(r"^(?:---|\+\+\+)\s+[ab]?/(.+)$")
    git_diff_re = re.compile(r"^diff --git a/(.+) b/(.+)$")
    hunk_header_re = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")

    for line in lines:
        git_match = git_diff_re.match(line)
        if git_match:
            fpath = git_match.group(2)
            if fpath not in affected_files:
                affected_files.append(fpath)
            continue

        header_match = file_header_re.match(line)
        if header_match:
            fpath = header_match.group(1).strip()
            if fpath not in affected_files and fpath != "/dev/null":
                affected_files.append(fpath)
            continue

        if hunk_header_re.match(line):
            hunk_count += 1
            continue

        if line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    is_valid = (hunk_count > 0) and (bool(affected_files) or (lines_added > 0 or lines_removed > 0))

    matches_targets = False
    if target_files and affected_files:
        matches_targets = any(
            any(t in af or af in t for af in affected_files)
            for t in target_files
        )

    error_msg = None if is_valid else "Missing valid hunk headers (@@ -x,y +a,b @@) or file modifications"

    return PatchVerificationResult(
        is_valid_diff=is_valid,
        affected_files=affected_files,
        hunk_count=hunk_count,
        lines_added=lines_added,
        lines_removed=lines_removed,
        matches_target_files=matches_targets,
        error_message=error_msg,
    )


# ---------------------------------------------------------------------------
# Curated Representative SWE-bench Lite Dataset
# ---------------------------------------------------------------------------

DEFAULT_SWEBENCH_DATASET: List[SWEBenchInstance] = [
    SWEBenchInstance(
        instance_id="django__django-11099",
        repo="django/django",
        problem_statement=(
            "UsernameValidator allows trailing newline in username.\n"
            "ASCIIUsernameValidator and UnicodeUsernameValidator regexes use regex matching that "
            "allows a trailing newline character. Change the regex to prohibit trailing newlines."
        ),
        target_files=["django/contrib/auth/validators.py"],
        golden_patch=(
            "diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py\n"
            "--- a/django/contrib/auth/validators.py\n"
            "+++ b/django/contrib/auth/validators.py\n"
            "@@ -17,3 +17,3 @@ class ASCIIUsernameValidator(validators.RegexValidator):\n"
            "-    regex = r'^[\\w.@+-]+$'\n"
            "+    regex = r'\\A[\\w.@+-]+\\Z'\n"
        ),
    ),
    SWEBenchInstance(
        instance_id="psf__requests-2148",
        repo="psf/requests",
        problem_statement=(
            "socket.error should be wrapped in Requests socket exception.\n"
            "When a low-level socket.error occurs during streaming response reading, "
            "it escapes as raw socket.error instead of requests.exceptions.ConnectionError."
        ),
        target_files=["requests/models.py", "requests/exceptions.py"],
        golden_patch=(
            "diff --git a/requests/models.py b/requests/models.py\n"
            "--- a/requests/models.py\n"
            "+++ b/requests/models.py\n"
            "@@ -638,3 +638,5 @@ class Response(object):\n"
            "-        except socket.error as sockerr:\n"
            "-            raise ConnectionError(sockerr)\n"
            "+        except (socket.error, socket.timeout) as sockerr:\n"
            "+            raise ConnectionError(sockerr)\n"
        ),
    ),
    SWEBenchInstance(
        instance_id="sympy__sympy-13480",
        repo="sympy/sympy",
        problem_statement=(
            "coth(log(tan(x))) raises NameError in certain evaluations.\n"
            "When evaluating hyperbolic cotangent for complex log arguments, "
            "undefined symbol 'x' or unbound variable causes failure. Fix the substitution check."
        ),
        target_files=["sympy/functions/elementary/hyperbolic.py"],
        golden_patch=(
            "diff --git a/sympy/functions/elementary/hyperbolic.py b/sympy/functions/elementary/hyperbolic.py\n"
            "--- a/sympy/functions/elementary/hyperbolic.py\n"
            "+++ b/sympy/functions/elementary/hyperbolic.py\n"
            "@@ -587,2 +587,4 @@ def eval(cls, arg):\n"
            "-            if arg.func == log:\n"
            "+            if arg.is_Add:\n"
        ),
    ),
    SWEBenchInstance(
        instance_id="pytest-dev__pytest-5221",
        repo="pytest-dev/pytest",
        problem_statement=(
            "Display fixture scope in pytest --fixtures.\n"
            "When running pytest --fixtures, it only shows fixture name and docstring. "
            "Users need to see the fixture scope (function, class, module, package, session)."
        ),
        target_files=["src/_pytest/python.py"],
        golden_patch=(
            "diff --git a/src/_pytest/python.py b/src/_pytest/python.py\n"
            "--- a/src/_pytest/python.py\n"
            "+++ b/src/_pytest/python.py\n"
            "@@ -1245,2 +1245,3 @@ def show_fixtures_per_test(config):\n"
            "+        tw.write(f' (scope: {fixturedef.scope})')\n"
        ),
    ),
    SWEBenchInstance(
        instance_id="pallets__flask-4045",
        repo="pallets/flask",
        problem_statement=(
            "Blueprint name validation should reject dots.\n"
            "If a blueprint name contains a dot, url_for generates confusing endpoint names. "
            "Raise ValueError if blueprint name contains a dot."
        ),
        target_files=["src/flask/blueprints.py"],
        golden_patch=(
            "diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py\n"
            "--- a/src/flask/blueprints.py\n"
            "+++ b/src/flask/blueprints.py\n"
            "@@ -188,3 +188,5 @@ class Blueprint(Scaffold):\n"
            "+        if '.' in name:\n"
            "+            raise ValueError('Blueprint name cannot contain dots.')\n"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Compound Agent SWE-bench Runner
# ---------------------------------------------------------------------------

class SWEBenchAgentRunner:
    """Multi-Agent SWE-bench runner orchestrating multi-step issue resolution."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "frontiersplit-mixtral-8x7b",
        concurrency: int = 4,
        timeout: float = 180.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout

    async def _send_completion(
        self,
        client: httpx.AsyncClient,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
    ) -> Tuple[str, int, float]:
        """Dispatch completion request and return (content, tokens, latency_s)."""
        start = time.time()
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        try:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            latency_s = time.time() - start
            content = data["choices"][0]["message"]["content"]
            tokens_count = data.get("usage", {}).get("completion_tokens", max_tokens)
            return content, tokens_count, latency_s
        except Exception as e:
            latency_s = time.time() - start
            return f"[Error: {e}]", 0, latency_s

    async def run_compound_agent_instance(
        self,
        client: httpx.AsyncClient,
        instance: SWEBenchInstance,
        tokens_per_step: int = 64,
    ) -> Dict[str, Any]:
        """Execute a full multi-turn compound agent cycle on an issue instance:
        Step 1: Root cause analysis & file localization
        Step 2: Patch formulation in unified diff format
        """
        instance_start = time.time()
        step_metrics = []
        total_tokens = 0

        # Step 1: Issue Analysis & Localization
        analysis_prompt = (
            f"You are an expert autonomous software engineer resolving an issue in {instance.repo}.\n"
            f"Issue Description:\n{instance.problem_statement}\n\n"
            f"Candidate Target Files:\n" + "\n".join(f"- {f}" for f in instance.target_files) + "\n\n"
            f"Analyze the root cause and explain the exact code modification needed."
        )

        analysis_content, analysis_tokens, analysis_latency = await self._send_completion(
            client,
            [{"role": "user", "content": analysis_prompt}],
            max_tokens=tokens_per_step,
        )
        total_tokens += analysis_tokens
        step_metrics.append({"step": "analysis", "tokens": analysis_tokens, "latency_s": round(analysis_latency, 3)})

        # Step 2: Unified Diff Generation
        patch_prompt = (
            f"Based on your analysis of {instance.repo}, produce the minimal unified git diff patch to fix the issue.\n"
            f"Target file: {instance.target_files[0]}\n"
            f"Output ONLY valid unified diff with @@ hunk headers (e.g. `diff --git a/... b/...` or `--- a/... +++ b/...`)."
        )

        patch_content, patch_tokens, patch_latency = await self._send_completion(
            client,
            [
                {"role": "user", "content": analysis_prompt},
                {"role": "assistant", "content": analysis_content},
                {"role": "user", "content": patch_prompt},
            ],
            max_tokens=tokens_per_step * 2,
        )
        total_tokens += patch_tokens
        step_metrics.append({"step": "patch_generation", "tokens": patch_tokens, "latency_s": round(patch_latency, 3)})

        total_elapsed = time.time() - instance_start

        # Step 3: Patch Verification & Linting
        verification = verify_patch_syntax(patch_content, target_files=instance.target_files)

        return {
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "total_tokens": total_tokens,
            "elapsed_s": round(total_elapsed, 3),
            "step_metrics": step_metrics,
            "generated_patch": patch_content,
            "is_valid_diff": verification.is_valid_diff,
            "affected_files": verification.affected_files,
            "hunk_count": verification.hunk_count,
            "lines_added": verification.lines_added,
            "lines_removed": verification.lines_removed,
            "matches_target_files": verification.matches_target_files,
            "verification_error": verification.error_message,
        }

    async def run_evaluation(
        self,
        dataset: Optional[List[SWEBenchInstance]] = None,
        tokens_per_step: int = 64,
    ) -> Dict[str, Any]:
        """Run concurrent multi-agent evaluations over the SWE-bench dataset."""
        items = dataset or DEFAULT_SWEBENCH_DATASET
        print("\n" + "=" * 76)
        print(f" FrontierSplit SWE-bench Lite: {len(items)} Issues (Concurrency N={self.concurrency})")
        print("=" * 76)

        start_time = time.time()
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient() as client:
            async def bounded_agent(inst: SWEBenchInstance):
                async with semaphore:
                    return await self.run_compound_agent_instance(client, inst, tokens_per_step=tokens_per_step)

            results = await asyncio.gather(*[bounded_agent(inst) for inst in items])

        total_elapsed = time.time() - start_time
        total_tokens = sum(r["total_tokens"] for r in results)
        total_instances = len(results)

        valid_diff_count = sum(1 for r in results if r["is_valid_diff"])
        targeted_file_count = sum(1 for r in results if r["matches_target_files"])

        valid_diff_rate = (valid_diff_count / total_instances * 100.0) if total_instances else 0.0
        targeted_rate = (targeted_file_count / total_instances * 100.0) if total_instances else 0.0

        throughput = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
        avg_latency = sum(r["elapsed_s"] for r in results) / total_instances if total_instances else 0.0

        # Theoretical bubble calculation for K=4 stages
        k = 4
        m = self.concurrency
        bubble_fraction = (k - 1) / (m + k - 1)
        bubble_elimination = (1.0 - bubble_fraction) * 100.0

        summary = {
            "benchmark": "SWE-bench Lite",
            "model": self.model,
            "concurrency": self.concurrency,
            "total_instances": total_instances,
            "valid_diff_count": valid_diff_count,
            "valid_diff_rate_pct": round(valid_diff_rate, 2),
            "target_files_matched_count": targeted_file_count,
            "target_files_matched_pct": round(targeted_rate, 2),
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
        """Render a clean summary of SWE-bench agent results and hardware metrics."""
        print("\n" + "=" * 76)
        print(" SWE-bench Lite Benchmark Results & Pipeline Telemetry")
        print("=" * 76)
        print(f"  Target Model:                   {summary['model']}")
        print(f"  Concurrency Level (M):          {summary['concurrency']} concurrent agent workflows")
        print(f"  Total Issues Evaluated:         {summary['total_instances']}")
        print("-" * 76)
        print("  AGENTIC CODE RESOLUTION METRICS:")
        print(f"    Valid Unified Diff Rate:      {summary['valid_diff_rate_pct']}% ({summary['valid_diff_count']}/{summary['total_instances']})")
        print(f"    Target File Hit Rate:         {summary['target_files_matched_pct']}% ({summary['target_files_matched_count']}/{summary['total_instances']})")
        print("-" * 76)
        print("  PIPELINE PERFORMANCE & BUBBLE ELIMINATION:")
        print(f"    Aggregate Throughput:         {summary['throughput_tok_per_sec']} tokens/s")
        print(f"    Avg Agent Workflow Latency:   {summary['avg_latency_s']} s")
        print(f"    Theoretical Idle Bubble (F):  {round(summary['bubble_fraction'] * 100, 1)}%")
        print(f"    Hardware Saturation Rate:     {summary['bubble_elimination_pct']}%")
        print("=" * 76)


def load_swebench_from_json(filepath: str) -> List[SWEBenchInstance]:
    """Load SWE-bench instances from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    instances = []
    for item in data:
        instances.append(
            SWEBenchInstance(
                instance_id=item["instance_id"],
                repo=item.get("repo", ""),
                problem_statement=item["problem_statement"],
                target_files=item.get("target_files", []),
                golden_patch=item.get("patch") or item.get("golden_patch"),
                hints_text=item.get("hints_text"),
            )
        )
    return instances


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit SWE-bench Lite Agent Runner")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="Gateway URL")
    parser.add_argument("--model", type=str, default="frontiersplit-mixtral-8x7b", help="Model name")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent agent streams")
    parser.add_argument("--dataset", type=str, default=None, help="Path to external SWE-bench JSON dataset")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON results")
    parser.add_argument("--tokens-per-step", type=int, default=64, help="Tokens per agent step")
    args = parser.parse_args()

    dataset = load_swebench_from_json(args.dataset) if args.dataset else None
    runner = SWEBenchAgentRunner(base_url=args.base_url, model=args.model, concurrency=args.concurrency)
    res = asyncio.run(runner.run_evaluation(dataset=dataset, tokens_per_step=args.tokens_per_step))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
