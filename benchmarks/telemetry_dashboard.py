"""FrontierSplit Telemetry Dashboard.

Real-time CLI dashboard tracking cluster health, active multi-agent streams,
aggregate throughput (tok/s), step latency, and pipeline bubble elimination.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict
import requests


def render_meter(percentage: float, width: int = 24) -> str:
    """Render a visual ASCII saturation bar."""
    filled = int(round(width * (percentage / 100.0)))
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {percentage:.1f}%"


def display_dashboard(gateway_url: str) -> None:
    """Fetch telemetry from the gateway and display terminal dashboard."""
    base_url = gateway_url.rstrip("/")

    try:
        health_resp = requests.get(f"{base_url}/health", timeout=3)
        health_data = health_resp.json() if health_resp.status_code == 200 else {}
        gateway_status = health_data.get("gateway", "offline")
    except Exception:
        gateway_status = "offline"
        health_data = {}

    try:
        telem_resp = requests.get(f"{base_url}/v1/telemetry", timeout=3)
        telem_data = telem_resp.json() if telem_resp.status_code == 200 else {}
    except Exception:
        telem_data = {}

    active_streams = telem_data.get("active_streams", 0)
    total_stages = telem_data.get("total_stages", 4)
    total_submitted = telem_data.get("total_submitted", 0)
    total_completed = telem_data.get("total_completed", 0)
    tokens_generated = telem_data.get("total_tokens_generated", 0)
    throughput = telem_data.get("throughput_tokens_per_sec", 0.0)
    avg_latency = telem_data.get("avg_step_latency_ms", 0.0)
    bubble_fraction = telem_data.get("theoretical_bubble_fraction", 0.75)
    bubble_elimination = telem_data.get("bubble_elimination_pct", 25.0)

    meter_str = render_meter(bubble_elimination)

    status_color = "\033[92m" if gateway_status == "healthy" else "\033[91m"
    reset_color = "\033[0m"
    bold = "\033[1m"

    output = f"""
{bold}========================================================================={reset_color}
{bold}          FRONTIERSPLIT DISTRIBUTED PIPELINE TELEMETRY                  {reset_color}
{bold}========================================================================={reset_color}
  Gateway Status:      {status_color}{gateway_status.upper()}{reset_color}  (URL: {base_url})
  Pipeline Topology:   {total_stages} Stages (PP) across cluster nodes
-------------------------------------------------------------------------
{bold}  MULTI-AGENT CONCURRENCY & STREAMING{reset_color}
  Active Streams (M):  {bold}{active_streams}{reset_color} concurrent requests
  Total Submitted:     {total_submitted} requests
  Total Completed:     {total_completed} requests
-------------------------------------------------------------------------
{bold}  CLUSTER THROUGHPUT & LATENCY{reset_color}
  Tokens Generated:    {bold}{tokens_generated}{reset_color} tokens
  Throughput:          {bold}{throughput} tokens/sec{reset_color} (aggregate)
  Avg Step Latency:    {avg_latency:.2f} ms / step
-------------------------------------------------------------------------
{bold}  PIPELINE BUBBLE ELIMINATION{reset_color}
  Bubble Idle Fraction (F_bubble):  {bubble_fraction * 100:.1f}%
  Hardware Saturation:              {meter_str}
{bold}========================================================================={reset_color}
"""
    print(output)


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Telemetry Monitor")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8000", help="Gateway URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    if args.once:
        display_dashboard(args.gateway_url)
        return

    print(f"Monitoring FrontierSplit at {args.gateway_url} (Ctrl+C to quit)...")
    try:
        while True:
            # Clear terminal screen
            sys.stdout.write("\033[2J\033[H")
            display_dashboard(args.gateway_url)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting telemetry monitor.")


if __name__ == "__main__":
    main()
