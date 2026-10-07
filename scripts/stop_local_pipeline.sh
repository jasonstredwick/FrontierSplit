#!/usr/bin/env bash
# FrontierSplit: Stop all local pipeline workers and gateway
set -euo pipefail

PID_FILE="/tmp/frontiersplit_local_pids.txt"

if [[ -f "${PID_FILE}" ]]; then
  echo "Stopping FrontierSplit local processes..."
  while read -r pid; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done < "${PID_FILE}"
  rm -f "${PID_FILE}"
fi

# Fallback cleanup on standard ports
pkill -f "frontiersplit.worker" 2>/dev/null || true
pkill -f "frontiersplit.gateway" 2>/dev/null || true

echo "=== FrontierSplit Local Pipeline Stopped ==="
