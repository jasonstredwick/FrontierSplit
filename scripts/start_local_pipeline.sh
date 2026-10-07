#!/usr/bin/env bash
# FrontierSplit: Start a full 4-stage pipeline and Ingress Gateway locally
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

PID_FILE="/tmp/frontiersplit_local_pids.txt"
rm -f "${PID_FILE}"

echo "=== Starting FrontierSplit 4-Stage Pipeline Locally ==="

# Stage 3 (Final Stage - LM Head)
echo "Starting Stage 3 (LM Head) on port 50054..."
${PYTHON} -m frontiersplit.worker --stage-id=3 --total-stages=4 --port=50054 --host=127.0.0.1 > /tmp/fs_stage3.log 2>&1 &
echo $! >> "${PID_FILE}"

# Stage 2 (Intermediate)
echo "Starting Stage 2 on port 50053..."
${PYTHON} -m frontiersplit.worker --stage-id=2 --total-stages=4 --port=50053 --host=127.0.0.1 --downstream-url=http://127.0.0.1:50054 > /tmp/fs_stage2.log 2>&1 &
echo $! >> "${PID_FILE}"

# Stage 1 (Intermediate)
echo "Starting Stage 1 on port 50052..."
${PYTHON} -m frontiersplit.worker --stage-id=1 --total-stages=4 --port=50052 --host=127.0.0.1 --downstream-url=http://127.0.0.1:50053 > /tmp/fs_stage1.log 2>&1 &
echo $! >> "${PID_FILE}"

# Stage 0 (Embeddings + Layers)
echo "Starting Stage 0 on port 50051..."
${PYTHON} -m frontiersplit.worker --stage-id=0 --total-stages=4 --port=50051 --host=127.0.0.1 --downstream-url=http://127.0.0.1:50052 > /tmp/fs_stage0.log 2>&1 &
echo $! >> "${PID_FILE}"

# Ingress Gateway
echo "Starting Ingress Gateway on port 8000..."
${PYTHON} -m frontiersplit.gateway --stage0-url=http://127.0.0.1:50051 --port=8000 --host=127.0.0.1 --total-stages=4 > /tmp/fs_gateway.log 2>&1 &
echo $! >> "${PID_FILE}"

echo "Waiting for services to initialize..."
for i in {1..10}; do
  if curl -s http://127.0.0.1:8000/health | grep -q '"gateway":"healthy"'; then
    echo "=== FrontierSplit Local Pipeline is Online and Healthy! ==="
    echo "Gateway: http://127.0.0.1:8000/v1/chat/completions"
    echo "Telemetry: http://127.0.0.1:8000/v1/telemetry"
    exit 0
  fi
  sleep 1
done

echo "Error: Pipeline failed to reach healthy status. Checking logs:"
cat /tmp/fs_gateway.log || true
exit 1
