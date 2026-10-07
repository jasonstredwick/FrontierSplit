#!/usr/bin/env bash
# ==============================================================================
# FrontierSplit: Unified Benchmark Execution Wrapper
# ==============================================================================
# Eliminates long, custom CLI invocations by auto-wiring cluster configuration
# from scripts/env.sh into benchmarks.run_evals.
#
# Usage:
#   ./scripts/run_benchmarks.sh [BENCHMARK] [NUM_RUNS] [CONCURRENCY] [EXP_ID]
#
# Examples:
#   ./scripts/run_benchmarks.sh                  # Runs 'all' benchmarks, 3 runs, M=4
#   ./scripts/run_benchmarks.sh ifeval 1         # Quick single-pass IFEval
#   ./scripts/run_benchmarks.sh swebench 3       # 3-run SWE-bench Lite
#   ./scripts/run_benchmarks.sh all 5 4          # 5-run statistical suite, M=4
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source cluster environment
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
  source "${SCRIPT_DIR}/env.sh"
else
  echo "Error: ${SCRIPT_DIR}/env.sh not found."
  exit 1
fi

# Positional arguments with smart defaults
BENCHMARK="${1:-all}"
NUM_RUNS="${2:-3}"
CONCURRENCY="${3:-4}"
EXP_ID="${4:-20261007_mistral7b_fp16_2x_t4}"

# Configuration (sourced dynamically from cluster_config.json via env.sh)
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
BASE_URL="http://${NODE_1_IP}:${GATEWAY_PORT}/v1"
GATEWAY_URL="http://${NODE_1_IP}:${GATEWAY_PORT}"
MODEL="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
EXPERIMENT_DIR="${ROOT_DIR}/experiments/${EXP_ID}"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
  VENV_PYTHON="python3"
fi

echo "=============================================================================="
echo " FrontierSplit Benchmark Suite: ${BENCHMARK} (Runs: ${NUM_RUNS}, Concurrency: ${CONCURRENCY})"
echo " Target Gateway:  ${BASE_URL}"
echo " Model:           ${MODEL}"
echo " Experiment Dir:  ${EXPERIMENT_DIR}"
echo "=============================================================================="

# Execute unified runner
exec "${VENV_PYTHON}" -m benchmarks.run_evals \
  --base-url "${BASE_URL}" \
  --gateway-url "${GATEWAY_URL}" \
  --model "${MODEL}" \
  --benchmarks "${BENCHMARK}" \
  --concurrency "${CONCURRENCY}" \
  --num-runs "${NUM_RUNS}" \
  --experiment-dir "${EXPERIMENT_DIR}"
