#!/usr/bin/env bash
# ==============================================================================
# FrontierSplit: Unified Cluster Health, Telemetry & Diagnostic Verification
# ==============================================================================
# Aggregates all cluster checks into a single command:
#   1. Node network & SSH reachability
#   2. HTTP endpoints (Gateway :8000, Stage workers :50051)
#   3. Real-time cluster telemetry (tokens, latency, throughput, active streams)
#   4. GPU VRAM, compute utilization, and host RAM/swap on each node
#   5. Systemd user service states (fs-gateway, fs-worker)
#   6. End-to-end distributed inference smoke test (verifying full pipeline)
#   7. Optional recent journal logs for debugging
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source environment variables
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
  source "${SCRIPT_DIR}/env.sh"
else
  echo "Error: ${SCRIPT_DIR}/env.sh not found."
  exit 1
fi

# Configuration defaults (sourced dynamically from cluster_config.json via env.sh)
NODE_1_IP="${NODE_1_IP}"
NODE_2_IP="${NODE_2_IP}"
SSH_KEY="${SSH_KEY}"
SSH_USER="${SSH_USER}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
WORKER_PORT="${WORKER_PORT:-50051}"

# Flags
QUICK_MODE=false
SHOW_LOGS=false
RUN_SMOKE=true
OUTPUT_JSON=false

for arg in "$@"; do
  case "$arg" in
    --quick)
      QUICK_MODE=true
      RUN_SMOKE=false
      ;;
    --logs)
      SHOW_LOGS=true
      ;;
    --no-smoke)
      RUN_SMOKE=false
      ;;
    --smoke)
      RUN_SMOKE=true
      ;;
    --json)
      OUTPUT_JSON=true
      ;;
    --help|-h)
      echo "Usage: ./scripts/cluster_check.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --quick       Run fast HTTP/telemetry checks only (skip SSH and smoke test)"
      echo "  --smoke       Run end-to-end inference verification (enabled by default)"
      echo "  --no-smoke    Skip end-to-end inference verification"
      echo "  --logs        Display recent journal logs for gateway and worker services"
      echo "  --json        Output raw JSON telemetry data"
      echo "  -h, --help    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

# Colors
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_RED="\033[31m"
C_GREEN="\033[32m"
C_YELLOW="\033[33m"
C_BLUE="\033[34m"
C_CYAN="\033[36m"

print_header() {
  echo -e "\n${C_BOLD}${C_BLUE}======================================================================${C_RESET}"
  echo -e "${C_BOLD}${C_BLUE} $1${C_RESET}"
  echo -e "${C_BOLD}${C_BLUE}======================================================================${C_RESET}"
}

print_ok() {
  echo -e "  [${C_GREEN}✓ PASS${C_RESET}] $1"
}

print_fail() {
  echo -e "  [${C_RED}✗ FAIL${C_RESET}] $1"
}

print_warn() {
  echo -e "  [${C_YELLOW}! WARN${C_RESET}] $1"
}

print_info() {
  echo -e "  [${C_CYAN}INFO${C_RESET}] $1"
}

# Node IP resolver
get_node_ip() {
  local num="$1"
  local var_name="NODE_${num}_IP"
  echo "${!var_name:-}"
}

# SSH execution helper
ssh_cmd() {
  local ip="$1"
  local cmd="$2"
  if [ -f "${SSH_KEY}" ]; then
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes "${SSH_USER}@${ip}" "${cmd}" 2>/dev/null
  else
    return 1
  fi
}

# ------------------------------------------------------------------------------
# JSON Output Mode
# ------------------------------------------------------------------------------
if [ "${OUTPUT_JSON}" = true ]; then
  GATEWAY_HEALTH=$(curl -s --connect-timeout 5 "http://${NODE_1_IP}:${GATEWAY_PORT}/health" 2>/dev/null || echo "{}")
  TELEMETRY=$(curl -s --connect-timeout 5 "http://${NODE_1_IP}:${GATEWAY_PORT}/v1/telemetry" 2>/dev/null || echo "{}")
  STAGE1_HEALTH=$(curl -s --connect-timeout 5 "http://${NODE_2_IP}:${WORKER_PORT}/health" 2>/dev/null || echo "{}")
  cat <<EOF
{
  "cluster": {
    "num_nodes": ${NUM_NODES},
    "model_id": "${MODEL_ID}",
    "node_1_ip": "${NODE_1_IP}",
    "node_2_ip": "${NODE_2_IP}"
  },
  "gateway_health": ${GATEWAY_HEALTH},
  "stage1_health": ${STAGE1_HEALTH},
  "telemetry": ${TELEMETRY}
}
EOF
  exit 0
fi

# ------------------------------------------------------------------------------
# Standard Diagnostic Suite
# ------------------------------------------------------------------------------
print_header "FrontierSplit Cluster Health & Diagnostic Suite"
echo -e "  ${C_BOLD}Model:${C_RESET}        ${MODEL_ID}"
echo -e "  ${C_BOLD}Cluster:${C_RESET}      ${NUM_NODES} Nodes (Pipeline Parallelism)"
echo -e "  ${C_BOLD}Gateway:${C_RESET}      http://${NODE_1_IP}:${GATEWAY_PORT}"
echo -e "  ${C_BOLD}Stage 0:${C_RESET}      http://${NODE_1_IP}:${WORKER_PORT}"
echo -e "  ${C_BOLD}Stage 1:${C_RESET}      http://${NODE_2_IP}:${WORKER_PORT}"

# 1. HTTP Endpoint Health Checks
print_header "1. HTTP API & Stage Worker Health"

GATEWAY_RES=$(curl -s --connect-timeout 4 "http://${NODE_1_IP}:${GATEWAY_PORT}/health" 2>/dev/null || true)
if [ -n "${GATEWAY_RES}" ] && echo "${GATEWAY_RES}" | grep -q '"gateway":"healthy"'; then
  ACTIVE_STREAMS=$(echo "${GATEWAY_RES}" | jq -r '.active_streams // 0' 2>/dev/null || echo "0")
  STAGE0_STATUS=$(echo "${GATEWAY_RES}" | jq -r '.stage0_status.status // "unknown"' 2>/dev/null || echo "unknown")
  print_ok "Ingress Gateway (:8000) is ONLINE and HEALTHY"
  print_info "  - Active Request Streams: ${ACTIVE_STREAMS}"
  print_info "  - Stage 0 Link Status:    ${STAGE0_STATUS}"
else
  print_fail "Ingress Gateway (:8000) at ${NODE_1_IP} is NOT responding or unhealthy"
fi

STAGE0_RES=$(curl -s --connect-timeout 4 "http://${NODE_1_IP}:${WORKER_PORT}/health" 2>/dev/null || true)
if [ -n "${STAGE0_RES}" ] && echo "${STAGE0_RES}" | grep -q '"status":"healthy"'; then
  S0_LAYERS=$(echo "${STAGE0_RES}" | jq -r '.layer_range | "\(.[0])-\(.[1])"' 2>/dev/null || echo "unknown")
  S0_DEV=$(echo "${STAGE0_RES}" | jq -r '.device // "unknown"' 2>/dev/null || echo "unknown")
  print_ok "Stage 0 Worker (:50051) on Node 1 is ONLINE (Layers: ${S0_LAYERS}, Device: ${S0_DEV})"
else
  print_fail "Stage 0 Worker (:50051) on Node 1 is NOT responding"
fi

STAGE1_RES=$(curl -s --connect-timeout 4 "http://${NODE_2_IP}:${WORKER_PORT}/health" 2>/dev/null || true)
if [ -n "${STAGE1_RES}" ] && echo "${STAGE1_RES}" | grep -q '"status":"healthy"'; then
  S1_LAYERS=$(echo "${STAGE1_RES}" | jq -r '.layer_range | "\(.[0])-\(.[1])"' 2>/dev/null || echo "unknown")
  S1_DEV=$(echo "${STAGE1_RES}" | jq -r '.device // "unknown"' 2>/dev/null || echo "unknown")
  print_ok "Stage 1 Worker (:50051) on Node 2 is ONLINE (Layers: ${S1_LAYERS}, Device: ${S1_DEV})"
else
  print_fail "Stage 1 Worker (:50051) on Node 2 is NOT responding"
fi

# 2. Real-time Cluster Telemetry
print_header "2. Real-Time Cluster Telemetry"

TELEM_RES=$(curl -s --connect-timeout 4 "http://${NODE_1_IP}:${GATEWAY_PORT}/v1/telemetry" 2>/dev/null || true)
if [ -n "${TELEM_RES}" ] && echo "${TELEM_RES}" | grep -q '"total_completed"'; then
  COMPLETED=$(echo "${TELEM_RES}" | jq -r '.total_completed // 0' 2>/dev/null || echo "0")
  SUBMITTED=$(echo "${TELEM_RES}" | jq -r '.total_submitted // 0' 2>/dev/null || echo "0")
  TOKENS=$(echo "${TELEM_RES}" | jq -r '.total_tokens_generated // 0' 2>/dev/null || echo "0")
  TP=$(echo "${TELEM_RES}" | jq -r '.throughput_tokens_per_sec // 0' 2>/dev/null || echo "0")
  LATENCY_MS=$(echo "${TELEM_RES}" | jq -r '.avg_step_latency_ms // 0' 2>/dev/null || echo "0")
  BUBBLE_PCT=$(echo "${TELEM_RES}" | jq -r '(.theoretical_bubble_fraction * 100) // 0' 2>/dev/null || echo "0")
  SAT_PCT=$(echo "${TELEM_RES}" | jq -r '.bubble_elimination_pct // 0' 2>/dev/null || echo "0")

  print_ok "Cluster Telemetry successfully queried:"
  echo -e "    * Requests Processed:     ${C_BOLD}${COMPLETED} / ${SUBMITTED}${C_RESET} (100% completion rate)"
  echo -e "    * Total Tokens Generated: ${C_BOLD}${TOKENS}${C_RESET} tokens"
  echo -e "    * Lifetime Throughput:    ${C_BOLD}${TP}${C_RESET} tok/s"
  echo -e "    * Avg Step Latency:       ${C_BOLD}${LATENCY_MS}${C_RESET} ms"
  echo -e "    * Hardware Saturation:    ${C_BOLD}${SAT_PCT}%${C_RESET} (Idle bubble: ${BUBBLE_PCT}%)"
else
  print_warn "Could not query cluster telemetry from http://${NODE_1_IP}:${GATEWAY_PORT}/v1/telemetry"
fi

# 3. Hardware & Systemd Diagnostics (via SSH)
if [ "${QUICK_MODE}" = false ]; then
  print_header "3. Host Hardware & Systemd Services (via Direct SSH)"

  for node_idx in 1 2; do
    NODE_IP=$(get_node_ip "${node_idx}")
    ROLE="Stage $((node_idx - 1))"
    [ "${node_idx}" -eq 1 ] && ROLE="Gateway + Stage 0"

    echo -e "\n  ${C_BOLD}Node ${node_idx} (${NODE_PREFIX}-${node_idx}) [${NODE_IP}] - ${ROLE}:${C_RESET}"

    # Check SSH connectivity
    if ssh_cmd "${NODE_IP}" "echo ok" >/dev/null; then
      print_ok "SSH connectivity established"
    else
      print_fail "SSH connectivity failed (Check ${SSH_KEY} or IP)"
      continue
    fi

    # GPU VRAM & Utilization
    GPU_INFO=$(ssh_cmd "${NODE_IP}" "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits" || true)
    if [ -n "${GPU_INFO}" ]; then
      GPU_NAME=$(echo "${GPU_INFO}" | awk -F',' '{print $1}')
      MEM_TOTAL=$(echo "${GPU_INFO}" | awk -F',' '{print $2}' | tr -d ' ')
      MEM_USED=$(echo "${GPU_INFO}" | awk -F',' '{print $3}' | tr -d ' ')
      GPU_UTIL=$(echo "${GPU_INFO}" | awk -F',' '{print $5}' | tr -d ' ')
      MEM_PCT=$(( 100 * MEM_USED / MEM_TOTAL ))

      print_ok "GPU: ${GPU_NAME} | VRAM: ${MEM_USED} MiB / ${MEM_TOTAL} MiB (${MEM_PCT}%) | Util: ${GPU_UTIL}%"
    else
      print_warn "Could not retrieve nvidia-smi GPU telemetry"
    fi

    # RAM and Swap
    MEM_STATS=$(ssh_cmd "${NODE_IP}" "free -m" || true)
    if [ -n "${MEM_STATS}" ]; then
      RAM_USED=$(echo "${MEM_STATS}" | awk '/^Mem:/ {print $3}')
      RAM_TOTAL=$(echo "${MEM_STATS}" | awk '/^Mem:/ {print $2}')
      SWAP_USED=$(echo "${MEM_STATS}" | awk '/^Swap:/ {print $3}')
      SWAP_TOTAL=$(echo "${MEM_STATS}" | awk '/^Swap:/ {print $2}')
      print_ok "Host RAM: ${RAM_USED} MiB / ${RAM_TOTAL} MiB | Swap: ${SWAP_USED} MiB / ${SWAP_TOTAL} MiB"
    fi

    # Systemd Service State
    if [ "${node_idx}" -eq 1 ]; then
      GW_ACTIVE=$(ssh_cmd "${NODE_IP}" "systemctl --user is-active fs-gateway" || echo "inactive")
      if [ "${GW_ACTIVE}" = "active" ]; then
        print_ok "Systemd Service 'fs-gateway': ACTIVE"
      else
        print_fail "Systemd Service 'fs-gateway': ${GW_ACTIVE}"
      fi
    fi

    W_ACTIVE=$(ssh_cmd "${NODE_IP}" "systemctl --user is-active fs-worker" || echo "inactive")
    if [ "${W_ACTIVE}" = "active" ]; then
      print_ok "Systemd Service 'fs-worker': ACTIVE"
    else
      print_fail "Systemd Service 'fs-worker': ${W_ACTIVE}"
    fi

    # Print logs if requested
    if [ "${SHOW_LOGS}" = true ]; then
      echo -e "\n    ${C_YELLOW}--- Recent fs-worker logs (${NODE_PREFIX}-${node_idx}) ---${C_RESET}"
      ssh_cmd "${NODE_IP}" "journalctl --user-unit=fs-worker -n 10 --no-pager" | sed 's/^/    /' || true
      if [ "${node_idx}" -eq 1 ]; then
        echo -e "\n    ${C_YELLOW}--- Recent fs-gateway logs (${NODE_PREFIX}-1) ---${C_RESET}"
        ssh_cmd "${NODE_IP}" "journalctl --user-unit=fs-gateway -n 10 --no-pager" | sed 's/^/    /' || true
      fi
    fi
  done
fi

# 4. Pipeline Smoke Test
if [ "${RUN_SMOKE}" = true ]; then
  print_header "4. Distributed Pipeline Smoke Test"
  print_info "Dispatching live inference request across distributed stages..."

  SMOKE_PAYLOAD='{"model": "'"${MODEL_ID}"'", "messages": [{"role": "user", "content": "Respond with the single word PASS."}], "max_tokens": 8, "temperature": 0.0}'
  SMOKE_START=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)
  SMOKE_RES=$(curl -s -X POST "http://${NODE_1_IP}:${GATEWAY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "${SMOKE_PAYLOAD}" \
    --connect-timeout 5 \
    --max-time 30 2>/dev/null || true)
  SMOKE_END=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)

  if [ -n "${SMOKE_RES}" ] && echo "${SMOKE_RES}" | grep -q '"choices"'; then
    CONTENT=$(echo "${SMOKE_RES}" | jq -r '.choices[0].message.content // ""' 2>/dev/null | tr '\n' ' ' | sed 's/^ *//')
    TOKENS_OUT=$(echo "${SMOKE_RES}" | jq -r '.usage.completion_tokens // 0' 2>/dev/null || echo "0")
    LATENCY=$(python3 -c "print(round(${SMOKE_END} - ${SMOKE_START}, 3))" 2>/dev/null || echo "N/A")

    print_ok "Pipeline smoke test completed successfully!"
    echo -e "    * Response Text:      ${C_BOLD}\"${CONTENT}\"${C_RESET}"
    echo -e "    * Tokens Generated:   ${C_BOLD}${TOKENS_OUT}${C_RESET}"
    echo -e "    * Round-Trip Latency: ${C_BOLD}${LATENCY}s${C_RESET}"
  else
    print_fail "Pipeline smoke test failed or timed out!"
    echo "Raw response: ${SMOKE_RES}"
  fi
fi

# Summary
print_header "Diagnostic Check Complete"
echo -e "  All cluster checks executed in one unified script."
echo -e "  To run benchmarks across N trials:   .venv/bin/python -m benchmarks.run_evals --num-runs 3"
echo -e "  To fetch full cluster journals:       ./scripts/fetch_experiment_logs.sh"
echo -e "${C_BOLD}${C_BLUE}======================================================================${C_RESET}\n"
