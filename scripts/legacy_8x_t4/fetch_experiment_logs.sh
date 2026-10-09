#!/usr/bin/env bash
# FrontierSplit: Fetch logs from all cluster nodes into an experiment directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

EXP_ID="${1:-20261007_mistral7b_fp16_2x_t4}"
TARGET_LOGS_DIR="${ROOT_DIR}/experiments/${EXP_ID}/logs"
mkdir -p "${TARGET_LOGS_DIR}"

echo "=== Fetching Logs for Experiment: ${EXP_ID} ==="
echo "Target directory: ${TARGET_LOGS_DIR}"

# Helper for executing commands on cluster nodes
node_ssh() {
  local node_num="$1"
  local cmd="$2"
  local ip_var="NODE_${node_num}_IP"
  local ip="${!ip_var:-}"

  if [ -f "${SSH_KEY:-}" ] && [ -n "${ip}" ]; then
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${ip}" "${cmd}"
  else
    ${GCLOUD} compute ssh "${NODE_PREFIX}-${node_num}" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
      --command="${cmd}"
  fi
}

# Node 1: Fetch gateway and worker logs
echo "Fetching Gateway and Stage 0 logs from ${NODE_PREFIX}-1..."
node_ssh 1 "journalctl --user-unit=fs-gateway -n 500 --no-pager" > "${TARGET_LOGS_DIR}/gateway.log" 2>&1 || true
node_ssh 1 "journalctl --user-unit=fs-worker -n 500 --no-pager" > "${TARGET_LOGS_DIR}/node_0_worker.log" 2>&1 || true

# Other nodes: worker logs
for i in $(seq 2 ${NUM_NODES}); do
  STAGE_ID=$((i - 1))
  echo "Fetching Stage ${STAGE_ID} logs from ${NODE_PREFIX}-${i}..."
  node_ssh "${i}" "journalctl --user-unit=fs-worker -n 500 --no-pager" > "${TARGET_LOGS_DIR}/node_${STAGE_ID}_worker.log" 2>&1 || true
done

echo "=== Logs fetched successfully to ${TARGET_LOGS_DIR} ==="
ls -lh "${TARGET_LOGS_DIR}"

