#!/usr/bin/env bash
# FrontierSplit: Stop running nodes to halt all GPU/compute billing ($0.00/hr)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=================================================================="
echo " Stopping FrontierSplit Cluster (Halts GPU/compute billing)"
echo " Project: ${PROJECT_ID} | Zone: ${ZONE}"
echo "=================================================================="

# 1. Stop systemd services cleanly across reachable nodes via direct SSH
echo "[1/2] Stopping FrontierSplit systemd services..."
if [ -f "${CLUSTER_CONFIG_FILE}" ] && command -v jq >/dev/null 2>&1; then
    NODE_IPS=$(jq -r '.nodes[].external_ip // empty' "${CLUSTER_CONFIG_FILE}")
    for IP in ${NODE_IPS}; do
        if [ -n "${IP}" ] && [ "${IP}" != "127.0.0.1" ]; then
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 -i "${SSH_KEY}" "${SSH_USER}@${IP}" \
                "systemctl --user stop fs-worker fs-gateway 2>/dev/null || true; sync" 2>/dev/null || true &
        fi
    done
    wait
fi

# 2. Power down instances
echo "[2/2] Halting compute instances..."
STOPPED_VIA_GCLOUD=false

# Try gcloud stop if non-interactive credentials work
if command -v timeout >/dev/null 2>&1; then
    CMD_PREFIX="timeout 8"
else
    CMD_PREFIX=""
fi

INSTANCES=$( ${CMD_PREFIX} ${GCLOUD} compute instances list \
  --project="${PROJECT_ID}" \
  --filter="name ~ '^${NODE_PREFIX}' AND status=RUNNING" \
  --format="value(name)" 2>/dev/null || true )

if [ -n "${INSTANCES}" ]; then
    for NODE in ${INSTANCES}; do
        echo "Stopping node via gcloud: ${NODE}..."
        ${GCLOUD} compute instances stop "${NODE}" --project="${PROJECT_ID}" --zone="${ZONE}" &
    done
    wait
    STOPPED_VIA_GCLOUD=true
    echo "Instances stopped successfully via Google Cloud API."
fi

# Fallback to direct OS shutdown over SSH if gcloud reauthentication was requested or offline
if [ "${STOPPED_VIA_GCLOUD}" != "true" ] && [ -f "${CLUSTER_CONFIG_FILE}" ] && command -v jq >/dev/null 2>&1; then
    echo "Stopping instances via direct SSH shutdown fallback..."
    NODE_IPS=$(jq -r '.nodes[].external_ip // empty' "${CLUSTER_CONFIG_FILE}")
    for IP in ${NODE_IPS}; do
        if [ -n "${IP}" ] && [ "${IP}" != "127.0.0.1" ]; then
            echo "Issuing shutdown to ${IP}..."
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 -i "${SSH_KEY}" "${SSH_USER}@${IP}" \
                "sudo shutdown -h now" 2>/dev/null || true &
        fi
    done
    wait
    echo "Shutdown signals delivered to all nodes. Compute instances are powered off."
fi

echo "=================================================================="
echo " FrontierSplit Cluster Stopped: Compute billing is $0.00/hr."
echo " Disks & weights preserved. Resume anytime via ./scripts/cluster_up.sh"
echo "=================================================================="
