#!/usr/bin/env bash
# FrontierSplit: Check Cloud TPU VM & Queued Resource status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tpu_env.sh"

echo "=================================================================="
echo " FrontierSplit Cloud TPU Status: ${TPU_NAME}"
echo " Queued Resource: ${QUEUED_RESOURCE_ID}"
echo " Zone: ${ZONE} | Project: ${PROJECT_ID}"
echo "=================================================================="

# 1. Queued Resource status
QR_EXISTS=$(${GCLOUD} compute tpus queued-resources list --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(name)" 2>/dev/null | grep -w "${QUEUED_RESOURCE_ID}" || true)
if [ -n "${QR_EXISTS}" ]; then
  QR_STATE=$(${GCLOUD} compute tpus queued-resources describe "${QUEUED_RESOURCE_ID}" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(state.state)" 2>/dev/null || echo "UNKNOWN")
  echo "  Queued Resource State: ${QR_STATE}"
else
  echo "  Queued Resource State: NOT SUBMITTED"
fi

# 2. TPU VM Node status
NODE_EXISTS=$(${GCLOUD} compute tpus tpu-vm list --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(name)" 2>/dev/null | grep -w "${TPU_NAME}" || true)
if [ -z "${NODE_EXISTS}" ]; then
  echo "  TPU VM Node State:     NOT PROVISIONED (Waiting in queue or deleted)"
  echo "=================================================================="
  exit 0
fi

DESC=$(${GCLOUD} compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --format="json")
STATE=$(echo "${DESC}" | jq -r '.state // "UNKNOWN"')
ACCEL=$(echo "${DESC}" | jq -r '.acceleratorType // "UNKNOWN"')
RUNTIME=$(echo "${DESC}" | jq -r '.runtimeVersion // "UNKNOWN"')
EXTERNAL_IP=$(echo "${DESC}" | jq -r '.networkEndpoints[0].accessConfig.externalIp // "NONE"')
INTERNAL_IP=$(echo "${DESC}" | jq -r '.networkEndpoints[0].ipAddress // "NONE"')

echo "  TPU VM Node State:     ${STATE}"
echo "  Accelerator:           ${ACCEL}"
echo "  Runtime Version:       ${RUNTIME}"
echo "  Internal IP:           ${INTERNAL_IP}"
echo "  External IP:           ${EXTERNAL_IP}"
echo "=================================================================="

if [ "${STATE}" = "READY" ] && [ "${EXTERNAL_IP}" != "NONE" ]; then
  echo "Checking API Gateway Health at http://${EXTERNAL_IP}:${GATEWAY_PORT}/health..."
  CURL_OUT=$(curl -s "http://${EXTERNAL_IP}:${GATEWAY_PORT}/health" --connect-timeout 3 2>/dev/null || true)
  if [ -n "${CURL_OUT}" ]; then
    echo "  Gateway Response: ${CURL_OUT}"
  else
    echo "  Gateway is not responding (services may be stopped or initializing)."
  fi
fi
