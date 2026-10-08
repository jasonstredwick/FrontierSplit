#!/usr/bin/env bash
# FrontierSplit: Spin up Cloud TPU VM via Queued Resources (v5e / v5litepod-8)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tpu_env.sh"

echo "=================================================================="
echo " Launching FrontierSplit Cloud TPU VM via Queued Resources"
echo " Queued Resource: ${QUEUED_RESOURCE_ID} | Node: ${TPU_NAME}"
echo " Accelerator:     ${ACCELERATOR_TYPE} (8 TPU v5e Chips)"
echo " Runtime Version: ${RUNTIME_VERSION}"
echo " Zone:            ${ZONE} | Project: ${PROJECT_ID}"
echo "=================================================================="

EXISTING_QR=$(${GCLOUD} compute tpus queued-resources list --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(name)" 2>/dev/null | grep -w "${QUEUED_RESOURCE_ID}" || true)

if [ -n "${EXISTING_QR}" ]; then
  QR_STATE=$(${GCLOUD} compute tpus queued-resources describe "${QUEUED_RESOURCE_ID}" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(state.state)" 2>/dev/null || echo "UNKNOWN")
  echo "Found existing Queued Resource '${QUEUED_RESOURCE_ID}' (State: ${QR_STATE})..."
else
  echo "Submitting Queued Resource request '${QUEUED_RESOURCE_ID}'..."
  ${GCLOUD} compute tpus queued-resources create "${QUEUED_RESOURCE_ID}" \
    --node-id="${TPU_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --accelerator-type="${ACCELERATOR_TYPE}" \
    --runtime-version="${RUNTIME_VERSION}" \
    --network="${NETWORK}" \
    --tags="${TAGS}" \
    --async
fi

echo "Waiting for Queued Resource '${QUEUED_RESOURCE_ID}' to become ACTIVE..."
for i in $(seq 1 120); do
  CURR_STATE=$(${GCLOUD} compute tpus queued-resources describe "${QUEUED_RESOURCE_ID}" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(state.state)" 2>/dev/null || echo "UNKNOWN")
  if [ "${CURR_STATE}" = "ACTIVE" ]; then
    echo "=== Queued Resource '${QUEUED_RESOURCE_ID}' is ACTIVE! ==="
    break
  elif [ "${CURR_STATE}" = "FAILED" ] || [ "${CURR_STATE}" = "SUSPENDED" ]; then
    echo "Error: Queued Resource reached state ${CURR_STATE}. Description:"
    ${GCLOUD} compute tpus queued-resources describe "${QUEUED_RESOURCE_ID}" --zone="${ZONE}" --project="${PROJECT_ID}"
    exit 1
  fi
  echo "Current state: ${CURR_STATE} (attempt ${i}/120, waiting 5s)..."
  sleep 5
done

# Check status of TPU VM
"${SCRIPT_DIR}/tpu_status.sh"
