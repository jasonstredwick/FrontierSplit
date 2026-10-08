#!/usr/bin/env bash
# FrontierSplit: Stop/Delete Cloud TPU VM (Halts compute billing to $0.00/hr)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tpu_env.sh"

echo "=================================================================="
echo " Stopping FrontierSplit Cloud TPU VM & Queued Resource"
echo " Queued Resource: ${QUEUED_RESOURCE_ID} | Node: ${TPU_NAME}"
echo " Zone: ${ZONE} | Project: ${PROJECT_ID}"
echo "=================================================================="

# Check and delete TPU VM node if exists
EXISTING_NODE=$(${GCLOUD} compute tpus tpu-vm list --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(name)" 2>/dev/null | grep -w "${TPU_NAME}" || true)
if [ -n "${EXISTING_NODE}" ]; then
  echo "Deleting TPU VM '${TPU_NAME}'..."
  ${GCLOUD} compute tpus tpu-vm delete "${TPU_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --quiet || true
fi

# Check and delete Queued Resource if exists
EXISTING_QR=$(${GCLOUD} compute tpus queued-resources list --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(name)" 2>/dev/null | grep -w "${QUEUED_RESOURCE_ID}" || true)
if [ -n "${EXISTING_QR}" ]; then
  echo "Deleting Queued Resource '${QUEUED_RESOURCE_ID}'..."
  ${GCLOUD} compute tpus queued-resources delete "${QUEUED_RESOURCE_ID}" --zone="${ZONE}" --project="${PROJECT_ID}" --quiet || true
fi

echo "=================================================================="
echo " FrontierSplit TPU Resources Cleared: Compute billing is \$0.00/hr."
echo " Re-provision anytime via ./scripts/tpu_up.sh"
echo "=================================================================="
