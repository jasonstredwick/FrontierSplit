#!/usr/bin/env bash
# FrontierSplit: Stop running nodes to halt all GPU/compute billing ($0.00/hr)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=================================================================="
echo " Stopping FrontierSplit Cluster (Halts GPU/compute billing)"
echo " Project: ${PROJECT_ID} | Zone: ${ZONE}"
echo "=================================================================="

INSTANCES=$(${GCLOUD} compute instances list \
  --project="${PROJECT_ID}" \
  --filter="name ~ '^${NODE_PREFIX}' AND status=RUNNING" \
  --format="value(name)" || true)

if [ -z "${INSTANCES}" ]; then
    echo "No running cluster instances found to stop."
    exit 0
fi

for NODE in ${INSTANCES}; do
    echo "Stopping node: ${NODE}..."
    ${GCLOUD} compute instances stop "${NODE}" --project="${PROJECT_ID}" --zone="${ZONE}" &
done

wait
echo "All running instances stopped successfully."
"${SCRIPT_DIR}/cluster_status.sh"
