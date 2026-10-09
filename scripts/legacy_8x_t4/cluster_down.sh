#!/usr/bin/env bash
# FrontierSplit: Completely delete cluster nodes and disks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=================================================================="
echo " TEARDOWN: Deleting FrontierSplit Cluster Instances & Disks"
echo " Project: ${PROJECT_ID} | Zone: ${ZONE}"
echo "=================================================================="

INSTANCES=$(${GCLOUD} compute instances list \
  --project="${PROJECT_ID}" \
  --filter="name ~ '^${NODE_PREFIX}'" \
  --format="value(name)" || true)

if [ -z "${INSTANCES}" ]; then
    echo "No cluster instances found to delete."
    exit 0
fi

for NODE in ${INSTANCES}; do
    echo "Deleting instance: ${NODE}..."
    ${GCLOUD} compute instances delete "${NODE}" --project="${PROJECT_ID}" --zone="${ZONE}" --delete-disks=all --quiet &
done

wait
echo "All cluster instances deleted successfully."
