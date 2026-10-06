#!/usr/bin/env bash
# FrontierSplit: Check status of cluster nodes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=================================================================="
echo " FrontierSplit Cluster Status (Project: ${PROJECT_ID}, Zone: ${ZONE})"
echo "=================================================================="

${GCLOUD} compute instances list \
  --project="${PROJECT_ID}" \
  --filter="name ~ '^${NODE_PREFIX}'" \
  --format="table(name,zone,machineType.basename(),status,networkInterfaces[0].networkIP:label=INTERNAL_IP,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP)" || {
    echo "No instances found or error listing instances."
}
