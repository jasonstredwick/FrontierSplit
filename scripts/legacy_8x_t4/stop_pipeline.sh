#!/usr/bin/env bash
# FrontierSplit: Stop pipeline workers and gateway across the cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=== Stopping FrontierSplit Services on Cluster ==="

for i in $(seq 1 ${NUM_NODES}); do
    NODE="${NODE_PREFIX}-${i}"
    echo "Stopping services on ${NODE}..."
    ${GCLOUD} compute ssh "${NODE}" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
      --command="systemctl --user stop fs-worker fs-gateway 2>/dev/null || true" &
done

wait
echo "All FrontierSplit services stopped."
