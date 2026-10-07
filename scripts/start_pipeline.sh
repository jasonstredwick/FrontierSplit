#!/usr/bin/env bash
# FrontierSplit: Start pipeline workers and gateway across the 4-node cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=== Starting FrontierSplit Services on Cluster (${NUM_NODES} Nodes in ${ZONE}) ==="

# Discover Internal IPs
echo "Discovering cluster internal IPs..."
NODE4_IP=$(${GCLOUD} compute instances describe "${NODE_PREFIX}-4" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(networkInterfaces[0].networkIP)")
NODE3_IP=$(${GCLOUD} compute instances describe "${NODE_PREFIX}-3" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(networkInterfaces[0].networkIP)")
NODE2_IP=$(${GCLOUD} compute instances describe "${NODE_PREFIX}-2" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(networkInterfaces[0].networkIP)")

# Node 4 (Stage 3 - Final Stage: LM Head)
echo "Starting Stage 3 on ${NODE_PREFIX}-4..."
${GCLOUD} compute ssh "${NODE_PREFIX}-4" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="git -C /opt/FrontierSplit pull origin main && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=3 --total-stages=4 --port=50051"

# Node 3 (Stage 2 - Intermediate Stage)
echo "Starting Stage 2 on ${NODE_PREFIX}-3..."
${GCLOUD} compute ssh "${NODE_PREFIX}-3" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="git -C /opt/FrontierSplit pull origin main && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=2 --total-stages=4 --port=50051 --downstream-url=http://${NODE4_IP}:50051"

# Node 2 (Stage 1 - Intermediate Stage)
echo "Starting Stage 1 on ${NODE_PREFIX}-2..."
${GCLOUD} compute ssh "${NODE_PREFIX}-2" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="git -C /opt/FrontierSplit pull origin main && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=1 --total-stages=4 --port=50051 --downstream-url=http://${NODE3_IP}:50051"

# Node 1 (Stage 0 + Ingress Gateway)
echo "Starting Stage 0 and Gateway on ${NODE_PREFIX}-1..."
${GCLOUD} compute ssh "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="git -C /opt/FrontierSplit pull origin main && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=0 --total-stages=4 --port=50051 --downstream-url=http://${NODE2_IP}:50051 && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-gateway python3 -m frontiersplit.gateway --stage0-url=http://localhost:50051 --port=8000 --total-stages=4"

GATEWAY_IP=$(${GCLOUD} compute instances describe "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo "=== FrontierSplit Pipeline Online! ==="
echo "Ingress Gateway: http://${GATEWAY_IP}:8000/v1/chat/completions"
echo "Telemetry URL:   http://${GATEWAY_IP}:8000/v1/telemetry"
