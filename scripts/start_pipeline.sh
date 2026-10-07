#!/usr/bin/env bash
# FrontierSplit: Start pipeline workers and gateway across the cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=== Starting FrontierSplit Services on Cluster ==="

# Node 3 (Stage 2 - Final Stage)
echo "Starting Stage 2 on frontiersplit-node-3..."
${GCLOUD} compute ssh frontiersplit-node-3 --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="git -C /opt/FrontierSplit pull origin main && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=2 --total-stages=3 --port=50051"

# Node 2 (Stage 1 - Intermediate Stage)
echo "Starting Stage 1 on frontiersplit-node-2..."
${GCLOUD} compute ssh frontiersplit-node-2 --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="git -C /opt/FrontierSplit pull origin main && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=1 --total-stages=3 --port=50051 --downstream-url=http://10.150.0.4:50051"

# Node 1 (Stage 0 + Ingress Gateway)
echo "Starting Stage 0 and Gateway on frontiersplit-node-1..."
${GCLOUD} compute ssh frontiersplit-node-1 --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="git -C /opt/FrontierSplit pull origin main && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=0 --total-stages=3 --port=50051 --downstream-url=http://10.150.0.3:50051 && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-gateway python3 -m frontiersplit.gateway --stage0-url=http://localhost:50051 --port=8000 --total-stages=3"

GATEWAY_IP=$(${GCLOUD} compute instances describe frontiersplit-node-1 --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(networkInterfaces[0].accessConfigs[0].natIP)" || echo "8.234.146.81")

echo "=== FrontierSplit Pipeline Online! ==="
echo "Ingress Gateway: http://${GATEWAY_IP}:8000/v1/chat/completions"
echo "Telemetry URL:   http://${GATEWAY_IP}:8000/v1/telemetry"
