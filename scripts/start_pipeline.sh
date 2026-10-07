#!/usr/bin/env bash
# FrontierSplit: Start pipeline workers and gateway across the cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=== Starting FrontierSplit Services on Cluster (${NUM_NODES} Nodes in ${ZONE}) ==="
echo "Model: ${MODEL_ID}"

# Discover Internal IPs
echo "Discovering cluster internal IPs..."
NODE_IPS=()
for i in $(seq 1 ${NUM_NODES}); do
  NODE_IPS[$i]=$(${GCLOUD} compute instances describe "${NODE_PREFIX}-${i}" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(networkInterfaces[0].networkIP)")
  echo "  ${NODE_PREFIX}-${i}: ${NODE_IPS[$i]}"
done

# Start workers from final stage down to stage 1
FINAL_NODE=${NUM_NODES}
FINAL_STAGE=$((NUM_NODES - 1))

echo "Starting Final Stage (${FINAL_STAGE}) on ${NODE_PREFIX}-${FINAL_NODE}..."
${GCLOUD} compute ssh "${NODE_PREFIX}-${FINAL_NODE}" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="sudo chown -R \$USER:\$USER /opt/FrontierSplit && git config --global --add safe.directory /opt/FrontierSplit && git -C /opt/FrontierSplit pull origin main && pip install -r /opt/FrontierSplit/requirements.txt && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=${FINAL_STAGE} --total-stages=${NUM_NODES} --port=50051 --model-name=${MODEL_ID}"

# Intermediate nodes: from FINAL_NODE-1 down to 2
for i in $(seq $((NUM_NODES - 1)) -1 2); do
  STAGE_ID=$((i - 1))
  NEXT_IP="${NODE_IPS[$((i + 1))]}"
  echo "Starting Stage ${STAGE_ID} on ${NODE_PREFIX}-${i} (downstream -> ${NEXT_IP}:50051)..."
  ${GCLOUD} compute ssh "${NODE_PREFIX}-${i}" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
    --command="sudo chown -R \$USER:\$USER /opt/FrontierSplit && git config --global --add safe.directory /opt/FrontierSplit && git -C /opt/FrontierSplit pull origin main && pip install -r /opt/FrontierSplit/requirements.txt && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=${STAGE_ID} --total-stages=${NUM_NODES} --port=50051 --downstream-url=http://${NEXT_IP}:50051 --model-name=${MODEL_ID}"
done

# Node 1 (Stage 0 + Ingress Gateway)
NEXT_IP="${NODE_IPS[2]}"
echo "Starting Stage 0 and Gateway on ${NODE_PREFIX}-1 (downstream -> ${NEXT_IP}:50051)..."
${GCLOUD} compute ssh "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="sudo chown -R \$USER:\$USER /opt/FrontierSplit && git config --global --add safe.directory /opt/FrontierSplit && git -C /opt/FrontierSplit pull origin main && pip install -r /opt/FrontierSplit/requirements.txt && systemctl --user reset-failed && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=0 --total-stages=${NUM_NODES} --port=50051 --downstream-url=http://${NEXT_IP}:50051 --model-name=${MODEL_ID} && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-gateway python3 -m frontiersplit.gateway --stage0-url=http://localhost:50051 --port=8000 --total-stages=${NUM_NODES} --model-name=${MODEL_ID}"

GATEWAY_IP=$(${GCLOUD} compute instances describe "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo "=== FrontierSplit Pipeline Online! ==="
echo "Ingress Gateway: http://${GATEWAY_IP}:8000/v1/chat/completions"
echo "Telemetry URL:   http://${GATEWAY_IP}:8000/v1/telemetry"
