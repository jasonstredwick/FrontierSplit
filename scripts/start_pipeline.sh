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

USE_BINARY_TRANSPORT="${USE_BINARY_TRANSPORT:-1}"
TCP_PORT="${TCP_PORT:-50052}"
echo "Binary TCP Transport: $([ "${USE_BINARY_TRANSPORT}" = "1" ] && echo "ENABLED (Port ${TCP_PORT})" || echo "DISABLED")"

# Start workers from final stage down to stage 1
FINAL_NODE=${NUM_NODES}
FINAL_STAGE=$((NUM_NODES - 1))

TCP_FLAG_FINAL=""
if [ "${USE_BINARY_TRANSPORT}" = "1" ]; then
  TCP_FLAG_FINAL="--tcp-port=${TCP_PORT}"
fi

echo "Ensuring Final Stage (${FINAL_STAGE}) is running on ${NODE_PREFIX}-${FINAL_NODE}..."
${GCLOUD} compute ssh "${NODE_PREFIX}-${FINAL_NODE}" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="sudo loginctl enable-linger \$USER && sudo chown -R \$USER:\$USER /opt/FrontierSplit && git config --global --add safe.directory /opt/FrontierSplit && git -C /opt/FrontierSplit pull origin main && pip install -r /opt/FrontierSplit/requirements.txt && (curl -s http://localhost:50051/health | grep -q '\"status\":\"healthy\"' && echo 'Stage ${FINAL_STAGE} is already healthy!') || (systemctl --user stop fs-worker 2>/dev/null || true && systemctl --user reset-failed 2>/dev/null || true && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=${FINAL_STAGE} --total-stages=${NUM_NODES} --port=50051 ${TCP_FLAG_FINAL} --model-name=${MODEL_ID})"

echo "Waiting for Stage ${FINAL_STAGE} on ${NODE_PREFIX}-${FINAL_NODE} to become healthy..."
${GCLOUD} compute ssh "${NODE_PREFIX}-${FINAL_NODE}" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="for i in \$(seq 1 300); do curl -s http://localhost:50051/health | grep -q '\"status\":\"healthy\"' && echo 'Stage healthy!' && exit 0; echo 'Waiting for worker / weights download...'; sleep 3; done; echo 'Timeout waiting for worker'; journalctl --user-unit=fs-worker -n 50 --no-pager; exit 1"

# Intermediate nodes: from FINAL_NODE-1 down to 2
for i in $(seq $((NUM_NODES - 1)) -1 2); do
  STAGE_ID=$((i - 1))
  NEXT_IP="${NODE_IPS[$((i + 1))]}"
  TCP_FLAGS_INTERMEDIATE=""
  if [ "${USE_BINARY_TRANSPORT}" = "1" ]; then
    TCP_FLAGS_INTERMEDIATE="--tcp-port=${TCP_PORT} --downstream-tcp=${NEXT_IP}:${TCP_PORT}"
  fi
  echo "Ensuring Stage ${STAGE_ID} is running on ${NODE_PREFIX}-${i} (downstream -> ${NEXT_IP}:50051)..."
  ${GCLOUD} compute ssh "${NODE_PREFIX}-${i}" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
    --command="sudo loginctl enable-linger \$USER && sudo chown -R \$USER:\$USER /opt/FrontierSplit && git config --global --add safe.directory /opt/FrontierSplit && git -C /opt/FrontierSplit pull origin main && pip install -r /opt/FrontierSplit/requirements.txt && (curl -s http://localhost:50051/health | grep -q '\"status\":\"healthy\"' && echo 'Stage ${STAGE_ID} is already healthy!') || (systemctl --user stop fs-worker 2>/dev/null || true && systemctl --user reset-failed 2>/dev/null || true && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=${STAGE_ID} --total-stages=${NUM_NODES} --port=50051 --downstream-url=http://${NEXT_IP}:50051 ${TCP_FLAGS_INTERMEDIATE} --model-name=${MODEL_ID})"

  echo "Waiting for Stage ${STAGE_ID} on ${NODE_PREFIX}-${i} to become healthy..."
  ${GCLOUD} compute ssh "${NODE_PREFIX}-${i}" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
    --command="for j in \$(seq 1 300); do curl -s http://localhost:50051/health | grep -q '\"status\":\"healthy\"' && echo 'Stage ${STAGE_ID} is healthy!' && exit 0; echo 'Waiting for worker / weights download...'; sleep 3; done; echo 'Timeout waiting for worker'; journalctl --user-unit=fs-worker -n 50 --no-pager; exit 1"
done

# Node 1 (Stage 0 + Ingress Gateway)
NEXT_IP="${NODE_IPS[2]}"
TCP_FLAGS_0=""
if [ "${USE_BINARY_TRANSPORT}" = "1" ]; then
  TCP_FLAGS_0="--tcp-port=${TCP_PORT} --downstream-tcp=${NEXT_IP}:${TCP_PORT}"
fi
echo "Ensuring Stage 0 is running on ${NODE_PREFIX}-1 (downstream -> ${NEXT_IP}:50051)..."
${GCLOUD} compute ssh "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="sudo loginctl enable-linger \$USER && sudo chown -R \$USER:\$USER /opt/FrontierSplit && git config --global --add safe.directory /opt/FrontierSplit && git -C /opt/FrontierSplit pull origin main && pip install -r /opt/FrontierSplit/requirements.txt && (curl -s http://localhost:50051/health | grep -q '\"status\":\"healthy\"' && echo 'Stage 0 is already healthy!') || (systemctl --user stop fs-worker 2>/dev/null || true && systemctl --user reset-failed 2>/dev/null || true && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-worker python3 -m frontiersplit.worker --stage-id=0 --total-stages=${NUM_NODES} --port=50051 --downstream-url=http://${NEXT_IP}:50051 ${TCP_FLAGS_0} --model-name=${MODEL_ID})"

echo "Waiting for Stage 0 on ${NODE_PREFIX}-1 to become healthy..."
${GCLOUD} compute ssh "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="for i in \$(seq 1 300); do curl -s http://localhost:50051/health | grep -q '\"status\":\"healthy\"' && echo 'Stage 0 is healthy!' && exit 0; echo 'Waiting for worker / weights download...'; sleep 3; done; echo 'Timeout waiting for worker'; journalctl --user-unit=fs-worker -n 50 --no-pager; exit 1"

GATEWAY_TCP_FLAG=""
if [ "${USE_BINARY_TRANSPORT}" = "1" ]; then
  GATEWAY_TCP_FLAG="--stage0-tcp=localhost:${TCP_PORT}"
fi
echo "Ensuring Ingress Gateway is running on ${NODE_PREFIX}-1..."
${GCLOUD} compute ssh "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="(curl -s http://localhost:8000/health | grep -q '\"gateway\":\"healthy\"' && echo 'Gateway is already healthy!') || (systemctl --user stop fs-gateway 2>/dev/null || true && systemctl --user reset-failed 2>/dev/null || true && systemd-run --user --working-directory=/opt/FrontierSplit --setenv=PYTHONPATH=/opt/FrontierSplit --unit=fs-gateway python3 -m frontiersplit.gateway --stage0-url=http://localhost:50051 ${GATEWAY_TCP_FLAG} --port=8000 --total-stages=${NUM_NODES} --model-name=${MODEL_ID})"

echo "Waiting for Ingress Gateway on ${NODE_PREFIX}-1 to become healthy..."
${GCLOUD} compute ssh "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --tunnel-through-iap \
  --command="for i in \$(seq 1 60); do curl -s http://localhost:8000/health | grep -q '\"gateway\":\"healthy\"' && echo 'Gateway is healthy!' && exit 0; echo 'Waiting for gateway...'; sleep 2; done; echo 'Timeout waiting for gateway'; journalctl --user-unit=fs-gateway -n 30 --no-pager; exit 1"

GATEWAY_IP=$(${GCLOUD} compute instances describe "${NODE_PREFIX}-1" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo "=== FrontierSplit Pipeline Online & Healthy! ==="
echo "Ingress Gateway: http://${GATEWAY_IP}:8000/v1/chat/completions"
echo "Health Status:   http://${GATEWAY_IP}:8000/health"
echo "Telemetry URL:   http://${GATEWAY_IP}:8000/v1/telemetry"
