#!/usr/bin/env bash
# FrontierSplit: Start 8-stage pipeline across TPU v5e cores on Cloud TPU VM
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tpu_env.sh"

echo "=================================================================="
echo " Starting FrontierSplit Pipeline on Cloud TPU VM: ${TPU_NAME}"
echo " Accelerator: ${ACCELERATOR_TYPE} (8 TPU Cores: xla:0..xla:7)"
echo " Model:       ${MODEL_ID}"
echo " Zone:        ${ZONE} | Project: ${PROJECT_ID}"
echo "=================================================================="

# Check if TPU VM exists and is in READY state
NODE_STATE=$(${GCLOUD} compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(state)" 2>/dev/null || echo "NOT_READY")
if [ "${NODE_STATE}" != "READY" ]; then
  echo "Error: TPU VM '${TPU_NAME}' is not in READY state (current: ${NODE_STATE})."
  echo "Please ensure the Queued Resource is ACTIVE first via ./scripts/tpu_status.sh."
  exit 1
fi

USE_BINARY_TRANSPORT="${USE_BINARY_TRANSPORT:-1}"
ENABLE_1F1B="${ENABLE_1F1B:-1}"
QUANTIZE_ACTIVATIONS="${QUANTIZE_ACTIVATIONS:-0}"
echo "Binary TCP Transport:    $([ "${USE_BINARY_TRANSPORT}" = "1" ] && echo "ENABLED" || echo "DISABLED (HTTP Baseline)")"
echo "Asynchronous 1F1B:       $([ "${ENABLE_1F1B}" = "1" ] && echo "ENABLED" || echo "DISABLED")"
echo "Activation Quantization: $([ "${QUANTIZE_ACTIVATIONS}" = "1" ] && echo "ENABLED (INT8)" || echo "DISABLED (FP16)")"

TPU_SSH() {
  ${GCLOUD} compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --command="$1"
}

echo "Updating repository on TPU VM..."
TPU_SSH "sudo mkdir -p /opt/FrontierSplit && sudo chown -R \$USER:\$USER /opt/FrontierSplit && \
  (git clone https://github.com/jasonstredwick/FrontierSplit.git /opt/FrontierSplit 2>/dev/null || (cd /opt/FrontierSplit && git pull origin main))"

echo "Creating launcher script on TPU VM..."
TPU_SSH "cat <<'EOF' > /tmp/fs_tpu_launcher.sh
#!/usr/bin/env bash
set -e
export PATH=/home/pixel/.local/bin:\$PATH
export PYTHONPATH=/opt/FrontierSplit
export HF_HOME=/dev/shm/huggingface
sudo fuser -k 50051/tcp 50052/tcp 50053/tcp 50054/tcp 50055/tcp 50056/tcp 50057/tcp 50058/tcp 50151/tcp 50152/tcp 50153/tcp 50154/tcp 50155/tcp 50156/tcp 50157/tcp 50158/tcp 8000/tcp 2>/dev/null || true
pkill -9 -f 'frontiersplit' 2>/dev/null || true
pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
for poll_i in \$(seq 1 15); do
  if ! sudo ss -tlpn | grep -qE '5005[1-8]|5015[1-8]|:8000 '; then
    break
  fi
  sleep 1
done

MODEL=\"${MODEL_ID}\"
USE_BIN=\"${USE_BINARY_TRANSPORT}\"
QUANT_FLAG=\"$([ "${QUANTIZE_ACTIVATIONS}" = "1" ] && echo "--quantize-activations")\"
BIN_FLAG=\"$([ \"\$USE_BIN\" = \"0\" ] && echo \"--disable-binary\")\"

# Launch all 8 TPU worker stages cleanly across the 8 TPU cores
echo \"Starting 8-stage TPU worker processes via xmp.spawn...\"
nohup python3 -m frontiersplit.tpu_runner \
  --total-stages=8 \
  --model-name=\$MODEL \
  \$BIN_FLAG \
  \$QUANT_FLAG > /tmp/fs_tpu_workers.log 2>&1 &

# Wait for all 8 stages to become healthy
echo \"Waiting for all 8 TPU worker stages to become healthy...\"
all_ok=0
for k in \$(seq 1 80); do
  all_ok=1
  for p in \$(seq 50051 50058); do
    if ! curl -s http://127.0.0.1:\$p/health 2>/dev/null | grep -q '\"status\":\"healthy\"'; then
      all_ok=0
      break
    fi
  done
  if [ \$all_ok -eq 1 ]; then
    echo \"All 8 TPU stages are healthy! Starting Ingress Gateway...\"
    break
  fi
  sleep 3
done

if [ \$all_ok -ne 1 ]; then
  echo \"Error: TPU stages failed to reach healthy state.\"
  tail -n 40 /tmp/fs_tpu_workers.log
  exit 1
fi

# Ingress Gateway
ENABLE_1F1B_FLAG=\"$([ "${ENABLE_1F1B}" = "0" ] && echo "--disable-1f1b")\"
nohup python3 -m frontiersplit.gateway \
  --stage0-url=http://127.0.0.1:50051 \
  \$([ \"\$USE_BIN\" = \"1\" ] && echo \"--stage0-tcp=127.0.0.1:50151\") \
  \$ENABLE_1F1B_FLAG \
  --port=${GATEWAY_PORT} \
  --total-stages=8 \
  --model-name=\$MODEL > /tmp/fs_gateway.log 2>&1 &
EOF
chmod +x /tmp/fs_tpu_launcher.sh && /tmp/fs_tpu_launcher.sh"

echo "Waiting for TPU pipeline to initialize..."
for i in $(seq 1 60); do
  if TPU_SSH "curl -s http://127.0.0.1:${GATEWAY_PORT}/health | grep -q '\"gateway\":\"healthy\"'" 2>/dev/null; then
    echo "=== FrontierSplit TPU Pipeline is ONLINE and HEALTHY! ==="
    "${SCRIPT_DIR}/tpu_status.sh"
    exit 0
  fi
  echo "Waiting for workers to load weights into TPU HBM (attempt ${i}/60, sleeping 3s)..."
  sleep 3
done

echo "Error: Pipeline failed to reach healthy status. Checking logs:"
TPU_SSH "cat /tmp/fs_gateway.log || true"
exit 1
