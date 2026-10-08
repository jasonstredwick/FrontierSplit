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

echo "Setting up repository and Python dependencies on TPU VM..."
TPU_SSH "sudo mkdir -p /opt/FrontierSplit && sudo chown -R \$USER:\$USER /opt/FrontierSplit && \
  (git clone https://github.com/jasonstredwick/FrontierSplit.git /opt/FrontierSplit 2>/dev/null || (cd /opt/FrontierSplit && git pull origin main)) && \
  pip install -q fastapi uvicorn pydantic requests httpx transformers accelerate safetensors"

echo "Creating launcher script on TPU VM..."
TPU_SSH "cat <<'EOF' > /tmp/fs_tpu_launcher.sh
#!/usr/bin/env bash
set -e
export PYTHONPATH=/opt/FrontierSplit
pkill -f 'frontiersplit.worker' 2>/dev/null || true
pkill -f 'frontiersplit.gateway' 2>/dev/null || true
sleep 1

MODEL=\"${MODEL_ID}\"
USE_BIN=\"${USE_BINARY_TRANSPORT}\"
QUANT_FLAG=\"$([ "${QUANTIZE_ACTIVATIONS}" = "1" ] && echo "--quantize-activations")\"

# Stage 7 (Final Stage: LM Head, layers 28..31)
nohup python3 -m frontiersplit.worker \
  --stage-id=7 \
  --total-stages=8 \
  --port=50058 \
  \$([ \"\$USE_BIN\" = \"1\" ] && echo \"--tcp-port=50158\") \
  --device=xla:7 \
  \$QUANT_FLAG \
  --model-name=\$MODEL > /tmp/fs_stage7.log 2>&1 &

# Stages 6 down to 1
for s in 6 5 4 3 2 1; do
  PORT=\$((50051 + s))
  TCP_PORT=\$((50151 + s))
  NEXT_PORT=\$((PORT + 1))
  NEXT_TCP_PORT=\$((TCP_PORT + 1))

  nohup python3 -m frontiersplit.worker \
    --stage-id=\$s \
    --total-stages=8 \
    --port=\$PORT \
    \$([ \"\$USE_BIN\" = \"1\" ] && echo \"--tcp-port=\$TCP_PORT --downstream-tcp=127.0.0.1:\$NEXT_TCP_PORT\") \
    --downstream-url=http://127.0.0.1:\$NEXT_PORT \
    --device=xla:\$s \
    \$QUANT_FLAG \
    --model-name=\$MODEL > /tmp/fs_stage\${s}.log 2>&1 &
done

# Stage 0
nohup python3 -m frontiersplit.worker \
  --stage-id=0 \
  --total-stages=8 \
  --port=50051 \
  \$([ \"\$USE_BIN\" = \"1\" ] && echo \"--tcp-port=50151 --downstream-tcp=127.0.0.1:50152\") \
  --downstream-url=http://127.0.0.1:50052 \
  --device=xla:0 \
  \$QUANT_FLAG \
  --model-name=\$MODEL > /tmp/fs_stage0.log 2>&1 &

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
