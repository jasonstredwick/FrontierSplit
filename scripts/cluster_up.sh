#!/usr/bin/env bash
# FrontierSplit: Spin up or start the 4-node L4 cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=================================================================="
echo " Launching FrontierSplit Cluster: ${NUM_NODES}x ${MACHINE_TYPE} (L4 GPU)"
echo " Project: ${PROJECT_ID} | Zone: ${ZONE}"
echo "=================================================================="

# 1. Ensure internal firewall rule for inter-node communication
echo "[1/3] Ensuring internal VPC firewall rule exists..."
if ! ${GCLOUD} compute firewall-rules describe "frontiersplit-internal-mesh" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Creating firewall rule 'frontiersplit-internal-mesh'..."
    ${GCLOUD} compute firewall-rules create "frontiersplit-internal-mesh" \
        --project="${PROJECT_ID}" \
        --network="${NETWORK}" \
        --allow="tcp:50051-50060,tcp:8000,tcp:22" \
        --source-ranges="10.128.0.0/9,0.0.0.0/0" \
        --target-tags="frontiersplit-node" \
        --description="Allow internal tensor transport and API gateway access"
else
    echo "Firewall rule already active."
fi

# 2. Check existing instances
echo "[2/3] Checking cluster instances..."
EXISTING_INSTANCES=$(${GCLOUD} compute instances list \
  --project="${PROJECT_ID}" \
  --filter="name ~ '^${NODE_PREFIX}'" \
  --format="value(name)" || true)

if [ -n "${EXISTING_INSTANCES}" ]; then
    for NODE_NAME in ${EXISTING_INSTANCES}; do
        STATUS=$(${GCLOUD} compute instances describe "${NODE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --format="value(status)")
        if [ "${STATUS}" == "TERMINATED" ]; then
            echo "Starting stopped instance: ${NODE_NAME}..."
            ${GCLOUD} compute instances start "${NODE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" &
        else
            echo "Instance ${NODE_NAME} is already ${STATUS}."
        fi
    done
else
    for i in $(seq 1 ${NUM_NODES}); do
        NODE_NAME="${NODE_PREFIX}-${i}"
        echo "Creating new instance: ${NODE_NAME} with 1x NVIDIA L4..."
        ${GCLOUD} compute instances create "${NODE_NAME}" \
            --project="${PROJECT_ID}" \
            --zone="${ZONE}" \
            --machine-type="${MACHINE_TYPE}" \
            --accelerator="${ACCELERATOR}" \
            --boot-disk-size="${DISK_SIZE}" \
            --boot-disk-type="${DISK_TYPE}" \
            --image-family="${IMAGE_FAMILY}" \
            --image-project="${IMAGE_PROJECT}" \
            --maintenance-policy="TERMINATE" \
            --tags="frontiersplit-node" \
            --metadata-from-file="startup-script=${SCRIPT_DIR}/startup_node.sh" &
    done
fi

wait

# 3. Print final status
echo "[3/3] Cluster nodes ready. Status:"
"${SCRIPT_DIR}/cluster_status.sh"
