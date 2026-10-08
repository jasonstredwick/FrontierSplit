#!/usr/bin/env bash
# FrontierSplit: Spin up, scale, or resume the pipeline cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=================================================================="
echo " Launching FrontierSplit Cluster: ${NUM_NODES}x ${MACHINE_TYPE} (${ACCELERATOR})"
echo " Model: ${MODEL_ID} (Unquantized FP16)"
echo " Project: ${PROJECT_ID} | Zone: ${ZONE}"
echo "=================================================================="

# 1. Ensure internal VPC firewall rule exists
echo "[1/4] Ensuring internal VPC firewall rule exists..."
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

# 2. Check and provision all N nodes
echo "[2/4] Verifying and provisioning ${NUM_NODES} cluster instances..."
EXISTING_INSTANCES=$(${GCLOUD} compute instances list \
  --project="${PROJECT_ID}" \
  --filter="name ~ '^${NODE_PREFIX}'" \
  --format="value(name)" || true)

for i in $(seq 1 ${NUM_NODES}); do
    NODE_NAME="${NODE_PREFIX}-${i}"
    
    if echo "${EXISTING_INSTANCES}" | grep -qw "${NODE_NAME}"; then
        STATUS=$(${GCLOUD} compute instances describe "${NODE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --format="value(status)")
        echo "Found existing instance ${NODE_NAME} (Status: ${STATUS})..."
        
        if [ "${STATUS}" == "TERMINATED" ]; then
            # Verify and update machine type if upgraded (e.g. n1-standard-8)
            CURR_TYPE=$(${GCLOUD} compute instances describe "${NODE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --format="value(machineType.basename())")
            if [ "${CURR_TYPE}" != "${MACHINE_TYPE}" ]; then
                echo "Upgrading ${NODE_NAME} machine type from ${CURR_TYPE} to ${MACHINE_TYPE}..."
                ${GCLOUD} compute instances set-machine-type "${NODE_NAME}" \
                    --machine-type="${MACHINE_TYPE}" \
                    --project="${PROJECT_ID}" \
                    --zone="${ZONE}" || true
            fi
            
            # Verify and resize disk if needed
            ${GCLOUD} compute disks resize "${NODE_NAME}" \
                --size="${DISK_SIZE}" \
                --project="${PROJECT_ID}" \
                --zone="${ZONE}" --quiet 2>/dev/null || true
            
            echo "Starting stopped instance: ${NODE_NAME}..."
            ${GCLOUD} compute instances start "${NODE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" &
        else
            echo "Instance ${NODE_NAME} is already ${STATUS}."
        fi
    else
        echo "Creating new instance: ${NODE_NAME} with ${MACHINE_TYPE} and ${ACCELERATOR}..."
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
    fi
done

wait
echo "All ${NUM_NODES} instances are up and running."

# 3. Synchronize cluster_config.json with live IPs
echo "[3/4] Discovering live network IPs and updating cluster_config.json..."
python3 -c "
import json
import subprocess

config_path = '${CLUSTER_CONFIG_FILE}'
num_nodes = ${NUM_NODES}
prefix = '${NODE_PREFIX}'
project = '${PROJECT_ID}'
zone = '${ZONE}'
gcloud = '${GCLOUD}'

try:
    with open(config_path, 'r') as f:
        cfg = json.load(f)
except Exception:
    cfg = {}

res = subprocess.check_output([
    gcloud, 'compute', 'instances', 'list',
    f'--project={project}', f'--filter=name ~ ^{prefix}', '--format=json'
], text=True)
instances = json.loads(res)
inst_map = {}
for inst in instances:
    name = inst.get('name', '')
    net = inst.get('networkInterfaces', [{}])[0]
    int_ip = net.get('networkIP', '')
    ext_ip = net.get('accessConfigs', [{}])[0].get('natIP', '')
    inst_map[name] = {'internal': int_ip, 'external': ext_ip}

cfg['cluster_name'] = f'frontiersplit-{num_nodes}x-t4'
cfg['model_id'] = '${MODEL_ID}'
cfg['machine_type'] = '${MACHINE_TYPE}'
cfg['disk_size'] = '${DISK_SIZE}'
cfg['precision'] = 'fp16'

total_layers = 32
layers_per_node = total_layers // num_nodes
nodes_list = []

for i in range(1, num_nodes + 1):
    name = f'{prefix}-{i}'
    info = inst_map.get(name, {'internal': '', 'external': ''})
    ext_ip = info['external']
    int_ip = info['internal']

    start_l = (i - 1) * layers_per_node
    end_l = (i * layers_per_node) - 1
    roles = []
    if i == 1:
        roles.append('gateway')
        roles.append('stage_0')
        cfg.setdefault('gateway', {})['host'] = ext_ip
    else:
        roles.append(f'stage_{i-1}')

    nodes_list.append({
        'node_id': i,
        'name': name,
        'stage_id': i - 1,
        'external_ip': ext_ip,
        'internal_ip': int_ip,
        'worker_port': 50051,
        'roles': roles,
        'layer_range': [start_l, end_l]
    })

cfg['nodes'] = nodes_list

with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=2)

print(f'Successfully synchronized {num_nodes} nodes into {config_path}')
" || true

# 4. Status summary
echo "[4/4] Cluster provisioning complete. Current status:"
"${SCRIPT_DIR}/cluster_status.sh"
