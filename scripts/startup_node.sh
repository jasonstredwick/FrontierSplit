#!/usr/bin/env bash
# FrontierSplit: Node Startup Script (Runs on VM boot)
set -euo pipefail

echo "=== [FrontierSplit] Starting Node Provisioning ==="

# 1. Update OS packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y build-essential git curl wget python3 python3-pip python3-venv

# 2. Install NVIDIA Drivers if not present
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA GPU Drivers..."
    curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output /tmp/install_gpu_driver.py
    python3 /tmp/install_gpu_driver.py
fi

echo "=== [FrontierSplit] Node Initialization Complete ==="
