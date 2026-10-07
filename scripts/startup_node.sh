#!/usr/bin/env bash
# FrontierSplit: Node Startup Script (Runs on VM boot)
set -euo pipefail

echo "=== [FrontierSplit] Node Initialization Starting ==="

# 1. Verify NVIDIA Driver & GPU presence
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU Detected:"
    nvidia-smi
else
    echo "WARNING: nvidia-smi not yet found in PATH"
fi

# 2. Clone or update repository
REPO_DIR="/opt/FrontierSplit"
if [ ! -d "${REPO_DIR}" ]; then
    echo "Cloning FrontierSplit repository..."
    git clone https://github.com/jasonstredwick/FrontierSplit.git "${REPO_DIR}"
else
    echo "Updating FrontierSplit repository..."
    cd "${REPO_DIR}" && git pull origin main || true
fi

# 3. Install requirements into system / DLVM python
if [ -f "${REPO_DIR}/requirements.txt" ]; then
    echo "Installing FrontierSplit dependencies..."
    pip install -r "${REPO_DIR}/requirements.txt" || true
fi

echo "=== [FrontierSplit] Node Ready for Pipeline Service ==="
