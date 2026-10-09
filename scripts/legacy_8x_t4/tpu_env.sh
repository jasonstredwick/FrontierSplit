#!/usr/bin/env bash
# FrontierSplit: Cloud TPU v5e Environment Configuration
# Designed for symmetrical comparison against 8x NVIDIA T4 cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-frontiersplit-proto}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"

QUEUED_RESOURCE_ID="${QUEUED_RESOURCE_ID:-qr-probe}"
TPU_NAME="${TPU_NAME:-frontiersplit-tpu}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-8}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"
NETWORK="${NETWORK:-default}"
TAGS="${TAGS:-frontiersplit-node}"

MODEL_ID="${MODEL_ID:-mistralai/Mixtral-8x7B-Instruct-v0.1}"
NUM_STAGES="${NUM_STAGES:-8}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"

GCLOUD="${GCLOUD:-/opt/homebrew/share/google-cloud-sdk/bin/gcloud}"
if [ ! -x "${GCLOUD}" ]; then
  GCLOUD="$(which gcloud || echo "gcloud")"
fi
