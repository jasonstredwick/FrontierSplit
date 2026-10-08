#!/usr/bin/env bash
# FrontierSplit: Cluster Configuration & Environment Variables
# Optimized for 8-Node Unquantized Mixtral 8x7B (FP16) Pipeline Parallelism

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Hardware configuration file
CLUSTER_CONFIG_FILE="${CLUSTER_CONFIG:-${ROOT_DIR}/cluster_config.json}"

if [ -f "${CLUSTER_CONFIG_FILE}" ] && command -v jq >/dev/null 2>&1; then
  PROJECT_ID="${PROJECT_ID:-$(jq -r '.project_id // "frontiersplit-proto"' "${CLUSTER_CONFIG_FILE}")}"
  REGION="${REGION:-$(jq -r '.region // "us-central1"' "${CLUSTER_CONFIG_FILE}")}"
  ZONE="${ZONE:-$(jq -r '.zone // "us-central1-a"' "${CLUSTER_CONFIG_FILE}")}"
  MODEL_ID="${MODEL_ID:-$(jq -r '.model_id // "mistralai/Mixtral-8x7B-Instruct-v0.1"' "${CLUSTER_CONFIG_FILE}")}"
  GATEWAY_HOST="${GATEWAY_HOST:-$(jq -r '.gateway.host // "127.0.0.1"' "${CLUSTER_CONFIG_FILE}")}"
  GATEWAY_PORT="${GATEWAY_PORT:-$(jq -r '.gateway.port // 8000' "${CLUSTER_CONFIG_FILE}")}"
  NUM_NODES="${NUM_NODES:-$(jq -r '.nodes | length // 8' "${CLUSTER_CONFIG_FILE}")}"
  MACHINE_TYPE="${MACHINE_TYPE:-$(jq -r '.machine_type // "n1-standard-8"' "${CLUSTER_CONFIG_FILE}")}"
  DISK_SIZE="${DISK_SIZE:-$(jq -r '.disk_size // "200GB"' "${CLUSTER_CONFIG_FILE}")}"
  SSH_KEY_RAW="$(jq -r '.ssh.key_path // ""' "${CLUSTER_CONFIG_FILE}")"
  SSH_KEY="${SSH_KEY:-${SSH_KEY_RAW/#\~/$HOME}}"
  SSH_USER="${SSH_USER:-$(jq -r '.ssh.user // "pixel"' "${CLUSTER_CONFIG_FILE}")}"
else
  PROJECT_ID="${PROJECT_ID:-frontiersplit-proto}"
  REGION="${REGION:-us-central1}"
  ZONE="${ZONE:-us-central1-a}"
  NUM_NODES="${NUM_NODES:-8}"
  MODEL_ID="${MODEL_ID:-mistralai/Mixtral-8x7B-Instruct-v0.1}"
  MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
  DISK_SIZE="${DISK_SIZE:-200GB}"
  GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
  GATEWAY_PORT="${GATEWAY_PORT:-8000}"
  SSH_KEY="${SSH_KEY:-$HOME/.ssh/google_compute_engine}"
  SSH_USER="${SSH_USER:-pixel}"
fi

ACCELERATOR="${ACCELERATOR:-type=nvidia-tesla-t4,count=1}"
DISK_TYPE="${DISK_TYPE:-pd-balanced}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
NETWORK="${NETWORK:-default}"
NODE_PREFIX="${NODE_PREFIX:-frontiersplit-node}"
GCLOUD="${GCLOUD:-/opt/homebrew/share/google-cloud-sdk/bin/gcloud}"
