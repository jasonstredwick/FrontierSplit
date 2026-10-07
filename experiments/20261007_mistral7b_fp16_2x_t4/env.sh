#!/usr/bin/env bash
# FrontierSplit: Cluster Configuration Snapshot for Experiment 20261007_mistral7b_fp16_2x_t4

PROJECT_ID="frontiersplit-proto"
REGION="us-central1"
ZONE="us-central1-a"
NUM_NODES=2
MACHINE_TYPE="n1-standard-4"
ACCELERATOR="type=nvidia-tesla-t4,count=1"
DISK_SIZE="100GB"
DISK_TYPE="pd-balanced"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
NETWORK="default"
NODE_PREFIX="frontiersplit-node"
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
GCLOUD="/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
NODE_1_IP="${NODE_1_IP:-136.80.22.76}"
NODE_2_IP="${NODE_2_IP:-34.30.176.33}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/google_compute_engine}"
SSH_USER="${SSH_USER:-pixel}"
