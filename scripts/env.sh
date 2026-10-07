#!/usr/bin/env bash
# FrontierSplit: Cluster Configuration & Environment Variables

PROJECT_ID="frontiersplit-proto"
REGION="us-central1"
ZONE="us-central1-a"
NUM_NODES=4
MACHINE_TYPE="g2-standard-4"
ACCELERATOR="type=nvidia-l4,count=1"
DISK_SIZE="100GB"
DISK_TYPE="pd-balanced"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
NETWORK="default"
NODE_PREFIX="frontiersplit-node"
GCLOUD="/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
