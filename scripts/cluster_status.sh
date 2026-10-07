#!/usr/bin/env bash
# FrontierSplit: Check status of cluster nodes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# Execute unified cluster diagnostic suite
exec "${SCRIPT_DIR}/cluster_check.sh" "$@"
