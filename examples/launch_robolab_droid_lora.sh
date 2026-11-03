#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${COSMOS_DROID_MODEL:-edge}" in
    edge) exec bash "$SCRIPT_DIR/launch_robolab_droid_edge_lora.sh" "$@" ;;
    nano) exec bash "$SCRIPT_DIR/launch_robolab_droid_nano_lora.sh" "$@" ;;
    *) echo 'COSMOS_DROID_MODEL must be edge or nano.' >&2; exit 2 ;;
esac
