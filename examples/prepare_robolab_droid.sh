#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${COSMOS_DROID_MODEL:-edge}" in
    edge) exec bash "$SCRIPT_DIR/prepare_robolab_droid_edge.sh" "$@" ;;
    nano) exec bash "$SCRIPT_DIR/prepare_robolab_droid_nano.sh" "$@" ;;
    *) echo 'COSMOS_DROID_MODEL must be edge or nano.' >&2; exit 2 ;;
esac
