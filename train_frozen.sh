#!/usr/bin/env bash
# Experiment 2: Frozen Backbone — train only projection heads
# Usage:
#   ./train_frozen.sh              # run both siglip & reglip frozen
#   ./train_frozen.sh siglip       # run siglip frozen only
#   ./train_frozen.sh reglip       # run reglip frozen only
#   ./train_frozen.sh --debug      # both, debug mode

set -euo pipefail

DEBUG=""
TARGET="both"

for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG="--debug" ;;
        siglip)  TARGET="siglip" ;;
        reglip)  TARGET="reglip" ;;
    esac
done

if [ "$TARGET" = "siglip" ] || [ "$TARGET" = "both" ]; then
    echo "=== Frozen SigLIP ==="
    python training/train_siglip.py --config configs/siglip_frozen_config.yaml $DEBUG
fi

if [ "$TARGET" = "reglip" ] || [ "$TARGET" = "both" ]; then
    echo "=== Frozen RegLIP ==="
    python training/train_reglip.py --config configs/reglip_frozen_config.yaml $DEBUG
fi

echo "=== Done ==="
