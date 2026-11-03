#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export COSMOS_DROID_CHECKPOINT="${COSMOS_DROID_CHECKPOINT:-/root/checkpoints/Cosmos3-Nano-Policy-DROID}"
export ROBOLAB_DEMOS="${ROBOLAB_DEMOS:-/root/RoboLab/output/cosmos3_droid_demos}"
export DROID_ROOT="${DROID_ROOT:-$ROBOLAB_DEMOS/train_ready/droid_plus_lerobot_640x360_20260412}"
export COSMOS_DROID_CONDITIONING="${COSMOS_DROID_CONDITIONING:-$ROBOLAB_DEMOS/train_ready/conditioning}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$ROBOLAB_DEMOS/training_assets/wan22_native.pth}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON="${COSMOS_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    COSMOS_DEVICE=cpu "$PYTHON" -m cosmos_framework.scripts.prepare_robolab_vae \
        --checkpoint "$COSMOS_DROID_CHECKPOINT" --output "$WAN_VAE_PATH"
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
    -m cosmos_framework.scripts.prepare_robolab_conditioning --sft-toml examples/toml/sft_config/robolab_droid_nano_lora.toml "$@"
