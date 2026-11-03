#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export COSMOS_DROID_CHECKPOINT="${COSMOS_DROID_CHECKPOINT:-/root/checkpoints/Cosmos3-Nano-Policy-DROID}"
export ROBOLAB_DEMOS="${ROBOLAB_DEMOS:-/root/RoboLab/output/cosmos3_droid_demos}"
export DROID_ROOT="${DROID_ROOT:-$ROBOLAB_DEMOS/train_ready/droid_plus_lerobot_640x360_20260412}"
export COSMOS_DROID_CONDITIONING="${COSMOS_DROID_CONDITIONING:-$ROBOLAB_DEMOS/train_ready/conditioning}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$ROBOLAB_DEMOS/training_assets/wan22_native.pth}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$ROBOLAB_DEMOS/training_runs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export COSMOS_RUN_STARTED_AT
COSMOS_RUN_STARTED_AT="$(date +%s)"
PYTHON="${COSMOS_PYTHON:-$PWD/.venv/bin/python}"
PER_DEVICE_BATCH="${COSMOS_DROID_PER_DEVICE_BATCH:-}"
if [[ -z "$PER_DEVICE_BATCH" ]]; then
    PER_DEVICE_BATCH="$("$PYTHON" - "$ROBOLAB_DEMOS/training_assets/batch128.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
print(json.loads(p.read_text())['per_device_batch_size'] if p.exists() else 64)
PY
)"
fi
if [[ ! "$PER_DEVICE_BATCH" =~ ^(1|2|4|8|16|32|64)$ ]]; then
    echo 'Per-device batch must be one of 64, 32, 16, 8, 4, 2, 1 for global batch 128.' >&2
    exit 2
fi
ACCUMULATION=$((64 / PER_DEVICE_BATCH))
EXTRA_ARGS=()
TAIL_OVERRIDES=("dataloader_train.max_samples_per_batch=$PER_DEVICE_BATCH" "trainer.grad_accum_iter=$ACCUMULATION")
IN_TAIL=false
for ARG in "$@"; do
    if [[ "$ARG" == "--" ]]; then
        IN_TAIL=true
    elif [[ "$IN_TAIL" == true ]]; then
        TAIL_OVERRIDES+=("$ARG")
    else
        EXTRA_ARGS+=("$ARG")
    fi
done
if [[ "${1:-}" == "--dryrun" ]]; then
    exec "$PYTHON" -m cosmos_framework.scripts.train --sft-toml examples/toml/sft_config/robolab_droid_nano_lora.toml "${EXTRA_ARGS[@]}" -- "${TAIL_OVERRIDES[@]}"
fi
if ! "$PYTHON" -c 'import transformer_engine.pytorch; from cosmos_framework.utils.generator.fused_adam import FusedAdam'; then
    echo 'Optimizer extension unavailable. Run: .venv/bin/python -m cosmos_framework.scripts.prepare_robolab_optimizer' >&2
    exit 2
fi
"$PYTHON" - "$ROBOLAB_DEMOS" "$COSMOS_DROID_CONDITIONING" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
provenance=json.loads((p/'train_ready/provenance.json').read_text())
assert len(provenance['counts']) == 8 and set(provenance['counts'].values()) == {25}, 'Need 25 successful demos for each of eight tasks'
assert provenance['validation_episodes'] == 0
cache=Path(sys.argv[2])/'manifest.json'
assert cache.exists() and json.loads(cache.read_text())['complete'], 'Precompute conditioning first'
PY
exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m cosmos_framework.scripts.train --sft-toml examples/toml/sft_config/robolab_droid_nano_lora.toml "${EXTRA_ARGS[@]}" -- "${TAIL_OVERRIDES[@]}"
