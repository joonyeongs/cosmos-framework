# RoboLab DROID training — Edge by default

Use **nvidia/Cosmos3-Edge-Policy-DROID** for the 200 successful RoboLab demos.
The generic launch/preparation commands and generic TOML now select Edge.
The Nano code, recipe and conditioning remain available as explicit alternatives.

| Setting | Edge default |
|---|---|
| Shared adapter | Generator video/action LoRA, rank 16, alpha 32 |
| Frozen components | Base weights; reasoning conditioning precomputed, reasoning branch absent during training |
| Data | 25 successful demos per task, 8 tasks, all 200 used for training |
| Batch | 32 samples/GPU × 2 GPUs × accumulation 2 = global batch 128 |
| Parallelism | Original two-way FSDP sharding |
| Schedule | 4,000 optimizer steps, LR 0.0001 linearly decaying to zero |
| Images | Random crop retained, color augmentation disabled |
| Input speedups | Compiled VAE encoder norms; 4 workers/GPU, prefetch factor 8 |
| Measured runtime | 132.7 seconds/step; about 147.4 hours (6.1 days) for 4,000 steps on two A6000 48 GB GPUs |

The time estimate comes from short executed benchmarks, not a full fine-tuning
run. See the [measurements](robolab_droid_edge_optimization.md).

## Download and launch on another server

Clone this repository's `feature/robolab-cosmos3-droid-lora-demos` branch.
Install its training environment using [setup.md](setup.md); the measured stack
used Python 3.13 and PyTorch 2.13/cu130. Keep this environment separate from OpenPI.

From the repository root, after authenticating with Hugging Face:

```bash
export TRANSFER_ROOT="$HOME/robolab-transfer"
export ROBOLAB_DEMOS="$TRANSFER_ROOT/data/cosmos3_droid_demos"
export COSMOS_DROID_MODEL=edge
export COSMOS_DROID_CHECKPOINT="$TRANSFER_ROOT/checkpoints/Cosmos3-Edge-Policy-DROID"
export COSMOS_DROID_CONDITIONING="$ROBOLAB_DEMOS/train_ready/conditioning_edge"
export WAN_VAE_PATH="$ROBOLAB_DEMOS/training_assets/wan22_native.pth"
export COSMOS_DROID_PER_DEVICE_BATCH=32
hf download joonyeongs/robolab-droid-200 --repo-type dataset \
  --local-dir "$TRANSFER_ROOT/data"
hf download nvidia/Cosmos3-Edge-Policy-DROID \
  --revision 0110e64ca982589f1a3d856cbdd7d465dccb79bc \
  --local-dir "$COSMOS_DROID_CHECKPOINT" \
  --include '*.json' '*.jinja' 'transformer/*' 'text_tokenizer/*' 'vae/*'
# The matching Edge conditioning cache is included in the dataset.
# Convert the bundled VAE once if its native file is not already present:
if [[ ! -f "$WAN_VAE_PATH" ]]; then
  COSMOS_DEVICE=cpu .venv/bin/python -m cosmos_framework.scripts.prepare_robolab_vae \
    --checkpoint "$COSMOS_DROID_CHECKPOINT" --output "$WAN_VAE_PATH"
fi
.venv/bin/python -m cosmos_framework.scripts.prepare_robolab_optimizer \
  --build-root "$ROBOLAB_DEMOS/training_assets/dependencies"
bash examples/launch_robolab_droid_lora.sh --dryrun
# Start the full run only when ready:
bash examples/launch_robolab_droid_lora.sh
```

The generic launcher dispatches to `launch_robolab_droid_edge_lora.sh` by default.
Its explicit Edge counterpart remains available. The generic TOML
`examples/toml/sft_config/robolab_droid_lora.toml` also selects Edge. Runtime paths
are set by the environment variables above; historical reports contain source
machine paths. Dataset selection metadata is in `training_defaults.json`.

To regenerate Edge conditioning, use `bash examples/prepare_robolab_droid.sh`.
This requires loading the reasoning branch once; ordinary cached training does
not. See the [Edge guide](robolab_droid_edge_lora.md) for encoder parity checks,
short training validation and checkpoint details. The memory-sweep validator
now defaults to Edge; `--model nano` explicitly selects Nano. It deliberately
probes 64/device first and is optional for the already selected 32 × 2 setup.

## Nano remains available

Use the [Nano guide](robolab_droid_nano_lora.md) and the explicit commands:

```bash
export COSMOS_DROID_MODEL=nano
export COSMOS_DROID_CHECKPOINT="$TRANSFER_ROOT/checkpoints/Cosmos3-Nano-Policy-DROID"
export COSMOS_DROID_CONDITIONING="$ROBOLAB_DEMOS/train_ready/conditioning"
export COSMOS_DROID_PER_DEVICE_BATCH=16
# If needed, regenerate Nano's cache using the matching Nano checkpoint:
bash examples/prepare_robolab_droid_nano.sh
# Or use the already included Nano conditioning cache:
bash examples/launch_robolab_droid_nano_lora.sh --dryrun
```

The Nano TOML is `examples/toml/sft_config/robolab_droid_nano_lora.toml`.
`COSMOS_DROID_MODEL=nano` also routes the generic shell commands to Nano.
The original internal Nano experiment name, `robolab_droid_lora`, is retained for
compatibility; the public generic TOML now selects `robolab_droid_edge_lora`.
Switch checkpoint, conditioning path and batch together when changing models.
OpenPI's separate dataset and π0.5 training configuration are unchanged.
