---
pretty_name: RoboLab DROID 200 successful demonstrations
tags:
- robotics
- droid
- robolab
- lerobot
---
# RoboLab DROID demonstrations and training transfer

Private dataset for `joonyeongs`: **200 successful episodes, 25 per task**, 157,925 synchronized frames at 15 Hz. Tasks: ToyInBinTask, CannedFoodInBinTask, ToolsPickingDrillTask, RedItemsInBinTask, RecycleCartonsVerticalCrateTask, PlasticBottlesInSquarePailTask, BlockStackingSpecifiedOrderTask, SpoonsInPotTask.

## Contents

- `cosmos3_droid_demos/original`: native HDF5, three 640×360 RGB videos per episode, camera calibration, robot/object states, all eight physics-substep motor targets and provenance.
- `cosmos3_droid_demos/train_ready/droid_plus_lerobot_640x360_20260412/success`: LeRobot v3 for Cosmos.
- `cosmos3_droid_demos/train_ready/conditioning`: precomputed frozen Cosmos text conditioning and parity fixture. Generated with the matching code and checkpoint; do not substitute a different model.
- `pi05_droid_lora/lerobot/robolab/droid_200`: OpenPI's pinned LeRobot v2.1 format.
- Reports, data audits and measured batch selections for both trainers.

All 200 episodes are training data; there is no held-out split. Only target poses vary within validated feasible reset distributions. Cosmos uses the last absolute joint target per interval. OpenPI uses reconstructed normalized joint velocity `(q_post - q_pre)/0.2`, plus the recorded gripper target. These are not original velocity commands. Nineteen frames exceed nominal velocity bounds and are retained without clipping. Training feasibility was tested; fine-tuned policy success was not evaluated.

Base model checkpoints, environment installations, failed runs, and simulator assets are not included. Obtain model weights from their official sources below. Historical JSON provenance contains source-machine paths; runtime paths are set by the environment variables below.

## Clone and download on another server

Authenticate with GitHub and Hugging Face as an account with access (for example `gh auth login` and `hf auth login`). Never place tokens in clone URLs or committed files. Install Git LFS and uv as prerequisites.

```bash
export TRANSFER_ROOT="$HOME/robolab-transfer"
mkdir -p "$TRANSFER_ROOT"
cd "$TRANSFER_ROOT"
GIT_LFS_SKIP_SMUDGE=1 git clone -b feature/robolab-cosmos3-droid-lora-demos https://github.com/joonyeongs/RoboLab.git
GIT_LFS_SKIP_SMUDGE=1 git clone -b feature/robolab-cosmos3-droid-lora-demos https://github.com/joonyeongs/cosmos-framework.git
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules -b feature/robolab-pi05-droid-lora https://github.com/joonyeongs/openpi.git
hf download joonyeongs/robolab-droid-200 --repo-type dataset --local-dir "$TRANSFER_ROOT/data"
```

Git LFS objects for upstream sample/simulator assets remain at their original upstream repositories. Training on these downloaded demonstrations does not require those assets. To retrieve them for other uses:

```bash
git -C "$TRANSFER_ROOT/RoboLab" remote add upstream https://github.com/NVlabs/RoboLab.git
git -C "$TRANSFER_ROOT/RoboLab" lfs fetch upstream HEAD
git -C "$TRANSFER_ROOT/RoboLab" lfs checkout
git -C "$TRANSFER_ROOT/cosmos-framework" remote add upstream https://github.com/NVIDIA/cosmos-framework.git
git -C "$TRANSFER_ROOT/cosmos-framework" lfs fetch upstream HEAD
git -C "$TRANSFER_ROOT/cosmos-framework" lfs checkout
```

## OpenPI training

```bash
cd "$TRANSFER_ROOT/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
export ROBOLAB_PI05_ROOT="$TRANSFER_ROOT/data/pi05_droid_lora"
export PI05_DROID_CHECKPOINT="$TRANSFER_ROOT/checkpoints/pi05_droid"
.venv/bin/python scripts/robolab/download_checkpoint.py
bash scripts/robolab/launch.sh 1 --dry-run
# Start training only when ready (select 1, 2, or 4):
bash scripts/robolab/launch.sh 1
```

The variants scale language rank/alpha from 16/16 and action expert rank/alpha from 32/32 by 1, 2, or 4. Upstream freeze policy also trains vision and action/time projections. All three were tested at 64/device × two A6000 GPUs × accumulation 1 = global batch 128. Default schedule: 20,000 optimizer steps; measured estimate about 72–73 hours. Add `--resume` for the same full run.

## Cosmos training

Install the training environment following the cloned repository's `docs/setup.md` for the new server's CUDA version (the original test used Python 3.13, PyTorch 2.13/cu130). Use its locked environment and training dependencies, rather than mixing the OpenPI and Cosmos environments.

```bash
cd "$TRANSFER_ROOT/cosmos-framework"
export ROBOLAB_DEMOS="$TRANSFER_ROOT/data/cosmos3_droid_demos"
export COSMOS_DROID_CHECKPOINT="$TRANSFER_ROOT/checkpoints/Cosmos3-Nano-Policy-DROID"
export WAN_VAE_PATH="$ROBOLAB_DEMOS/training_assets/wan22_native.pth"
hf download nvidia/Cosmos3-Nano-Policy-DROID --local-dir "$COSMOS_DROID_CHECKPOINT"
# Prepare the native VAE once; cached text conditioning is already included.
COSMOS_DEVICE=cpu .venv/bin/python -m cosmos_framework.scripts.prepare_robolab_vae \
  --checkpoint "$COSMOS_DROID_CHECKPOINT" --output "$WAN_VAE_PATH"
# Verify/rebuild the optimizer extension for this environment if required:
.venv/bin/python -m cosmos_framework.scripts.prepare_robolab_optimizer \
  --build-root "$ROBOLAB_DEMOS/training_assets/dependencies"
bash examples/launch_robolab_droid_lora.sh --dryrun
# Start only when ready:
bash examples/launch_robolab_droid_lora.sh
```

Cosmos uses generator video/action LoRA with frozen offline reasoning conditioning. The measured setting is 16/device × two GPUs × accumulation 4 = global batch 128, with 4,000 optimizer steps and an estimated 16.4 days on two A6000s. Hardware changes can change memory limits and timing; batch overrides preserve global batch 128. A clean installation on a second server has not been tested here.

See the included reports and each training repository's documentation for details. No full fine-tuning runs were executed during preparation.

Inherited GitHub Actions workflow files are omitted from this published branch, as authorized, because the publishing token lacks workflow scope. Upstream history and training code are retained.
