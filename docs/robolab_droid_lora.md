# RoboLab DROID generator LoRA

This feature uses 200 successful RoboLab demonstrations: 25 per selected task,
one shared adapter, and no held-out validation. Both repositories use branch
`feature/robolab-cosmos3-droid-lora-demos`.

## Data and preparation

Original HDF5 episodes and three 640×360, 15 Hz camera videos live under
`/root/RoboLab/output/cosmos3_droid_demos/original`. The exported LeRobot v3 root
is `train_ready/droid_plus_lerobot_640x360_20260412/success` beneath that directory.
The native DROID loader reads raw absolute joint actions and robot state, and
composes wrist/left/right views. The original crop/rescale and ColorJitter
augmentation remain enabled. All 200 episodes participate in training; the
loader's normal complete 33-image/32-action windows are used.

After collection and export, run from `/root/cosmos-framework`:

```bash
bash examples/prepare_robolab_droid.sh
```

Preparation converts the bundled Wan2.2 VAE by a checked key-only mapping and
precomputes text K/V using the released
`/root/checkpoints/Cosmos3-Nano-Policy-DROID` weights. It covers every task's
native JSON prompt, all possible idle-frame-count variants, and the null CFG
condition. It performs no optimizer steps. A local full-model fixture and
cached/full-path comparisons check video and action output parity. Preparation
loads the frozen understanding branch once; training removes its modules on
the meta device before materializing or reading their weights. The tokenizer
is still used to identify cached prompts.

## Requested batch-size test

After all 200 successful demos have been collected, exported, and conditioned:

```bash
.venv/bin/python -m cosmos_framework.scripts.validate_robolab_training
```

The test starts with **64 samples per GPU, two GPUs, accumulation 1**. Only an
out-of-memory failure triggers the next setting: 32×2, 16×4, 8×8, 4×16, 2×32,
then 1×64 per GPU. Every setting has global batch 128. Other errors stop the
sweep for investigation. The first fitting setting runs four optimizer steps,
saves adapter checkpoints, and resumes for one further step. It does not run
the full 4,000-step job.

`training_assets/batch128.json` is written only after the test and resume pass.
It records the selected batch/accumulation, GPU memory, log paths, measured
optimizer-step time, and the estimated duration of a 4,000-step run. Timing
includes gradient accumulation, data-loading gaps, and callbacks; the first
step is excluded from the steady-state estimate. Startup and 16 scheduled
adapter saves are added. This is a short-run estimate, not a completed full
training result or an evaluation of the fine-tuned policy's success rate.

## Ready-to-run full training

Once validation is complete, the user can run:

```bash
bash examples/launch_robolab_droid_lora.sh
```

The launcher uses the validated batch configuration automatically. Before a
validated selection exists, its default remains the requested first trial
(64/device, accumulation 1). `COSMOS_DROID_PER_DEVICE_BATCH` can explicitly
override that choice; accumulation is computed to preserve global batch 128.
The launcher verifies the 200-episode quota and complete conditioning first.
`--dryrun` resolves configuration without constructing a GPU model.

The full recipe is 4,000 **optimizer** steps with rank-16, alpha-32, zero-dropout
LoRA; learning rate 0.0001 decays linearly to zero. Native AdamW settings,
gradient clipping, video/action flow-matching objectives, and image augmentation
are retained. Only generator attention/MLP projections, video input/output
projections, and action input/output adapters train. All pretrained base weights,
including DROID action boundaries, stay frozen. EMA is disabled, full activation
checkpointing is enabled, and BF16 generator parameters use two-GPU FSDP2.

Checkpoints are adapter-only DCP plus optimizer/scheduler state every 250 steps;
the base checkpoint is loaded again on resume. Re-running against the same job
directory resumes it. These adapter checkpoints are not standalone full HF
models. `training_audit.json` records actual samples per optimizer step, frozen
boundary checks, changed adapter groups, absence of understanding parameters,
parity results, timings, and per-rank memory peaks.

Environment overrides: `ROBOLAB_DEMOS`, `COSMOS_DROID_CHECKPOINT`, `DROID_ROOT`,
`COSMOS_DROID_CONDITIONING`, `WAN_VAE_PATH`, `IMAGINAIRE_OUTPUT_ROOT`,
`CUDA_VISIBLE_DEVICES`, and `COSMOS_PYTHON`. Shell launchers preserve the user's
library-path settings. The fine-tuning validation is the end of this task;
the full training run requires a later user request.
