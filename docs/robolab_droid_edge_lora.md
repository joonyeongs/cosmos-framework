# RoboLab DROID Edge LoRA benchmark

See the [executed benchmark results](robolab_droid_edge_lora_results.md) for
measured step times and the projected full-run duration.

This recipe uses the same 200 demonstrations and training settings as
[the Nano recipe](robolab_droid_lora.md): two RTX A6000 48 GB GPUs, global
batch 128, BF16, full activation checkpointing, generator-only LoRA rank 16 /
alpha 32, native video/action losses and random crop/rescale, and a 4,000-step
linear learning-rate schedule. No full training run is launched by validation.

Color augmentation is now disabled in both RoboLab recipes. Edge additionally
compiles only VAE encoder normalization operations, preserving BF16 intermediate
casts; convolution and attention execute eagerly. Four data workers per GPU now
prefetch 32 clips (prefetch factor 8). Original two-way FSDP sharding is retained.
See the
[optimization report](robolab_droid_edge_optimization.md) for numerical checks
and measured performance. Historical baseline results retain their original
augmentation and parallelism settings.

The latest measured median is **132.7 seconds/optimizer step**, projecting to
**147.4 hours (6.1 days)** for 4,000 steps. No full run was executed.

The base is `nvidia/Cosmos3-Edge-Policy-DROID`, revision
`0110e64ca982589f1a3d856cbdd7d465dccb79bc`, staged at
`/root/checkpoints/Cosmos3-Edge-Policy-DROID`. It uses a 28-layer Nemotron
backbone with hidden width 2,048. Edge's MLP has up/down projections without
Nano's gate projection; LoRA targets the corresponding available projections.
The pretrained action boundaries remain frozen.

Download the checkpoint's root metadata, `text_tokenizer/`, and `transformer/`
files from that revision. The existing native Wan2.2 VAE is reused. Set
`COSMOS_DROID_CHECKPOINT` for a different checkpoint location. Then run:

```bash
.venv/bin/hf download nvidia/Cosmos3-Edge-Policy-DROID \
  --revision 0110e64ca982589f1a3d856cbdd7d465dccb79bc \
  --local-dir /root/checkpoints/Cosmos3-Edge-Policy-DROID \
  --include '*.json' '*.jinja' 'transformer/*' 'text_tokenizer/*' \
  --exclude 'vae/*' 'vision_encoder/*' 'scheduler/*'
bash examples/prepare_robolab_droid_edge.sh
.venv/bin/python -m cosmos_framework.scripts.validate_robolab_training --model edge --steps 4
```

Preparation writes a separate `train_ready/conditioning_edge` cache and checks
full-model versus cached video/action outputs. Training removes the understanding
tower before allocation and checks output parity again. Edge's frozen
generator cross-attention key normalization is retained; it is not a reasoning
parameter. Every required surviving base parameter must load from the released
checkpoint with a matching shape.

Validation starts at 64 samples/device and one accumulation step. Only CUDA OOM
triggers halving the device batch and doubling accumulation. The first fitting
configuration runs four optimizer steps and then one checkpoint-resumed step.
The audit checks finite losses, actual global batch, video/action adapter updates,
unchanged frozen boundary weights, and absence of reasoning parameters.

Results are separate from Nano:

- `training_assets/edge_batch128.json`: measured speed, memory trials and full-run estimate.
- `training_runs/robolab/cosmos3_droid_edge_lora/<run>/training_audit_initial.json`: four measured steps.
- The same run's `training_audit.json`: checkpoint-resumed step and final checks.
- `logs/edge_validation_*.log`: raw execution logs.

The steady-step estimate excludes the first optimizer step, includes data loading
and all accumulation steps, and adds startup plus 16 adapter checkpoint saves to
the projected 4,000-step duration. It is a throughput benchmark, not a measurement
of fine-tuned policy success.

To launch a full run later, after validation:

```bash
bash examples/launch_robolab_droid_edge_lora.sh
```

The launcher automatically reads the validated Edge batch configuration.
`COSMOS_DROID_PER_DEVICE_BATCH` overrides it while preserving global batch 128.
Other path overrides match the Nano launcher; Edge defaults select its own base,
conditioning directory, experiment and output group.

## Profile the bottleneck

```bash
COSMOS_DROID_PROFILE=1 COSMOS_DROID_PER_DEVICE_BATCH=32 \
  bash examples/launch_robolab_droid_edge_lora.sh -- \
  job.name="edge_profile_$(date +%Y%m%dT%H%M%S)" trainer.max_iter=2
```

The opt-in callback records synchronized wall times per GPU in
`phase_profile_rank0.json` and `phase_profile_rank1.json` in the run directory.
It measures data-loader waits, video encoding, generator forward, cached text
conditioning, loss computation, backward (including activation recomputation and
distributed communication), and the optimizer update. The first step is warm-up.
The video-encoding timer is nested within input preparation, which is itself
nested within total forward time; do not sum nested timers together. Synchronizing
at phase boundaries can perturb overlap, so compare the instrumented wall time
against the uninstrumented benchmark before using the breakdown.
