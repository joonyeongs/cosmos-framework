# RoboLab Edge live-encoding optimization

Random crop/rescale remains enabled. ColorJitter is disabled in the RoboLab Nano
and Edge recipes; original demonstrations and cached text are unchanged. Edge
also enables normalization-only VAE compilation. Only short training benchmarks
are run; the full 4,000-step training job is not launched.

**Final measured median: 132.7 seconds/optimizer step; projected 4,000-step
duration: 147.4 hours (6.1 days).** The original parallelism is retained.
This is a 1.69× throughput improvement over the 224.5-second baseline.

## End-to-end results

| Configuration | Steady seconds/step | Projected 4,000-step days |
|---|---:|---:|
| Historical baseline, color augmentation enabled | 224.5 | 10.4 |
| Color off, VAE normalization compiled | 164.8 | 7.6 |
| Same plus prefetch factor 8 (selected) | 132.7 | 6.1 |

Four steps tested color removal/normalization compilation, three tested replication,
and three tested deeper prefetching: **10 real optimizer updates** across those
three comparison runs. Every
run passed finite-loss, actual global batch 128, video/action adapter-gradient
updates and frozen-boundary checks; no reasoning parameters were loaded.

The selected run's three wall times were 166.513 (warm-up), 132.752 and 132.592
seconds. The median excludes warm-up. Peak allocated memory was 33.59 GiB/GPU.
The short runs had synchronized phase profiling enabled, so this estimate includes
that profiling overhead. Startup and 16 scheduled adapter saves are added to the
4,000-step projection. The full training run and policy-quality evaluation were
not performed. Original checkpoint-resume validation remains documented in the
[baseline report](robolab_droid_edge_lora_results.md).

The raw loader still uses four workers/GPU; prefetch factor 8 keeps up to 32 clips
ready instead of 8. Average steady data wait dropped to **0.12 seconds/step**,
from 32.4 after color removal and 91.9 in the original profile. Other selected-run
means were VAE 44.0 seconds, generator forward 20.2 seconds, and
backward/recomputation/communication 67.5 seconds. Random crops are still sampled
online. Nano receives color removal only; its updated speed has not been measured.

An accounting bug initially subtracted the final checkpoint save even though it
occurred after the last step timer. That incorrectly reported 131.5 seconds/step.
The final figures above are corrected. Raw step times are unchanged, the previous
audit files are preserved as `training_audit_before_timing_fix.json`, and a
regression test covers scheduled versus post-loop saves. All **16 focused tests**
passed, along with Ruff and diff checks.

[Training measurements and per-rank phase timings](robolab_droid_edge_optimization_training.json).
Local selection metadata is in `training_assets/edge_batch128_optimized.json` under
the demo root; the existing launch script continues using batch 32/accumulation 2.

Reproduce the selected short training run, using a fresh job name:

```bash
COSMOS_DROID_PROFILE=1 COSMOS_DROID_PER_DEVICE_BATCH=32 \
  bash examples/launch_robolab_droid_edge_lora.sh -- \
  job.name="edge_optimized_$(date +%Y%m%dT%H%M%S)" \
  trainer.max_iter=3 checkpoint.save_iter=250
```

## Encoder experiments

Four real clips (indices 0, 1000, 10000, 50000, seed 42) at 33 × 544 × 736 were
compared on an RTX A6000. Eager encoding took 0.752 seconds/clip; scoped encoder
normalization compilation took 0.681 seconds/clip, a 9.5% time reduction. All
four clips passed `abs(error) <= 0.02 + 0.02 * abs(reference)` for every latent
element. Maximum absolute error was 0.046875 and relative RMSE was 0.293–0.319%.
These checks establish small BF16 differences, not bitwise equivalence or a
measurement of trained policy quality.

Batching 2 or 4 clips and trying temporal chunks of 8, 16, 24 or 32 frames gave
no useful speedup (roughly 0.747–0.768 seconds/clip). The original serial encoding
and 24-frame chunk setting remain. Batching increased peak memory substantially.

Whole-chunk `torch.compile` took about 0.52 seconds/clip but produced incorrect
latents: maximum error about 2.86, relative RMSE about 48%. Preserving precision
casts and disabling layout optimization did not resolve it. That variant is
**rejected**. The accepted implementation compiles only `RMS_norm.forward` on
encoder instances, with `emulate_precision_casts=True`; it leaves attention,
convolutions, decoder, generator compilation settings, and checkpoint keys intact.

One-thread CPU clip preparation without color augmentation took 2.51–2.74
seconds, down from the original 7.25–7.50 seconds. Random cropping remains live
on every dataset access; no fixed latent dataset is introduced.

## Parallelism comparison

The original layout is retained: two-way FSDP sharding, one replica group,
32 samples/device and accumulation 2 (global batch 128). Context, tensor and
pipeline parallelism are disabled. Each GPU independently runs its own VAE.

A three-step comparison used a full generator copy per GPU (shard degree 1,
replicate degree 2), with all other input, optimizer and batch settings unchanged.
The frozen generator is 1,423,383,232 elements, or 2.65 GiB at BF16. Replication
fits, but provides no material throughput improvement on these NVLink-connected
A6000s:

| Layout, with color off and compiled VAE norms | Steady seconds/step | Peak allocated GiB/GPU |
|---|---:|---:|
| Original: shard 2, replicate 1 | 164.833 | 33.59 |
| Tested: shard 1, replicate 2 | 165.547 | 34.95 |

The 0.4% timing difference is too small to establish a reliable gain from these
short runs. Backward/recompute stayed around 67 seconds in both layouts. Per the
user's instruction, the published recipe keeps the original parallelism.
Both runs passed actual-global-batch, finite-loss, adapter-update and frozen-base
checks. All four repaired diagnostics were forced to run at step 2 in the
replication test and completed successfully.

To reproduce that comparison, add
`model.config.parallelism.data_parallel_shard_degree=1`
`model.config.parallelism.data_parallel_replicate_degree=2` and
`dataloader_train.dataloader.prefetch_factor=2` to the launch overrides. Use
prefetch factor 2 for the sharded comparison too; the selected recipe now defaults
to 8. The benchmark did not change activation checkpointing, batch size or LR.

## Smaller batch / more accumulation comparison

Global batch stays 128, on two GPUs with the original FSDP layout and the selected
input optimizations. Full optimizer-step times include every accumulation step;
each run excludes its first, warm-up step.

| Samples/GPU | Accumulation | Steady seconds/optimizer step | Peak allocated GiB/GPU | Status |
|---:|---:|---:|---:|---|
| **32** | **2** | **132.7** | **33.6** | Selected; 3 steps, all audits passed |
| 16 | 4 | 133.1 | 18.3 | 3 steps, all audits passed |
| 8 | 8 | 133.3 | 14.0 | Stopped by user after 2 steps; only one steady sample |

Smaller batches substantially reduced memory use without demonstrating a speed
gain. The user selected **32 × 2** and requested stopping 8 × 8. That process was
interrupted and both GPUs were verified idle. The interrupted run did not reach
its final frozen-weight/adapter-update audit and is not treated as a completed
validation. No additional training was started.

The Edge TOML and experiment now default to 32 samples/GPU and accumulation 2;
the launcher also falls back to 32 when no local validated selection file exists.
Original parallelism remains unchanged. The separate memory-sweep validator still
supports probing larger batches when explicitly run. The earlier 64 × 1 trial
OOMed in the generator MLP while requesting 3.40 GiB; it predates the input
optimizations and was not repeated in this sweep.

[Raw batch-comparison measurements](robolab_droid_edge_batch_sweep.json).

## Relationship to the supplied RoboCasa commit

The external commit was described by the user and was not retrieved. Our DROID
configuration already has `encode_exact_durations=[33]`, so native 33-frame clips
are encoded as 1 + 24 + 8 frames. Its 9-frame-to-69-frame padding reduction is
therefore not an additional opportunity here. Three camera views are composed
spatially into each batch-one clip; they are not a batch of three VAE inputs.
The existing AOT dispatcher also rejects batch sizes other than one and falls
back to eager, so it cannot silently use a batch-one package on batch three.
Supporting batch-aware AOT would matter for a future batched encoder, but our
batching measurements do not justify that change.

The described diagnostic bug does apply: scheduled MoE callbacks accessed a
reasoning projection that cached-text training replaces. They now find the mesh
on actual MoE blocks and return empty metrics for a dense generator. Regression
tests exercise all four diagnostics with the reasoning projection removed.

## Reproduce encoder validation

The measured encoder/data experiments are also saved in
[machine-readable form](robolab_droid_edge_optimization_experiments.json),
including rejected compilation variants.

Export the same checkpoint/dataset variables as
`examples/launch_robolab_droid_edge_lora.sh`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=1 \
  -m cosmos_framework.scripts.benchmark_robolab_vae \
  --output /tmp/robolab_vae_comparison.json
```

To disable just the normalization optimization for a training comparison, pass
`model.config.tokenizer.compile_encoder_norms=false` after the launcher's `--`.
Generator compilation stays disabled in both comparisons.
Run the encoder comparison again when changing the GPU or PyTorch build; the
numerical results above apply to the measured stack, and the eager override is
available if the comparison fails.
