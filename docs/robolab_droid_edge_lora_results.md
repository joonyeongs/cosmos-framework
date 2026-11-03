# Executed Edge LoRA benchmark — 2026-11-03

Five actual optimizer updates completed: four benchmark steps plus one step after
restoring the adapter and optimizer checkpoint. The full 4,000-step run was not executed.

**Measured steady median: 224.5 seconds/optimizer step.**
Projected 4,000-step duration: **249.5 hours
(10.4 days)**. This replaces the earlier unmeasured
6–9-day estimate. The projection includes startup and 16 scheduled adapter saves.
The three post-warm-up steps span 224.003–224.642 seconds.

| Step | Wall seconds including save | Seconds excluding save | Loss |
|---|---:|---:|---:|
| 1 | 244.099 | 244.099 | 2.04158 |
| 2 | 225.908 | 224.493 | 2.68754 |
| 3 | 224.642 | 224.642 | 1.95037 |
| 4 | 225.199 | 224.003 | 1.98867 |

Step 1 is excluded from the steady median. The restored step 5 completed in
240.672 seconds including its fresh-process warm-up.

## Configuration and comparison

- Two RTX A6000 48 GB GPUs; all 200 RoboLab demonstrations, same data and augmentation as Nano.
- Global batch 128: **32/device × 2 GPUs × accumulation 2**. The initial 64/device × accumulation 1 trial failed with CUDA OOM.
- Generator video/action LoRA rank 16, alpha 32, dropout 0; all pretrained weights frozen.
- Cached text conditioning; no understanding parameters loaded during training.
- BF16, full activation checkpointing, two-GPU FSDP, compilation and EMA disabled.
- 33-frame 544×736 composite videos, 32-action horizon; native DROID joint-position labels.
- LR 0.0001, linear decay over 4,000 optimizer steps; adapter saves every 250 steps.
- Maximum live allocated GPU memory: 33.60 GiB; maximum reserved: 41.96 GiB.

Nano measured 353.2 seconds/step with 16/device × accumulation 4,
projecting to 16.4 days. Edge measured a
**1.57× end-to-end speedup**.
These are short-run throughput measurements, not policy-success evaluations.

## Verification and reproduction

All 281 Edge conditioning variants matched full-model video/action outputs exactly.
The model with its understanding tower removed also matched exactly. Both the
initial run and checkpoint-restored run passed finite-loss, global-batch,
video/action adapter-update, and frozen-boundary-weight checks. The focused CPU
regression suite passed 12 tests; changed Python files passed Ruff.

[Recipe and commands](robolab_droid_edge_lora.md) ·
[Machine-readable measurements](robolab_droid_edge_lora_benchmark.json)

Raw logs and audits are recorded in the machine-readable measurements. Preparation
took 7.1 minutes once; it is separate from the training estimate.

## Executed bottleneck profile

Two additional real optimizer updates were instrumented on the same GPUs with the
same batch, augmentation, video/action losses and frozen-reasoner setup. The second
step took **228.6 seconds**, within 2% of the uninstrumented 224.5-second median.
Both ranks agreed closely. Phase boundaries synchronize CUDA, so the baseline
benchmark remains the source for the 4,000-step ETA.

| Part of one optimizer step | Seconds | Share |
|---|---:|---:|
| Data-loader wait | 91.9 | 40.2% |
| Backward, recomputation and communication | 67.3 | 29.5% |
| Frozen video encoder (VAE) | 48.3 | 21.1% |
| Generator forward | 20.2 | 8.8% |
| Other (including conditioning and optimizer) | 0.9 | 0.4% |

Cached conditioning averaged 0.522 seconds/step; the optimizer update
averaged 0.029 seconds. Neither is a material bottleneck. The frozen
VAE is called 64 times per GPU per optimizer step: one encode for each of 128 clips.
LoRA reduces trainable parameters and optimizer state; it still requires generator
forward/backward computation. Full activation checkpointing adds recomputation to
backward. This profile does not separately attribute backward time to compute
versus distributed communication.

### CPU preparation

A separate one-thread CPU profile sampled four augmented clips at indices 100,
1000, 10000 and 50000 after warm-up. Preparation took 7.25–7.50 seconds per clip.
Across these samples, color jitter consumed **63.7%** of wall time; hue conversion
alone consumed **43.1%** (a subset of color jitter). Video decoding consumed only
**7.9%**. Composition plus its augmentation consumed 81.2% (also inclusive, not
additive with the preceding percentages). The original augmentation is applied to
all three full-resolution views before exterior views are downscaled.

The configured raw loader has batch size 1, four workers per GPU and prefetch
factor 2: nominally eight clips queued per GPU. The outer synchronous packing
loader needs 32 clips per microbatch. Thus the ready queue is smaller than a
microbatch and does not keep a full next packed batch ready during GPU work.
The container CPU quota is about 30.72 cores, versus eight data-loader workers
across the two ranks.

### Prioritized improvements to test

1. Increase raw-loader prefetch depth (for example 2 → 8) so a complete 32-clip
   microbatch can be prepared while GPUs compute. Increase workers to 8–12/GPU
   if needed and within the CPU/memory budget. These preserve the augmentation
   and training objective. **This optimization has not yet been benchmarked.**
2. Optimize color augmentation on CPU or move equivalent augmentation to GPU.
   The existing `jitter_after_compose` option reduces processed pixels, but changes
   operation order; validate its image semantics before enabling it.
3. Investigate batched VAE encoding. The frozen encoder currently handles clips
   individually. Simply caching one latent per clip would remove random pixel
   augmentation across visits, so it is not equivalent to the approved recipe.
4. Only then tune checkpointing/compilation or distributed execution. Backward
   remains a substantial cost, but the measured first bottleneck is data delivery.

Hiding all measured data wait would give a theoretical ~136.7 seconds/step
(~6.3 days/4,000 steps), **not a demonstrated optimized runtime**. CPU contention,
memory use and overlap must be measured before claiming that speedup.

[Raw per-rank timings and CPU profile](robolab_droid_edge_lora_profile.json).
Implementation references: `cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py:306`
(view composition/augmentation), `cosmos_framework/configs/base/experiment/action/posttrain_config/robolab_droid_lora.py:64`
(loader configuration), `cosmos_framework/data/generator/joint_dataloader.py:1373`
(synchronous packing), and `cosmos_framework/model/generator/omni_mot_model.py:5354`
(per-clip VAE encoding).
