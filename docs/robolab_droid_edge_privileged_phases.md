# Cosmos3 Edge fine-tuning on privileged harness phases

The [Hugging Face dataset](https://huggingface.co/datasets/joonyeongs/cosmos3-edge-harness-privileged-phases-150) stores 25 successful RoboLab episodes for each of six tasks. Use the `robolab-droid-train-edge-privileged-phases` branch of [Cosmos Framework](https://github.com/joonyeongs/cosmos-framework/tree/robolab-droid-train-edge-privileged-phases). Its LeRobot loader honors `training_start_ranges` in the episode metadata. Canned Food and Toy have empty ranges and supply zero training clips.

| Task | Episodes in export | Eligible starts |
| --- | ---: | ---: |
| Canned Food | 25 | 0 |
| Toy | 25 | 0 |
| Red Items | 25 | 3,729 |
| Cartons | 25 | 6,338 |
| Bottles | 25 | 5,955 |
| Stack | 25 | 21,494 |
| Total | 150 | 37,516 |

Each privileged-code interval is extended by ten frames at each end. Overlapping ranges are merged. An episode of length `N` can supply starts in `[0, N - 32)` because the model conditions on the start frame and requires 32 future video frames and 32 actions. Future targets can cross action-source boundaries within the same episode. Source labels are episode metadata. The inputs are the recorded RGB, instruction, and ordinary robot state. The stored gripper is 0 open and 1 closed, which the standard DROID reader and server convert to the released policy convention.

## Two-GPU launch

Install the training environment and stage the base checkpoint as described in [the Edge recipe](robolab_droid_edge_lora.md). The LeRobot video reader needs FFmpeg shared libraries compatible with TorchCodec. Authenticate Hugging Face on the training server to download the private dataset. Use a fresh conditioning directory for these four task prompts.

```bash
export ROBOLAB_DEMOS="$HOME/datasets/cosmos3-edge-harness-privileged-phases-150"
.venv/bin/hf download joonyeongs/cosmos3-edge-harness-privileged-phases-150 \
  --repo-type dataset --local-dir "$ROBOLAB_DEMOS"
export DROID_ROOT="$ROBOLAB_DEMOS/train_ready/droid_plus_lerobot_640x360_20260412"
export COSMOS_DROID_CHECKPOINT="$HOME/checkpoints/Cosmos3-Edge-Policy-DROID"
export COSMOS_DROID_GLOBAL_BATCH=64
export COSMOS_DROID_PER_DEVICE_BATCH=32

CUDA_VISIBLE_DEVICES=0 bash examples/prepare_robolab_droid_edge.sh --expected-tasks 4
CUDA_VISIBLE_DEVICES=0,1 bash examples/launch_robolab_droid_edge_lora.sh --dryrun
CUDA_VISIBLE_DEVICES=0,1 bash examples/launch_robolab_droid_edge_lora.sh
```

For generator full fine-tuning, set `COSMOS_DROID_TOML=examples/toml/sft_config/robolab_droid_edge_full.toml` when invoking the launcher. On GPUs that need a smaller microbatch, reduce `COSMOS_DROID_PER_DEVICE_BATCH`. The launcher increases accumulation to preserve global batch 64. Use a distinct output root or job name for each independent run.

The launcher reads the selected budget of **500 optimizer steps** from the dataset provenance. It uses 25 warmup steps and retains the learning rate from the selected recipe.

## Matching the previous training exposure

The [pinned dataset revision](https://huggingface.co/datasets/joonyeongs/cosmos3-edge-failure-prevention-200/tree/3fb6ca0f79e9fcecdc9e7a648cdf5ab1096223be) used by the 2,000-step run at global batch 64 contains 131,146 eligible starts across eight tasks. Matching that exposure would take `round(2,000 × 37,516 / 131,146) = 572` steps. The selected 500 steps correspond to about 0.853 passes' worth of sampled starts at the same batch size.

Restricting the historical dataset to the same six exported tasks gives 92,009 starts. Matching that six-task reference at the same global batch would take 815 steps.

## Optional: replay Cosmos3 phases at 1:4

Training only on privileged-phase windows can overwrite skills the base policy already had. In the hybrid evaluation, the 500-step model failed the Red Items mug phase, which the base policy had performed. The replay option also samples windows that start in steps the **base Cosmos3 policy** executed during collection, so the fine-tuned model keeps rehearsing them. Everything else is unchanged: same export, same 500 steps, batch, learning rate and schedule. The original privileged-only recipe remains the default.

**Which windows are sampled.**

- **Primary pool:** the existing `training_start_ranges`, which are the privileged-code intervals padded by ten frames.
- **Replay pool:** starts inside `source_intervals` entries whose source is `cosmos3`.
  - Each entry is padded by ten frames on both sides (`replay_padding_frames=10`, clipped to the episode) and overlapping entries are merged, matching the privileged padding.
  - Starts already in the primary pool are removed, so no window is in both pools.
  - Other `code_without_privileged` steps are not sampled.
  - Each start needs the same 32 future frames.
- The loader logs the replay start count at startup.
- Unpadded Cosmos3 starts on the exported data, as a lower bound:

| Task | Primary starts | Cosmos3 replay starts (unpadded) |
| --- | ---: | ---: |
| Canned Food | 0 | 1,141 |
| Toy | 0 | 768 |
| Red Items | 3,729 | 5,683 |
| Cartons | 6,338 | 0 |
| Bottles | 5,955 | 13,806 |
| Stack | 21,494 | 0 |
| Total | 37,516 | 21,398 |

**How the pools are mixed.**

- Each data-loader stream, one per rank and worker, interleaves the pools at a fixed cadence: one replay window after every four primary windows (`replay_fraction=0.2`).
- Each pool keeps the usual shuffled episode order with sequential windows inside an episode, is sharded across streams, and cycles independently.
- The primary pool's block order is identical to the privileged-only recipe; the replay pool uses its own seed offset.
- At 500 steps × global batch 64 = 32,000 sampled windows, this gives 25,600 privileged-phase windows (0.68 passes over 37,516) and 6,400 replay windows (under 0.30 passes over the padded replay pool).

Canned Food and Toy contribute replay windows, so the text-conditioning cache must cover all six task prompts. Use a fresh conditioning directory:

```bash
export COSMOS_DROID_COSMOS3_REPLAY=1
export COSMOS_DROID_CONDITIONING="$ROBOLAB_DEMOS/train_ready/conditioning_edge_cosmos3_replay"

CUDA_VISIBLE_DEVICES=0 bash examples/prepare_robolab_droid_edge.sh --expected-tasks 6
CUDA_VISIBLE_DEVICES=0,1 bash examples/launch_robolab_droid_edge_lora.sh --dryrun
CUDA_VISIBLE_DEVICES=0,1 bash examples/launch_robolab_droid_edge_lora.sh
# Generator full fine-tuning:
COSMOS_DROID_TOML=examples/toml/sft_config/robolab_droid_edge_full.toml \
  CUDA_VISIBLE_DEVICES=0,1 bash examples/launch_robolab_droid_edge_lora.sh
```

What the switch does:

- With `COSMOS_DROID_COSMOS3_REPLAY=1` the launcher sets `replay_sources=cosmos3` and `replay_fraction=0.2` on the DROID dataset, and names the job `privileged_phases_cosmos3_replay`.
- It checks that the conditioning cache holds exactly the task prompts of both pools.
- `COSMOS_DROID_REPLAY_FRACTION` changes the share; for example, 0.25 gives 1:3.
- Without the variable, the launcher, cache check and sampling behave exactly as before.
