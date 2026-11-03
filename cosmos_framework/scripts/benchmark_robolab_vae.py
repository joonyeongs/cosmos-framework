# SPDX-License-Identifier: OpenMDW-1.1
"""Compare eager and normalization-only compiled VAE on actual cropped RoboLab clips.

Run with the same path environment as the Edge launcher, using torchrun with
--standalone --nproc-per-node=1. This does not launch policy training.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils import distributed
from cosmos_framework.utils.lazy_config import instantiate


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1000, 10000, 50000])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True
    distributed.init()
    config = load_experiment_from_toml(
        "examples/toml/sft_config/robolab_droid_edge_lora.toml",
        extra_overrides=[
            "dataloader_train.dataloader.datasets.droid.dataset.iterable_shuffle=false",
            "model.config.tokenizer.compile_encoder_norms=false",
        ],
    )
    dataset = instantiate(config.dataloader_train.dataloader.datasets.droid.dataset)
    clips = [dataset[index]["video"].cuda().float()[None] / 127.5 - 1 for index in args.indices]
    tokenizer = instantiate(config.model.config.tokenizer)
    reference = [tokenizer.encode(clip).float() for clip in clips]

    def measure():
        measurements = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for clip in clips:
                tokenizer.encode(clip)
            torch.cuda.synchronize()
            measurements.append((time.perf_counter() - started) / len(clips))
        return measurements

    eager_times = measure()
    state_keys = tuple(tokenizer.model.model.state_dict())
    tokenizer.compile_encoder_norms()
    started = time.perf_counter()
    actual = [tokenizer.encode(clip).float() for clip in clips]
    torch.cuda.synchronize()
    first_seconds = time.perf_counter() - started
    assert tuple(tokenizer.model.model.state_dict()) == state_keys
    parity = []
    for index, ref, out in zip(args.indices, reference, actual):
        assert ref.shape == out.shape
        difference = out - ref
        metrics = {
            "index": index,
            "max_abs": difference.abs().max().item(),
            "rmse": difference.square().mean().sqrt().item(),
            "relative_rmse": (difference.square().mean() / ref.square().mean()).sqrt().item(),
            "tolerance_fail_fraction": (difference.abs() > (0.02 + 0.02 * ref.abs())).float().mean().item(),
        }
        parity.append(metrics)
        assert metrics["tolerance_fail_fraction"] == 0, metrics
    compiled_times = measure()
    result = {
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "seed": 42,
        "input_shapes": [list(clip.shape) for clip in clips],
        "random_crop": True,
        "color_augmentation": False,
        "eager_seconds_per_clip": statistics.median(eager_times),
        "compiled_seconds_per_clip": statistics.median(compiled_times),
        "eager_timings": eager_times,
        "compiled_timings": compiled_times,
        "compile_and_first_encodes_seconds": first_seconds,
        "parity": parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
