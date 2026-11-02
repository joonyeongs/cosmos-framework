# SPDX-License-Identifier: OpenMDW-1.1
"""Precompute task/null text K/V once; the training process loads no reasoner.

Run on one free GPU after collection, using the environment defaults from
examples/prepare_robolab_droid.sh. This does no optimizer steps.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
from cosmos_framework.utils import distributed, misc
from cosmos_framework.utils.lazy_config import instantiate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-toml", default="examples/toml/sft_config/robolab_droid_lora.toml")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--expected-tasks", type=int, default=8)
    args = parser.parse_args()
    config = load_experiment_from_toml(
        args.sft_toml,
        extra_overrides=[
            "model.precompute_only=true",
            "model.config.parallelism.data_parallel_shard_degree=1",
            "model.config.parallelism.data_parallel_replicate_degree=1",
            "model.config.activation_checkpointing.mode=none",
            "model.config.compile.enabled=false",
            "dataloader_train.dataloader.datasets.droid.dataset.iterable_shuffle=false",
            "dataloader_train.dataloader.datasets.droid.dataset.use_image_augmentation=false",
            "dataloader_train.dataloader.datasets.droid.dataset.cfg_dropout_rate=0.0",
        ],
    )
    dataset = instantiate(config.dataloader_train.dataloader.datasets.droid.dataset)
    tasks = {}
    for first, length in dataset.get_shuffle_blocks():
        sample = dataset[first]
        prompt = json.loads(sample["ai_caption"])
        task = prompt["actions"][0]["description"]
        tasks.setdefault(task, sample)
        if len(tasks) == args.expected_tasks:
            break
    if len(tasks) != args.expected_tasks:
        raise ValueError(f"Expected {args.expected_tasks} unique task instructions, found {len(tasks)}")
    # The native JSON formatter includes a clip-dependent idle-frame count.
    # Preserve it exactly and cover the entire possible range, including the
    # missing-count form; image crop/color augmentation does not alter it.
    samples = {}
    for sample in tasks.values():
        for idle in [None, *range(sample["action"].shape[0] + 1)]:
            prompt = json.loads(sample["ai_caption"])
            entry = prompt["actions"][0]
            if idle is None:
                entry.pop("idle_frame", None)
            else:
                entry["idle_frame"] = f"{idle} out of {sample['action'].shape[0]}."
            variant = dict(sample, ai_caption=prompt)
            variant = dataset._transform.text_tokenizer(variant)
            samples[variant["ai_caption"]] = variant
    dataset._transform.text_tokenizer.cfg_dropout_rate = 1.0
    null_sample = dataset[0]
    assert null_sample["ai_caption"] == ""
    samples[""] = null_sample
    conditioning = Path(os.environ["COSMOS_DROID_CONDITIONING"])
    conditioning.mkdir(parents=True, exist_ok=True)
    prompt_report = {
        "task_instructions": list(tasks),
        "task_prompts": list(samples),
        "include_null_cfg_condition": True,
        "video_shapes": sorted({tuple(sample["video"].shape) for sample in samples.values()}),
    }
    (conditioning / "prompts.json").write_text(json.dumps(prompt_report, indent=2) + "\n")
    if args.metadata_only:
        print(json.dumps(prompt_report, indent=2))
        return
    if (conditioning / "manifest.json").exists():
        raise FileExistsError("Preserving existing conditioning; use a fresh directory to regenerate")
    started = time.monotonic()
    distributed.init()
    misc.set_random_seed(seed=42, by_rank=True)
    model = instantiate(config.model)
    model.eval()
    records = []
    for prompt, sample in samples.items():
        dataloader = DataLoader([sample], batch_size=1, collate_fn=custom_collate_fn, num_workers=0)
        packed_loader = PackingDataLoader(
            dataloader=dataloader,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            patch_spatial=2,
            max_samples_per_batch=1,
            dataset_name="action_droid",
        )
        batch = next(iter(packed_loader))
        with torch.no_grad():
            output, loss = model.training_step(batch, iteration=0)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite preparation loss")
        records.append({"prompt": prompt, **model.last_precomputed})
        print("CONDITIONING " + json.dumps(records[-1]), flush=True)
        # Do not retain a previous prompt's GPU outputs through the next encode.
        del output, loss, batch, packed_loader, dataloader
    fixture = conditioning / "full_model_parity.pt"
    with fixture.open("rb") as f:
        fixture_hash = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = {
        "complete": True,
        "tasks": args.expected_tasks,
        "conditions": records,
        "parity_fixture_sha256": fixture_hash,
        "base_load": model.base_load_report,
        "precomputation_seconds": time.monotonic() - started,
        "optimizer_steps": 0,
        "vae_cache": model.precompute_vae_stats,
        "training_loads_understanding_branch": False,
    }
    (conditioning / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    provenance_path = conditioning.parent / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        provenance["conditioning_status"] = "complete"
        provenance["conditioning_manifest"] = str(conditioning / "manifest.json")
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {len(records)} conditions in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
