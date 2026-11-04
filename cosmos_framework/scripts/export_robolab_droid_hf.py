# SPDX-License-Identifier: OpenMDW-1.1
"""Export a RoboLab DROID fine-tuning checkpoint as an inference-ready HF policy checkpoint.

The RoboLab recipes save only the trained weights in DCP: LoRA adapters (LoRA runs) or
the fine-tuned generator weights (generator full fine-tuning). This script writes a
complete copy of the released base checkpoint (``nvidia/Cosmos3-Edge-Policy-DROID``
layout: root config/index, ``transformer/``, ``vae/``, ``vision_encoder/``,
``text_tokenizer/``, ``scheduler/``) in which only the transformer weight shards change:

* LoRA: every adapter is merged, ``W += (alpha / rank) * B @ A``. The DROID action
  boundaries (``DomainLoRA`` on ``action2llm`` / ``llm2action``) store per-domain
  ``[input, output]`` weights, so their domain-independent delta is added, transposed,
  to every domain row.
* Full fine-tuning: trained tensors replace the released ones.

Key names, shard layout, dtypes and the indexes are unchanged, so the result loads
exactly like the base model, e.g.
``python -m cosmos_framework.scripts.action_policy_server_robolab --checkpoint-path <output>``.

Usage:
    python -m cosmos_framework.scripts.export_robolab_droid_hf \\
        --run-dir <run>/  --iteration 2000 \\
        --base <full local copy of nvidia/Cosmos3-Edge-Policy-DROID> --output <dir>
"""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner

from cosmos_framework.model.generator.robolab_droid_lora import checkpoint_key

DOMAIN_LINEARS = ("action2llm", "llm2action")


def load_trained_weights(model_dir: Path) -> dict[str, torch.Tensor]:
    state: dict = {}
    dcp.state_dict_loader._load_state_dict(
        state, storage_reader=FileSystemReader(str(model_dir)), planner=_EmptyStateDictLoadPlanner(), no_dist=True
    )
    native = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = key.removeprefix("net.").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        native[name] = value
    if not native:
        raise ValueError(f"No tensors in {model_dir}")
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Run directory containing config.yaml and checkpoints/"
    )
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--base", type=Path, required=True, help="Complete local copy of the released base checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    config = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    model_config = config["model"]["config"]
    model_dir = args.run_dir / "checkpoints" / f"iter_{args.iteration:09d}" / "model"
    trained = load_trained_weights(model_dir)
    is_lora = any(".lora_" in key for key in trained)
    if is_lora and not all(".lora_" in key for key in trained):
        raise ValueError("Mixed adapter and full weights in one checkpoint")

    base_transformer = args.base / "transformer"
    index = json.loads((base_transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    native_to_released = {checkpoint_key(released): (released, filename) for released, filename in index.items()}

    # (released key, file) -> new tensor (float32 until written)
    updates: dict[tuple[str, str], torch.Tensor] = {}

    def released_tensor(native_name):
        if native_name not in native_to_released:
            raise KeyError(f"No released tensor for {native_name}")
        released, filename = native_to_released[native_name]
        with safe_open(str(base_transformer / filename), framework="pt") as source:
            return (released, filename), source.get_tensor(released)

    report: dict = {"mode": "lora" if is_lora else "generator_full_finetune", "modules": 0, "tensors": 0}
    if is_lora:
        rank, alpha = int(model_config["lora_rank"]), float(model_config["lora_alpha"])
        scale = alpha / rank
        modules = sorted({key.rsplit(".lora_", 1)[0] for key in trained})
        for module in modules:
            a = trained[f"{module}.lora_A.weight"].float()
            b = trained[f"{module}.lora_B.weight"].float()
            if a.shape[0] != rank or b.shape[1] != rank:
                raise ValueError(f"{module}: adapter rank {a.shape[0]} != configured {rank}")
            delta = scale * (b @ a)  # [out, in]
            if module in DOMAIN_LINEARS:
                key, weight = released_tensor(f"{module}.fc.weight")  # [domains, in * out], viewed [in, out]
                if weight.shape[1] != delta.numel():
                    raise ValueError(f"{module}: domain weight {tuple(weight.shape)} vs delta {tuple(delta.shape)}")
                merged = weight.float() + delta.T.reshape(1, -1)
            else:
                key, weight = released_tensor(f"{module}.weight")
                if weight.shape != delta.shape:
                    raise ValueError(f"{module}: weight {tuple(weight.shape)} vs delta {tuple(delta.shape)}")
                merged = weight.float() + delta
            updates[key] = merged
            report["modules"] += 1
        report.update(lora_rank=rank, lora_alpha=alpha, scale=scale)
    else:
        for name, value in trained.items():
            key, weight = released_tensor(name)
            if weight.shape != value.shape:
                raise ValueError(f"{name}: released {tuple(weight.shape)} vs trained {tuple(value.shape)}")
            updates[key] = value.float()
        report["modules"] = len(trained)
    report["tensors"] = len(updates)

    # Copy the base checkpoint, then rewrite only the transformer shards that changed.
    shutil.copytree(args.base, args.output, ignore=shutil.ignore_patterns(".cache", ".git*"))
    by_file = defaultdict(dict)
    for (released, filename), value in updates.items():
        by_file[filename][released] = value
    for filename, file_updates in sorted(by_file.items()):
        path = base_transformer / filename
        with safe_open(str(path), framework="pt") as source:
            metadata = source.metadata()
            tensors = {}
            for key in source.keys():
                original = source.get_tensor(key)
                tensors[key] = file_updates[key].to(original.dtype) if key in file_updates else original
        target = args.output / "transformer" / filename
        target.unlink()
        save_file(tensors, str(target), metadata=metadata)
        print(f"rewrote {filename}: {len(file_updates)} updated tensors")

    run = {
        "base_model": "nvidia/Cosmos3-Edge-Policy-DROID",
        "base_revision": "0110e64ca982589f1a3d856cbdd7d465dccb79bc",
        "source_run_dir": str(args.run_dir),
        "source_checkpoint": str(model_dir),
        "iteration": args.iteration,
        "experiment": config["job"].get("experiment") if isinstance(config.get("job"), dict) else None,
        "job_name": config["job"].get("name") if isinstance(config.get("job"), dict) else None,
        "export": report,
        "transformer_sha256": {
            filename: hashlib.sha256((args.output / "transformer" / filename).read_bytes()).hexdigest()
            for filename in sorted(by_file)
        },
    }
    (args.output / "robolab_finetune.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
