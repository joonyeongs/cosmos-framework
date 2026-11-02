# SPDX-License-Identifier: OpenMDW-1.1
"""Convert the bundled Policy-DROID VAE losslessly to native training storage."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanVAE_


def vae_key(key):
    for root, blocks, inner, sample, sampler in (
        ("encoder", "downsamples", "downsamples", 2, "downsampler"),
        ("decoder", "upsamples", "upsamples", 3, "upsampler"),
    ):
        match = re.match(rf"{root}\.{blocks}\.(\d+)\.{inner}\.(\d+)\.(.*)", key)
        if match:
            stage, index, rest = match.groups()
            name = sampler if int(index) == sample else f"resnets.{index}"
            key = f"{root}.{'down_blocks' if root == 'encoder' else 'up_blocks'}.{stage}.{name}.{rest}"
            break
    key = key.replace(".middle.0.", ".mid_block.resnets.0.").replace(".middle.2.", ".mid_block.resnets.1.")
    key = key.replace(".middle.1.", ".mid_block.attentions.0.")
    for old, new in (
        ("residual.0.", "norm1."),
        ("residual.2.", "conv1."),
        ("residual.3.", "norm2."),
        ("residual.6.", "conv2."),
        ("shortcut.", "conv_shortcut."),
    ):
        key = key.replace(old, new)
    key = key.replace(".head.0.", ".norm_out.").replace(".head.2.", ".conv_out.")
    for old, new in (
        ("encoder.conv1.", "encoder.conv_in."),
        ("decoder.conv1.", "decoder.conv_in."),
        ("conv1.", "quant_conv."),
        ("conv2.", "post_quant_conv."),
    ):
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/checkpoints/Cosmos3-Nano-Policy-DROID"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with torch.device("meta"):
        model = WanVAE_()
    state = model.state_dict()
    source = args.checkpoint / "vae/diffusion_pytorch_model.safetensors"
    converted = {}
    with safe_open(str(source), framework="pt", device="cpu") as f:
        expected = {vae_key(k) for k in state}
        if set(f.keys()) != expected:
            raise ValueError(
                f"VAE mapping differs: missing={expected - set(f.keys())}, extra={set(f.keys()) - expected}"
            )
        for key, target in state.items():
            value = f.get_tensor(vae_key(key))
            if value.shape != target.shape:
                raise ValueError(f"VAE shape differs: {key}, {value.shape}, {target.shape}")
            # The native loader assigns checkpoint tensors on rank 0 but
            # materializes default-dtype parameters on the other ranks before
            # broadcasting. Match those native initialization dtypes exactly.
            # BF16 -> FP32 storage is lossless; the VAE casts back to BF16 for
            # computation, preserving the bundled checkpoint's actual values.
            converted[key] = value.to(dtype=target.dtype)
            if not torch.equal(converted[key].to(value.dtype), value):
                raise ValueError(f"Lossy VAE storage conversion: {key}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(converted, temporary)
    temporary.replace(args.output)
    report = {
        "source": str(source),
        "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "tensors": len(converted),
        "conversion": "Key renaming and lossless storage conversion to native initialization dtypes; runtime VAE compute remains BF16",
        "storage_dtypes": sorted({str(value.dtype) for value in converted.values()}),
        "lossless_roundtrip_verified": True,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
