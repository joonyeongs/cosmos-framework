# SPDX-License-Identifier: OpenMDW-1.1
"""Repair the bundled TE 2.12 extension's PyTorch ABI without replacing its core."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from importlib.metadata import distribution
from pathlib import Path
from urllib.request import urlopen

import torch

SOURCE_URL = "https://files.pythonhosted.org/packages/b0/aa/5872f0944e88b9a7a0074f207178e18d183916bc05887bf1d76947769d94/transformer_engine_torch-2.12.0.tar.gz"
SOURCE_SHA256 = "ca12cb27fa8b68818f2f11c5c4147c62ffec3d30f9db61642ea3f1b3baf3d428"
IMPORT_PROBE = "import transformer_engine.pytorch; from cosmos_framework.utils.generator.fused_adam import FusedAdam"
KERNEL_PROBE = (
    IMPORT_PROBE
    + """
import torch
p = torch.nn.Parameter(torch.zeros(32, device='cuda', dtype=torch.bfloat16))
optimizer = FusedAdam([p], lr=0.0001, master_weights=True)
p.grad = torch.ones_like(p)
optimizer.step()
torch.cuda.synchronize()
assert torch.isfinite(p).all() and torch.count_nonzero(p) == p.numel()
print('Native fused Adam CUDA update passed with BF16 parameters and FP32 master weights')
"""
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/root/RoboLab/output/cosmos3_droid_demos/training_assets/dependencies"),
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    probe = subprocess.run([sys.executable, "-c", IMPORT_PROBE], cwd=repo, capture_output=True, text=True)
    if probe.returncode == 0:
        print("Optimizer extension already loads successfully")
        return
    package = distribution("transformer_engine")
    if package.version.split("+")[0] not in ("2.12", "2.12.0"):
        raise RuntimeError(f"This pinned repair requires the installed TE 2.12 core, found {package.version}")
    if not torch.cuda.is_available() or torch.version.cuda.split(".")[0] != "13":
        raise RuntimeError("The bundled core requires an available CUDA 13 runtime")
    root = args.build_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "transformer_engine_torch-2.12.0.tar.gz"
    if not archive.exists():
        with urlopen(SOURCE_URL) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if sha256(archive) != SOURCE_SHA256:
        raise ValueError("Transformer Engine source archive checksum mismatch")
    source = root / "transformer_engine_torch-2.12.0"
    if not source.exists():
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter="data")
    architectures = sorted(
        {"".join(map(str, torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())}
    )
    environment = dict(
        os.environ,
        NVTE_FRAMEWORK="pytorch",
        NVTE_CUDA_ARCHS=";".join(architectures),
        NVTE_PYTORCH_FORCE_BUILD="TRUE",
        MAX_JOBS=str(args.jobs),
        NVTE_BUILD_MAX_JOBS=str(args.jobs),
    )
    command = [sys.executable, "setup.py", "build_ext", "--inplace"]
    log = root / "optimizer_extension_build.log"
    with log.open("w") as stream:
        subprocess.run(command, cwd=source, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    extensions = list(source.glob("transformer_engine_torch*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"Expected one compiled extension, found {extensions}")
    built = extensions[0]
    target = Path(package.locate_file("transformer_engine")) / built.name
    if not target.is_file():
        raise RuntimeError(f"Expected the bundled extension at {target}")
    original = root / (target.name + ".original")
    if original.exists():
        if sha256(original) != sha256(target):
            raise FileExistsError(f"Preserving a different previous repair backup: {original}")
    else:
        shutil.copy2(target, original)
    temporary = target.with_suffix(".repair.tmp")
    shutil.copy2(built, temporary)
    temporary.replace(target)
    try:
        verification = subprocess.run(
            [sys.executable, "-c", KERNEL_PROBE], cwd=repo, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as error:
        shutil.copy2(original, target)
        (root / "optimizer_extension_verification_failed.log").write_text(error.stdout + error.stderr)
        raise RuntimeError("Rebuilt extension verification failed; original binary restored") from error
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformer_engine_core": package.version,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "cuda_architectures": architectures,
        "build_command": command,
        "extension": str(target),
        "original_extension_sha256": sha256(original),
        "rebuilt_extension_sha256": sha256(target),
        "kernel_verification": verification.stdout.strip(),
        "core_library_replaced": False,
    }
    (root / "optimizer_extension_repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
