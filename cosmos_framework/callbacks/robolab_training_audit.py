# SPDX-License-Identifier: OpenMDW-1.1
"""Measure real optimizer-step throughput and verify adapter-only updates."""

import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from cosmos_framework.utils.callback import Callback


def digest(parameter):
    value = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class RoboLabTrainingAudit(Callback):
    def on_train_start(self, model, iteration=0):
        self.started = time.monotonic()
        self.previous_optimizer_end = self.started
        self.startup_s = time.time() - float(os.environ.get("COSMOS_RUN_STARTED_AT", time.time()))
        self.starts, self.steps, self.saves, self.samples = {}, [], [], {}
        self.base_hashes = {
            name: digest(param)
            for name, param in model.net.named_parameters()
            if name in ("action2llm.fc.weight", "llm2action.fc.weight", "vae2llm.weight", "llm2vae.weight")
        }
        if len(self.base_hashes) != 4:
            raise AssertionError(f"Could not locate all frozen boundary weights: {list(self.base_hashes)}")
        self.adapter_hashes = {
            name: digest(param) for name, param in model.net.named_parameters() if param.requires_grad
        }
        self.path = Path(self.config.job.path_local) / "training_audit.json"
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.report = {
            "startup_seconds": self.startup_s,
            "world_size": self.world_size,
            "per_device_batch_size": self.config.dataloader_train.max_samples_per_batch,
            "gradient_accumulation_steps": self.config.trainer.grad_accum_iter,
            "effective_batch_size": self.world_size
            * self.config.trainer.grad_accum_iter
            * self.config.dataloader_train.max_samples_per_batch,
            "base_load": model.base_load_report,
            "trainable_names": sorted(self.adapter_hashes),
            "pruned_vs_full_model_parity": model.parity_report,
            "reasoning_parameters_present": [
                n for n, _ in model.net.language_model.named_parameters() if "moe_gen" not in n
            ],
            "resumed_at_iteration": iteration,
        }
        if self.report["reasoning_parameters_present"]:
            raise AssertionError("Understanding weights loaded during training")
        torch.cuda.reset_peak_memory_stats()

    def on_training_step_start(self, model, data_batch, iteration=0):
        if iteration not in self.starts:
            torch.cuda.synchronize()
            self.starts[iteration] = time.monotonic()
            self.samples[iteration] = 0
        actual_samples = len(data_batch[model.input_video_key])
        if actual_samples != self.config.dataloader_train.max_samples_per_batch:
            raise AssertionError(
                f"Packed batch has {actual_samples} samples; expected {self.config.dataloader_train.max_samples_per_batch}"
            )
        self.samples[iteration] += actual_samples

    def on_save_checkpoint_start(self, model, iteration=0):
        self.save_started = time.monotonic()

    def on_save_checkpoint_end(self, model, iteration=0):
        self.saves.append({"iteration": iteration, "seconds": time.monotonic() - self.save_started})

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        self.starts.pop(iteration - 1)
        now = time.monotonic()
        # Include data loading and callbacks between optimizer steps, not just
        # CUDA compute; this is the wall-time estimate for the user's full run.
        elapsed = now - self.previous_optimizer_end
        self.previous_optimizer_end = now
        actual_global = self.samples.pop(iteration - 1) * self.world_size
        if actual_global != 128:
            raise AssertionError(f"Expected global batch 128, observed {actual_global}")
        if not torch.isfinite(loss).all():
            raise AssertionError("Non-finite training loss")
        wall = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
        if dist.is_initialized():
            dist.all_reduce(wall, op=dist.ReduceOp.MAX)
        self.steps.append(
            {
                "iteration": iteration,
                "seconds_including_checkpoint": wall.item(),
                "loss": loss.item(),
                "actual_global_batch_size": actual_global,
            }
        )
        self._write(model)

    def _write(self, model):
        checkpoint_seconds = {entry["iteration"]: entry["seconds"] for entry in self.saves}
        steady = [
            max(0.0, row["seconds_including_checkpoint"] - checkpoint_seconds.get(row["iteration"], 0.0))
            for row in self.steps[1:]
        ]
        save_median = statistics.median(checkpoint_seconds.values()) if checkpoint_seconds else 0.0
        peak = (
            torch.tensor(
                [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                dtype=torch.float64,
                device="cuda",
            )
            / 1024**3
        )
        peaks = [torch.zeros_like(peak) for _ in range(self.world_size)]
        if dist.is_initialized():
            dist.all_gather(peaks, peak)
        else:
            peaks = [peak]
        self.report.update(
            {
                "steps": self.steps,
                "checkpoint_saves": self.saves,
                "gpu_peak_gib_by_rank": [value.tolist() for value in peaks],
            }
        )
        if steady:
            step_median = statistics.median(steady)
            self.report.update(
                {
                    "steady_optimizer_step_seconds": step_median,
                    "estimated_4000_step_seconds": self.startup_s + 4000 * step_median + 16 * save_median,
                    "observed_optimizer_step_seconds_range": [min(steady), max(steady)],
                    "estimate_basis": "Measured two-GPU optimizer steps, including all gradient accumulation; first step excluded; startup plus 16 adapter saves added",
                }
            )
        if self.rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.report, indent=2) + "\n")
            tmp.replace(self.path)

    def on_train_end(self, model, iteration=0):
        for name, param in model.net.named_parameters():
            if name in self.base_hashes and digest(param) != self.base_hashes[name]:
                raise AssertionError(f"Frozen base weight changed: {name}")
        changed = [
            name
            for name, param in model.net.named_parameters()
            if name in self.adapter_hashes and digest(param) != self.adapter_hashes[name]
        ]
        if self.steps and not changed:
            raise AssertionError("No adapter updated")
        # A can change from weight decay alone. B starts at zero, so require
        # updates to B in every video/action group as evidence of loss gradients.
        updated_groups = {}
        for part in ("action2llm", "llm2action", "vae2llm", "llm2vae", "moe_gen"):
            updated_groups[part] = [name for name in changed if part in name and ".lora_B." in name]
            if self.steps and not updated_groups[part]:
                raise AssertionError(f"No loss-gradient update reached {part} B adapters")
        self.report.update(
            {
                "frozen_weight_checks_passed": True,
                "changed_adapters": changed,
                "adapter_loss_gradient_checks": updated_groups,
                "completed_optimizer_steps_this_run": len(self.steps),
            }
        )
        self._write(model)
