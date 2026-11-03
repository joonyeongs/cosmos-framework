# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in synchronized phase timings; writes small JSON reports, not GPU traces."""

import json
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


class RoboLabPhaseProfile(Callback):
    def on_train_start(self, model, iteration=0):
        self.path = Path(self.config.job.path_local) / f"phase_profile_rank{dist.get_rank()}.json"
        self.rows = []
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
        self.starts = {}
        self.originals = {}
        for method, label in (
            ("encode", "vae_encode"),
            ("_get_training_inputs", "inputs_including_vae"),
            ("denoise", "generator_forward"),
            ("build_memory_state", "cached_conditioning"),
            ("_compute_losses", "loss_computation"),
        ):
            original = getattr(model, method)
            self.originals[method] = original
            setattr(model, method, self._wrap(original, label))
        torch.cuda.synchronize()
        self.previous_end = time.perf_counter()

    def _wrap(self, original, label):
        @wraps(original)
        def measured(*args, **kwargs):
            self._start(label)
            try:
                return original(*args, **kwargs)
            finally:
                self._end(label)

        return measured

    def _start(self, label):
        torch.cuda.synchronize()
        self.starts[label] = time.perf_counter()

    def _end(self, label):
        torch.cuda.synchronize()
        self.seconds[label] += time.perf_counter() - self.starts.pop(label)
        self.counts[label] += 1

    def on_before_dataloading(self, iteration=0):
        self._start("dataloader_wait")

    def on_after_dataloading(self, iteration=0):
        self._end("dataloader_wait")

    def on_before_forward(self, iteration=0):
        self._start("forward_total")

    def on_after_forward(self, iteration=0):
        self._end("forward_total")

    def on_before_backward(self, model, loss, iteration=0):
        self._start("backward_including_recompute_and_communication")

    def on_after_backward(self, model, iteration=0):
        self._end("backward_including_recompute_and_communication")

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        self._start("optimizer_update")

    def on_before_zero_grad(self, model, optimizer, scheduler, iteration=0):
        self._end("optimizer_update")

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        ended = time.perf_counter()
        self.rows.append(
            {
                "iteration": iteration,
                "wall_seconds": ended - self.previous_end,
                "seconds": dict(self.seconds),
                "calls": dict(self.counts),
                "loss": loss.item(),
                "vision_tokens_last_microbatch": output_batch.get("vision_token_length"),
                "action_tokens_last_microbatch": output_batch.get("action_token_length"),
            }
        )
        self.path.write_text(json.dumps(self.rows, indent=2) + "\n")
        self.seconds.clear()
        self.counts.clear()
        self.previous_end = time.perf_counter()

    def on_train_end(self, model, iteration=0):
        for method, original in self.originals.items():
            setattr(model, method, original)
