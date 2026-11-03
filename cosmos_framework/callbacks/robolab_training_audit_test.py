# SPDX-License-Identifier: OpenMDW-1.1
"""Distributed audit regression: inactive local action shards are valid."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cosmos_framework.callbacks.robolab_training_audit import RoboLabTrainingAudit, digest, steady_step_times


def _audit_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        for scenario in ("active_only_on_rank_zero", "no_action_update", "frozen_change_on_rank_one"):
            net = nn.Module()
            for group in ("action2llm", "llm2action", "vae2llm", "llm2vae", "moe_gen"):
                module = nn.Module()
                module.lora_B = nn.Linear(2, 2, bias=False)
                nn.init.zeros_(module.lora_B.weight)
                module.base = nn.Parameter(torch.ones(2), requires_grad=False)
                net.add_module(group, module)
            callback = RoboLabTrainingAudit()
            callback.base_hashes = {n: digest(p) for n, p in net.named_parameters() if not p.requires_grad}
            callback.adapter_hashes = {n: digest(p) for n, p in net.named_parameters() if p.requires_grad}
            callback.steps = [{}]
            callback.report = {}
            callback._write = lambda model: None
            with torch.no_grad():
                for name, parameter in net.named_parameters():
                    if parameter.requires_grad and (rank == 0 or "llm2action" not in name):
                        if scenario != "no_action_update" or "llm2action" not in name:
                            parameter.add_(0.01)
                if scenario == "frozen_change_on_rank_one" and rank == 1:
                    net.llm2action.base.add_(0.1)
            error = None
            try:
                callback.on_train_end(SimpleNamespace(net=net), iteration=1)
            except AssertionError as exc:
                error = str(exc)
            if scenario == "active_only_on_rank_zero":
                assert error is None
                assert callback.report["frozen_weight_checks_passed"]
                assert all(callback.report["adapter_loss_gradient_checks"].values())
            elif scenario == "no_action_update":
                assert error and "No loss-gradient update reached llm2action" in error
            else:
                assert error and "Frozen base weights changed" in error
    finally:
        dist.destroy_process_group()


def test_audit_combines_shards_and_rejects_global_failures(tmp_path):
    mp.spawn(_audit_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.L0
def test_scheduled_save_is_subtracted_but_final_save_outside_timer_is_not():
    steps = [
        {"iteration": 1, "seconds_including_checkpoint": 200},
        {"iteration": 2, "seconds_including_checkpoint": 102},
        {"iteration": 3, "seconds_including_checkpoint": 100},
    ]
    saves = [
        {"iteration": 2, "seconds": 2, "included_in_step_timing": True},
        {"iteration": 3, "seconds": 3, "included_in_step_timing": False},
    ]
    assert steady_step_times(steps, saves) == [100, 100]
