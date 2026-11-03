# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
from torch import nn

from cosmos_framework.callbacks.expert_heatmap import compute_expert_heatmap
from cosmos_framework.callbacks.moe_specialization_callback import (
    compute_moe_coactivation_metrics,
    compute_moe_router_geometry_metrics,
)
from cosmos_framework.callbacks.moe_stability_callback import compute_moe_stability_metrics


@pytest.mark.L0
@pytest.mark.parametrize(
    "diagnostic",
    [
        compute_expert_heatmap,
        compute_moe_coactivation_metrics,
        compute_moe_router_geometry_metrics,
        compute_moe_stability_metrics,
    ],
)
def test_dense_generator_with_pruned_reasoner_skips_moe_diagnostics(diagnostic):
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(q_proj=nn.Identity()), mlp=nn.Identity(), mlp_moe_gen=nn.Linear(4, 4)
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    assert diagnostic(model) == {}
