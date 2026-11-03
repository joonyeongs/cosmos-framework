# SPDX-License-Identifier: OpenMDW-1.1
"""Find the diagnostic reduction mesh without assuming a reasoning tower exists."""

from torch.distributed.tensor import DTensor

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock


def get_moe_device_mesh(vfm):
    for layer in vfm.language_model.model.layers:
        for name in ("mlp", "mlp_moe_gen"):
            block = getattr(layer, name, None)
            if isinstance(block, Qwen3VLMoeTextSparseMoeBlock):
                for parameter in block.parameters():
                    if isinstance(parameter, DTensor):
                        return parameter.device_mesh
    return None
