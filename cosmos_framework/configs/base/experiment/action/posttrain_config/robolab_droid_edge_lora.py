# SPDX-License-Identifier: OpenMDW-1.1
"""Edge counterpart of the validated Nano RoboLab generator-only recipe."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.robolab_droid_lora import robolab_droid_lora
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L

robolab_droid_edge_lora = copy.deepcopy(robolab_droid_lora)
robolab_droid_edge_lora.dataloader_train.max_samples_per_batch = 32
robolab_droid_edge_lora.trainer.grad_accum_iter = 1  # global batch 64 = 32 x 2 GPUs
c = robolab_droid_edge_lora.model.config
# Compile only the numerically validated normalization kernels; keep attention
# and convolutions eager (whole-encoder compilation failed latent parity).
c.tokenizer.compile_encoder_norms = True
# Four workers can keep a full 32-clip microbatch ready while the GPUs compute.
robolab_droid_edge_lora.dataloader_train.dataloader.prefetch_factor = 8
c.vlm_config = copy.deepcopy(EDGE_MODEL_CONFIG["vlm_config"])
c.vlm_config.tokenizer = L(build_processor_lazy)(
    tokenizer_type="${oc.env:COSMOS_DROID_CHECKPOINT}",
)
c.vlm_config.model_name = "nvidia/Cosmos3-Edge-Policy-DROID"
# ReLU-squared Edge MLPs have up/down projections and no gated projection.
c.lora_target_modules = c.lora_target_modules.replace("mlp_moe_gen.gate_proj,", "")

ConfigStore.instance().store(
    group="experiment", package="_global_", name="robolab_droid_edge_lora", node=robolab_droid_edge_lora
)
