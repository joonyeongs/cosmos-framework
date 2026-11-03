# SPDX-License-Identifier: OpenMDW-1.1
"""Generator full fine-tuning counterpart of the RoboLab Edge LoRA recipe.

Same data, cached text conditioning, batch and schedule as ``robolab_droid_edge_lora``;
instead of adapters, every surviving generator-side weight is trained (see
``FULL_FINETUNE_KEYS``). The understanding tower stays pruned and frozen, which keeps
the cached text K/V valid.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.robolab_droid_edge_lora import (
    robolab_droid_edge_lora,
)
from cosmos_framework.model.generator.robolab_droid_lora import FULL_FINETUNE_KEYS

robolab_droid_edge_full = copy.deepcopy(robolab_droid_edge_lora)
r = robolab_droid_edge_full
r.model.full_finetune = True
r.model.config.lora_enabled = False
# Optimizer allowlist mirrors the model's selection; the model additionally keeps
# k_norm_und_for_gen frozen even though it contains "_gen".
r.optimizer.keys_to_select = list(FULL_FINETUNE_KEYS)
r.optimizer.lr_multipliers = {}
# Full-weight learning rate: matches the peak of the RoboCasa Edge SFT recipe
# (5e-5 x f_max 0.4); 1e-4 is an adapter learning rate.
r.optimizer.lr = 2e-5

ConfigStore.instance().store(
    group="experiment", package="_global_", name="robolab_droid_edge_full", node=robolab_droid_edge_full
)
