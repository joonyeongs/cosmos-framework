# SPDX-License-Identifier: OpenMDW-1.1
"""Eight RoboLab tasks, cached text conditioning, shared generator LoRA."""

import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.robolab_training_audit import RoboLabTrainingAudit
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.model.generator.robolab_droid_lora import CachedDroidLoRAModel
from cosmos_framework.utils.lazy_config import LazyCall as L

robolab_droid_lora = copy.deepcopy(action_policy_droid_nano)
r = robolab_droid_lora
r.model._target_ = CachedDroidLoRAModel
r.model.base_checkpoint = "${oc.env:COSMOS_DROID_CHECKPOINT}"
r.model.conditioning_dir = "${oc.env:COSMOS_DROID_CONDITIONING}"
r.model.precompute_only = False
c = r.model.config
c.ema.enabled = False
c.lora_enabled = True
c.lora_rank = 16
c.lora_alpha = 32
c.lora_target_modules = ",".join(
    [
        "q_proj_moe_gen",
        "k_proj_moe_gen",
        "v_proj_moe_gen",
        "o_proj_moe_gen",
        "mlp_moe_gen.up_proj",
        "mlp_moe_gen.gate_proj",
        "mlp_moe_gen.down_proj",
        "vae2llm",
        "llm2vae",
    ]
)
c.compile.enabled = False
c.parallelism.data_parallel_shard_degree = 2
c.parallelism.data_parallel_replicate_degree = 1
c.parallelism.fsdp_master_dtype = "bfloat16"
c.diffusion_expert_config.load_weights_from_pretrained = False
c.vlm_config.tokenizer.pretrained_model_name = "${oc.env:COSMOS_DROID_CHECKPOINT}"
r.optimizer.keys_to_select = ["lora_"]
r.optimizer.lr_multipliers = {}
r.optimizer.lr = 1e-4
r.scheduler.cycle_lengths = [4000]
r.scheduler.f_start = [1.0]
r.scheduler.f_max = [1.0]
r.scheduler.f_min = [0.0]
r.scheduler.warm_up_steps = [0]
r.trainer.max_iter = 4000
r.trainer.grad_accum_iter = 1  # First memory trial: 64 samples/rank * 2 ranks = 128.
r.trainer.logging_iter = 1
r.trainer.seed = 42
r.checkpoint.load_path = ""  # Load exact Policy-DROID HF weights in model constructor.
r.checkpoint.keys_to_skip_loading = []  # Never discard pretrained action weights or resume adapters.
r.checkpoint.load_training_state = True
r.checkpoint.save_iter = 250
r.checkpoint.strict_resume = False
r.dataloader_train.max_samples_per_batch = 64
r.dataloader_train.dataloader.batch_size = 1
r.dataloader_train.dataloader.num_workers = 4
dataset = r.dataloader_train.dataloader.datasets.droid.dataset
dataset.split = "full"
dataset.split_val_ratio = 0.0
dataset.use_success_only = True
dataset.use_image_augmentation = True
dataset.use_color_augmentation = False
# Optional Cosmos3-phase replay for privileged-phase training (off by default;
# examples/launch_robolab_droid_edge_lora.sh enables it with COSMOS_DROID_COSMOS3_REPLAY=1).
dataset.replay_sources = None
dataset.replay_fraction = 0.0
r.trainer.callbacks.robolab_training_audit = L(RoboLabTrainingAudit)()
if os.environ.get("COSMOS_DROID_PROFILE") == "1":
    from cosmos_framework.callbacks.robolab_phase_profile import RoboLabPhaseProfile

    r.trainer.callbacks.robolab_phase_profile = L(RoboLabPhaseProfile)()

ConfigStore.instance().store(group="experiment", package="_global_", name="robolab_droid_lora", node=r)
