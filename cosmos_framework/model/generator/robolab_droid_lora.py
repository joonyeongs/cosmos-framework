# SPDX-License-Identifier: OpenMDW-1.1
"""Generator-only DROID LoRA with offline, frozen understanding K/V.

Uses the framework's native flow-matching loss, packing, VAE, FSDP2 and trainer.
The understanding pathway is removed on meta, before any weight allocation.
Only the offline preparation process constructs the full text pathway.
"""

import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn import functional as F
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear
from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
    InferenceTextKVMemoryState,
    UndKVCache,
    install_inference_memory_attention_dispatch,
    make_inference_text_kv_cache,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def checkpoint_key(name):
    """Map released Diffusers transformer names to native VFM names."""
    prefixes = {
        "action_proj_in.": "action2llm.",
        "action_proj_out.": "llm2action.",
        "proj_in.": "vae2llm.",
        "proj_out.": "llm2vae.",
        "time_embedder.linear_1.": "time_embedder.mlp.0.",
        "time_embedder.linear_2.": "time_embedder.mlp.2.",
    }
    for old, new in prefixes.items():
        if name.startswith(old):
            return new + name[len(old) :]
    if name.startswith("lm_head."):
        return "language_model." + name
    if name.startswith(("layers.", "embed_tokens.", "norm.", "norm_moe_gen.")):
        replacements = {
            "add_q_proj": "q_proj_moe_gen",
            "add_k_proj": "k_proj_moe_gen",
            "add_v_proj": "v_proj_moe_gen",
            "to_add_out": "o_proj_moe_gen",
            "norm_added_q": "q_norm_moe_gen",
            "norm_added_k": "k_norm_moe_gen",
            "to_q": "q_proj",
            "to_k": "k_proj",
            "to_v": "v_proj",
            "to_out": "o_proj",
            "norm_q": "q_norm",
            "norm_k": "k_norm",
        }
        for old, new in replacements.items():
            name = name.replace(f".self_attn.{old}.", f".self_attn.{new}.")
        return "language_model.model." + name
    return name


@torch.no_grad()
def load_released_weights(net, checkpoint):
    """Read only parameters that actually exist; never read pruned tower tensors."""
    checkpoint = Path(checkpoint)
    index_path = checkpoint / "transformer/diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    wanted = {
        n.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", ""): p
        for n, p in net.named_parameters()
        if ".lora_" not in n
    }
    by_file = defaultdict(list)
    for source, filename in index.items():
        native = checkpoint_key(source)
        if native in wanted:
            by_file[filename].append((source, native))
    loaded = set()
    for filename, pairs in by_file.items():
        with safe_open(str(checkpoint / "transformer" / filename), framework="pt", device="cpu") as source:
            for key, native in pairs:
                target = wanted[native]
                value = source.get_tensor(key)
                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError(f"Shape mismatch {native}: {value.shape} != {target.shape}")
                if isinstance(target, DTensor):
                    value = distribute_tensor(
                        value.to(device=target.device, dtype=target.dtype), target.device_mesh, target.placements
                    )
                else:
                    value = value.to(device=target.device, dtype=target.dtype)
                target.copy_(value)
                loaded.add(native)
    missing = set(wanted) - loaded
    if missing:
        raise ValueError(f"Uninitialized base parameters: {sorted(missing)}")
    return {
        "base_checkpoint": str(checkpoint),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "loaded_parameters": sorted(loaded),
        "loaded_elements": sum(wanted[n].numel() for n in loaded),
    }


class EmptyLinear(nn.Module):
    """Shape-preserving placeholder for the absent text tower; rejects real work."""

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        if x.numel():
            raise RuntimeError("Understanding computation requested in generator-only training")
        return x.new_empty((*x.shape[:-1], self.out_features))


class TextShapeOnly(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("dtype_anchor", torch.empty(0), persistent=False)

    def forward(self, ids):
        return torch.zeros((*ids.shape, self.hidden_size), dtype=self.dtype_anchor.dtype, device=ids.device)


class FrozenFloat32Linear(nn.Linear):
    """Keep frozen BF16 storage while honoring the native FP32 timestep path."""

    def __init__(self, base):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.requires_grad_(False)

    def forward(self, x):
        return F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())


def prune_understanding_on_meta(net):
    lm = net.language_model
    if any(p.device.type != "meta" for p in lm.parameters()):
        raise ValueError("Prune before materialization, so understanding weights are never loaded")
    lm.model.embed_tokens = TextShapeOnly(lm.model.config.hidden_size)
    lm.lm_head = EmptyLinear(lm.vocab_size)
    lm.model.norm = nn.Identity()
    if hasattr(lm, "visual"):
        del lm.visual
    for layer in lm.model.layers:
        layer.mlp = nn.Identity()
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(attn, name, EmptyLinear(getattr(attn, name).out_features))
        attn.q_norm = nn.Identity()
        attn.k_norm = nn.Identity()
    remaining = [name for name, _ in lm.named_parameters() if "moe_gen" not in name]
    if remaining:
        raise ValueError(f"Understanding parameters survived pruning: {remaining}")


class DomainLoRA(DomainAwareLinear):
    """LoRA residual on the pretrained action boundary; keep every base weight."""

    def __init__(self, base, rank, alpha):
        nn.Module.__init__(self)
        self.input_size, self.output_size, self.num_domains = base.input_size, base.output_size, base.num_domains
        self.fc, self.bias = base.fc, base.bias
        self.fc.requires_grad_(False)
        self.bias.requires_grad_(False)
        self.lora_A = nn.Linear(
            self.input_size, rank, bias=False, device=self.fc.weight.device, dtype=self.fc.weight.dtype
        )
        self.lora_B = nn.Linear(
            rank, self.output_size, bias=False, device=self.fc.weight.device, dtype=self.fc.weight.dtype
        )
        self.scale = alpha / rank

    def forward(self, x, domain_id):
        return super().forward(x, domain_id) + self.lora_B(self.lora_A(x)) * self.scale

    def init_adapter(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)


def conditioning_items(packed):
    """One reusable text prefix per packed sample, independent of batch size."""
    all_ids = packed.text_ids.detach().cpu()
    indexes = packed.text_indexes.detach().cpu()
    all_positions = packed.position_ids[..., packed.text_indexes].detach().cpu()
    start = 0
    for length in packed.sample_lens:
        selected = (indexes >= start) & (indexes < start + length)
        ids = all_ids[selected].contiguous()
        positions = all_positions[..., selected].contiguous()
        digest = hashlib.sha256(ids.numpy().tobytes() + positions.numpy().tobytes()).hexdigest()
        yield digest, ids, positions
        start += length


def conditioning_key(packed):
    if len(packed.sample_lens) != 1:
        raise ValueError("Offline preparation expects exactly one prompt at a time")
    return next(conditioning_items(packed))


class CachedDroidLoRAModel(OmniMoTModel):
    def __init__(self, config, base_checkpoint, conditioning_dir, precompute_only=False):
        self.base_checkpoint_path = str(base_checkpoint)
        self.conditioning_dir = Path(conditioning_dir)
        self.precompute_only = bool(precompute_only)
        self._precompute_vae_cache = None
        self.precompute_vae_stats = {"encodes": 0, "cache_hits": 0, "determinism_checks": 0}
        # Idle-frame metadata produces multiple prefixes per task. Keep only
        # two on GPU; the current forward/backward retains its own references.
        self._conditioning = OrderedDict()
        if config.ema.enabled:
            raise ValueError("This adapter recipe requires EMA disabled")
        super().__init__(config)
        self.base_load_report = load_released_weights(self.net, self.base_checkpoint_path)
        self._assert_adapter_scope()
        if not self.precompute_only:
            self._validate_pruned_parity()

    def install_attention_dispatch(self, net):
        if not self.precompute_only:
            prune_understanding_on_meta(net)
        # Native timestep inputs and compute are FP32. A float32 autocast
        # context does not upcast BF16 parameters, so make that cast explicit
        # without changing parameter storage, names, or pretrained values.
        for index in (0, 2):
            net.time_embedder.mlp[index] = FrozenFloat32Linear(net.time_embedder.mlp[index])
        for name in ("action2llm", "llm2action"):
            base = getattr(net, name)
            if not isinstance(base, DomainAwareLinear):
                raise TypeError(f"Expected pretrained DROID domain-aware {name}")
            setattr(net, name, DomainLoRA(base, self.config.lora_rank, self.config.lora_alpha))
        install_inference_memory_attention_dispatch(net)

    def _init_lora_weights_post_materialization(self, network):
        super()._init_lora_weights_post_materialization(network)
        for module in network.modules():
            if isinstance(module, DomainLoRA):
                module.init_adapter()

    def _assert_adapter_scope(self):
        names = [name for name, param in self.net.named_parameters() if param.requires_grad]
        if not names or any(".lora_" not in name for name in names):
            raise ValueError(f"Unexpected trainable parameters: {names}")
        for part in ("action2llm", "llm2action", "vae2llm", "llm2vae", "moe_gen"):
            if not any(part in name for name in names):
                raise ValueError(f"Missing {part} adapters")

    def load_pretrained_model_if_needed(self, **kwargs):
        # The exact Policy-DROID weights were streamed at construction. Do not
        # run the base Nano recipe's understanding-to-generator initialization.
        assert self.base_load_report

    def encode(self, state):
        if not self.precompute_only:
            return super().encode(state)
        # Idle-count prompt variants reuse the same fixed video. Cache only
        # byte-identical inputs during offline preparation; training (including
        # its random image augmentation) always runs the native encoder.
        cached = self._precompute_vae_cache
        if cached is not None and torch.equal(state, cached[0]):
            if not cached[2]:
                repeated = super().encode(state)
                self.precompute_vae_stats["encodes"] += 1
                torch.testing.assert_close(repeated, cached[1], rtol=1e-5, atol=1e-6)
                self.precompute_vae_stats["determinism_checks"] += 1
                self._precompute_vae_cache = (cached[0], cached[1], True)
            self.precompute_vae_stats["cache_hits"] += 1
            return cached[1].clone()
        encoded = super().encode(state)
        self.precompute_vae_stats["encodes"] += 1
        self._precompute_vae_cache = (state.detach().clone(), encoded.detach().clone(), False)
        return encoded

    @torch.no_grad()
    def _validate_pruned_parity(self):
        # This fixture is generated locally by preparation, never downloaded.
        example_path = self.conditioning_dir / "full_model_parity.pt"
        manifest = json.loads((self.conditioning_dir / "manifest.json").read_text())
        with example_path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != manifest["parity_fixture_sha256"]:
                raise ValueError("Parity fixture checksum mismatch")
        example = torch.load(example_path, map_location="cpu", weights_only=False)
        packed = example["packed"]
        packed.to_cuda()
        memory = self.build_memory_state(packed, {})
        actual = self.net(packed_seq=packed, memory=memory)
        self.parity_report = {}
        for name in ("preds_vision", "preds_action"):
            errors = []
            for reference, prediction in zip(example["predictions"][name], actual[name]):
                reference = reference.to(prediction.device)
                torch.testing.assert_close(prediction, reference, rtol=0.02, atol=0.02)
                errors.append(float((prediction.float() - reference.float()).abs().max()))
            self.parity_report[name] = errors

    def build_memory_state(self, packed_seq, memory_info):
        if self.precompute_only:
            return InferenceTextKVMemoryState(make_inference_text_kv_cache(len(self.net.language_model.model.layers)))
        samples = [
            self._load_conditioning(key, ids, positions, packed_seq.text_ids.device)
            for key, ids, positions in conditioning_items(packed_seq)
        ]
        if len(samples) == 1:
            return InferenceTextKVMemoryState(samples[0])
        combined = []
        for layer in range(len(samples[0])):
            cache = UndKVCache()
            cache.store(
                torch.cat([sample[layer].get()[0] for sample in samples], dim=1),
                torch.cat([sample[layer].get()[1] for sample in samples], dim=1),
            )
            combined.append(cache)
        return InferenceTextKVMemoryState(combined)

    def _load_conditioning(self, key, ids, positions, device):
        if key not in self._conditioning:
            path = self.conditioning_dir / f"{key}.safetensors"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing precomputed conditioning {key}; rerun preparation for this exact prompt/layout"
                )
            with safe_open(str(path), framework="pt") as source:
                if source.metadata().get("base_index_sha256") != self.base_load_report["index_sha256"]:
                    raise ValueError("Conditioning was produced by a different base checkpoint")
            tensors = load_file(str(path), device=str(device))
            if not torch.equal(tensors.pop("text_ids").cpu(), ids) or not torch.equal(
                tensors.pop("positions").cpu(), positions
            ):
                raise ValueError("Conditioning content/hash mismatch")
            caches = []
            for i in range(len(self.net.language_model.model.layers)):
                cache = UndKVCache()
                cache.store(tensors[f"{i}.k"], tensors[f"{i}.v"])
                caches.append(cache)
            self._conditioning[key] = caches
            while len(self._conditioning) > 2:
                self._conditioning.popitem(last=False)
        self._conditioning.move_to_end(key)
        return self._conditioning[key]

    def denoise(self, net=None, data_batch_packed=None, memory=None, video_temporal_causal=None):
        out = super().denoise(
            net=net, data_batch_packed=data_batch_packed, memory=memory, video_temporal_causal=video_temporal_causal
        )
        if self.precompute_only:
            key, ids, positions = conditioning_key(data_batch_packed)
            tensors = {"text_ids": ids, "positions": positions}
            for i, cache in enumerate(memory._text_kv_cache):
                k, v = cache.get()
                tensors[f"{i}.k"], tensors[f"{i}.v"] = k.cpu().contiguous(), v.cpu().contiguous()
            self.conditioning_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                tensors,
                str(self.conditioning_dir / f"{key}.safetensors"),
                metadata={"base_index_sha256": self.base_load_report["index_sha256"]},
            )
            fixture = self.conditioning_dir / "full_model_parity.pt"
            if not fixture.exists():
                torch.save(
                    {
                        "packed": data_batch_packed,
                        "predictions": {name: out[name] for name in ("preds_vision", "preds_action")},
                    },
                    fixture,
                )
            # Same noised input, same weights: establish cached/full-path parity.
            reused = super().denoise(
                net=net, data_batch_packed=data_batch_packed, memory=memory, video_temporal_causal=video_temporal_causal
            )
            parity = {}
            for name in ("preds_vision", "preds_action"):
                errors = []
                for reference, actual in zip(out[name], reused[name]):
                    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
                    errors.append(float((actual.float() - reference.float()).abs().max()))
                parity[name] = errors
            self.last_precomputed = {"key": key, "parity_max_abs": parity, "text_tokens": len(ids)}
        return out

    def _net_state_dict(self, net, prefix="", keep_vars=False):
        # DCP saves adapters and optimizer state, avoiding 16 copies of frozen weights.
        return {
            key: value for key, value in net.state_dict(prefix=prefix, keep_vars=keep_vars).items() if ".lora_" in key
        }

    def _load_net_state_dict(self, net, state_dict):
        if any(".lora_" not in key for key in state_dict):
            raise ValueError("Only adapter checkpoints may resume this recipe")
        result = net.load_state_dict(state_dict, strict=False)
        missing = [key for key in result.missing_keys if ".lora_" in key]
        if missing or result.unexpected_keys:
            raise ValueError(f"Incomplete adapter checkpoint: {missing}, {result.unexpected_keys}")
        return _IncompatibleKeys([], [])
