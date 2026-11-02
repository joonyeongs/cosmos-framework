# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks for pretrained action preservation and cached-attention gradients."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

import cosmos_framework.model.generator.mot.inference_text_kv_memory as memory_module
from cosmos_framework.checkpoint.dcp import ModelWrapper
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear
from cosmos_framework.model.generator.mot.inference_text_kv_memory import InferenceTextKVMemoryState, UndKVCache
from cosmos_framework.model.generator.mot.inference_text_kv_memory_test import _gen_pack
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from cosmos_framework.model.generator.robolab_droid_lora import (
    CachedDroidLoRAModel,
    DomainLoRA,
    conditioning_items,
    load_released_weights,
    prune_understanding_on_meta,
)
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("shape", [(3, 8), (3, 5, 8)])
def test_action_adapter_preserves_pretrained_weights_and_trains(shape):
    torch.manual_seed(42)
    base = DomainAwareLinear(8, 12, 3)
    saved = {name: parameter.detach().clone() for name, parameter in base.named_parameters()}
    layer = DomainLoRA(base, rank=4, alpha=8)
    layer.init_adapter()
    x = torch.randn(shape)
    domains = torch.tensor([2, 0, 1])
    torch.testing.assert_close(layer(x, domains), base(x, domains), rtol=0, atol=0)
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=0.01)
    initial_a = layer.lora_A.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        layer(x, domains).square().mean().backward()
        optimizer.step()
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(parameter, saved[name], rtol=0, atol=0)
        assert parameter.grad is None and not parameter.requires_grad
    assert not torch.equal(layer.lora_A.weight, initial_a)
    assert torch.count_nonzero(layer.lora_B.weight)


def test_pruning_native_nano_on_meta_removes_all_understanding_parameters():
    config_path = Path(__file__).parents[0] / "reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    with torch.device("meta"):
        lm = Qwen3VLTextForCausalLM(Qwen3VLMoTConfig.from_json_file(json_file=str(config_path)))
    net = SimpleNamespace(language_model=lm)
    before = {n: p.shape for n, p in lm.named_parameters() if "moe_gen" in n}
    prune_understanding_on_meta(net)
    assert before
    assert {n: p.shape for n, p in lm.named_parameters()} == before
    assert all(p.device.type == "meta" for p in lm.parameters())


def test_selective_loader_never_opens_understanding_shards(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    net = nn.Module()
    net.vae2llm = nn.Linear(3, 4)
    values = {
        "proj_in.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "proj_in.bias": torch.arange(4, dtype=torch.float32),
    }
    save_file(values, str(transformer / "generator.safetensors"))
    index = {key: "generator.safetensors" for key in values}
    index["embed_tokens.weight"] = "understanding-must-not-be-opened.safetensors"
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    report = load_released_weights(net, tmp_path)
    assert report["loaded_parameters"] == ["vae2llm.bias", "vae2llm.weight"]
    torch.testing.assert_close(net.vae2llm.weight, values["proj_in.weight"], rtol=0, atol=0)
    torch.testing.assert_close(net.vae2llm.bias, values["proj_in.bias"], rtol=0, atol=0)


def test_adapter_state_roundtrip_does_not_overwrite_base():
    net = nn.Module()
    net.vae2llm = LoraInjectedLinear(nn.Linear(4, 5), rank=2, alpha=4)
    base = net.vae2llm.weight.detach().clone()
    state = {k: v.clone() for k, v in CachedDroidLoRAModel._net_state_dict(None, net).items()}
    assert set(state) == {"vae2llm.lora_A.weight", "vae2llm.lora_B.weight"}
    with torch.no_grad():
        net.vae2llm.lora_A.weight.zero_()
    CachedDroidLoRAModel._load_net_state_dict(None, net, state)
    torch.testing.assert_close(net.vae2llm.lora_A.weight, state["vae2llm.lora_A.weight"], rtol=0, atol=0)
    torch.testing.assert_close(net.vae2llm.weight, base, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incomplete adapter checkpoint"):
        CachedDroidLoRAModel._load_net_state_dict(None, net, {})


def test_native_dcp_adapter_save_and_restore(tmp_path):
    # Exercise the framework's real state-dict wrapper without constructing the
    # multi-billion-parameter model or taking a collection GPU.
    model = CachedDroidLoRAModel.__new__(CachedDroidLoRAModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(ema=SimpleNamespace(enabled=False))
    model.net = nn.Module()
    model.net.vae2llm = LoraInjectedLinear(nn.Linear(4, 5), rank=2, alpha=4)
    expected = {name: p.detach().clone() for name, p in model.net.named_parameters()}
    wrapper = ModelWrapper(model)
    assert all(".lora_" in name for name in wrapper.state_dict())
    dcp.save({"model": wrapper}, checkpoint_id=tmp_path / "adapter")
    with torch.no_grad():
        model.net.vae2llm.lora_A.weight.fill_(3.0)
        model.net.vae2llm.lora_B.weight.fill_(4.0)
    dcp.load({"model": wrapper}, checkpoint_id=tmp_path / "adapter")
    for name, parameter in model.net.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("num_samples", [1, 2])
def test_cached_text_attention_preserves_generator_gradients(monkeypatch, num_samples):
    torch.manual_seed(4)
    cached_k, cached_v = [torch.randn(1, 3, 2, 4, requires_grad=True) for _ in range(2)]
    q, k, v = [torch.randn(5, 2, 4, requires_grad=True) for _ in range(3)]
    cache = UndKVCache()
    cache.store(cached_k, cached_v)
    und_offsets = torch.tensor([0, 3] if num_samples == 1 else [0, 1, 3])
    gen_offsets = torch.tensor([0, 5] if num_samples == 1 else [0, 2, 5])
    state = InferenceTextKVMemoryState([cache])
    state.init(
        {"_num_full_tokens": 5, "_causal_seq_offsets": und_offsets, "_full_only_seq_offsets": gen_offsets},
        torch.device("cpu"),
    )

    def sdpa(*, query, key, value, **kwargs):
        if "cumulative_seqlen_Q" in kwargs:
            qo, ko = kwargs["cumulative_seqlen_Q"], kwargs["cumulative_seqlen_KV"]
            return torch.cat(
                [
                    sdpa(
                        query=query[:, qo[i] : qo[i + 1]],
                        key=key[:, ko[i] : ko[i + 1]],
                        value=value[:, ko[i] : ko[i + 1]],
                    )
                    for i in range(len(qo) - 1)
                ],
                dim=1,
            )
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(memory_module, "attention", sdpa)
    output, _ = memory_module._attention_gen_with_cached_text(
        *[_gen_pack(t, und_offsets, gen_offsets) for t in (q, k, v)], state.read_for_layer(0)
    )
    actual = output["full_only_seq"]
    actual.square().sum().backward()
    gradients = [t.grad.clone() for t in (q, k, v)]
    assert cached_k.grad is None and cached_v.grad is None
    for t in (q, k, v):
        t.grad = None
    expected = (
        torch.cat(
            [
                sdpa(
                    query=q[None, gen_offsets[i] : gen_offsets[i + 1]],
                    key=torch.cat(
                        [
                            cached_k[:, und_offsets[i] : und_offsets[i + 1]].detach(),
                            k[None, gen_offsets[i] : gen_offsets[i + 1]],
                        ],
                        dim=1,
                    ),
                    value=torch.cat(
                        [
                            cached_v[:, und_offsets[i] : und_offsets[i + 1]].detach(),
                            v[None, gen_offsets[i] : gen_offsets[i + 1]],
                        ],
                        dim=1,
                    ),
                )
                for i in range(num_samples)
            ],
            dim=1,
        )
        .squeeze(0)
        .flatten(-2)
    )
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for original, t in zip(gradients, (q, k, v)):
        torch.testing.assert_close(original, t.grad, rtol=1e-6, atol=1e-6)
        assert t.grad.abs().sum() > 0


def test_conditioning_keys_do_not_depend_on_packed_batch_offsets():
    first_positions = torch.arange(7).repeat(3, 1)
    second_positions = torch.arange(6).repeat(3, 1)
    one = SimpleNamespace(
        text_ids=torch.tensor([3, 4]), text_indexes=torch.tensor([0, 1]), position_ids=first_positions, sample_lens=[7]
    )
    two = SimpleNamespace(
        text_ids=torch.tensor([5, 6]), text_indexes=torch.tensor([0, 1]), position_ids=second_positions, sample_lens=[6]
    )
    packed = SimpleNamespace(
        text_ids=torch.tensor([3, 4, 5, 6]),
        text_indexes=torch.tensor([0, 1, 7, 8]),
        position_ids=torch.cat([first_positions, second_positions], dim=1),
        sample_lens=[7, 6],
    )
    assert [item[0] for item in conditioning_items(packed)] == [
        next(conditioning_items(one))[0],
        next(conditioning_items(two))[0],
    ]
