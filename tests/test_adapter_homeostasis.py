"""
tests/test_adapter_homeostasis.py -- SHY synaptic downscaling on MT adapters
(sleep cycle, plan B), M1 adapter route.

Pins the safety + correctness contract of mt_lnn.adapter_homeostasis:

  1. The frozen base model is NEVER touched -- every base weight is bit-identical
     after a downscale pass (the core promise: homeostasis acts only on the small
     trainable adapter, the base keeps its language ability).
  2. Adapter excitatory weights are rescaled EXACTLY by the factor, and the
     rescale is direction-preserving (cosine before/after == 1) -- the SHY claim
     that the memory pattern is kept, only its magnitude shrinks.
  3. Normalisation/bias terms are spared by default (SHY targets weights).
  4. factor == 1.0 is a no-op; a model with no adapters is a safe all-zero no-op.

Scope honesty: this verifies the downscaling MECHANISM and its frozen-base
isolation. Whether periodic adapter downscaling improves continual-learning
retention is a separate effectiveness question for the training track.
"""

import copy

import torch
import torch.nn as nn

from mt_lnn.llama_adapter import attach_mt_adapters, MTResidualAdapter
from mt_lnn.adapter_homeostasis import (
    default_protect,
    downscale_adapters,
    AdapterDownscaleReport,
)


# ---------------------------------------------------------------------------
# Tiny frozen backbone + MT adapters (mirrors tests/test_llama_adapter.py)
# ---------------------------------------------------------------------------

class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        return (self.proj(hidden_states),)


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size=32, n_layers=4):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [TinyDecoderLayer(hidden_size) for _ in range(n_layers)]
        )

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def _wrapped_model():
    torch.manual_seed(0)
    model = TinyBackbone()
    attach_mt_adapters(
        model, every=2, n_protofilaments=4, n_time_scales=2,
        map_hidden_dim=8, init_scale=1e-2,
    )
    return model


# ---------------------------------------------------------------------------
# 1. Frozen base is never touched
# ---------------------------------------------------------------------------

def test_downscale_leaves_frozen_base_bit_identical():
    model = _wrapped_model()
    before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad      # the frozen base
    }
    assert before, "expected a non-empty frozen base"

    report = downscale_adapters(model, factor=0.5)
    assert report.n_adapters > 0 and report.n_tensors > 0

    for name, p in model.named_parameters():
        if name in before:
            assert torch.equal(p.detach(), before[name]), \
                f"frozen base weight {name} was modified by adapter downscaling"


# ---------------------------------------------------------------------------
# 2. Adapter weights are scaled exactly + direction preserved
# ---------------------------------------------------------------------------

def test_adapter_weights_scaled_exactly_and_direction_preserved():
    model = _wrapped_model()
    factor = 0.5
    # Snapshot the adapter weight tensors that SHOULD be rescaled.
    targets = {}
    for mod_name, module in model.named_modules():
        if isinstance(module, MTResidualAdapter):
            for name, p in module.named_parameters():
                if p.requires_grad and not default_protect(name):
                    targets[f"{mod_name}.{name}"] = p.detach().clone()
    assert targets

    downscale_adapters(model, factor=factor)

    name_to_param = dict(model.named_parameters())
    for full_name, old in targets.items():
        new = name_to_param[full_name].detach()
        assert torch.allclose(new, old * factor, atol=1e-6), \
            f"{full_name} not rescaled by exactly {factor}"
        # Direction-preserving: cosine(before, after) == 1.
        if old.abs().sum() > 0:
            cos = torch.nn.functional.cosine_similarity(
                old.flatten(), new.flatten(), dim=0
            )
            assert torch.allclose(cos, torch.tensor(1.0), atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Norm/bias terms are spared by default
# ---------------------------------------------------------------------------

def test_default_protect_spares_norm_and_bias():
    assert default_protect("layer.bias") is True
    assert default_protect("input_norm.weight") is True
    assert default_protect("proj.weight") is False

    model = _wrapped_model()
    spared = {}
    for mod_name, module in model.named_modules():
        if isinstance(module, MTResidualAdapter):
            for name, p in module.named_parameters():
                if p.requires_grad and default_protect(name):
                    spared[f"{mod_name}.{name}"] = p.detach().clone()

    downscale_adapters(model, factor=0.5)
    name_to_param = dict(model.named_parameters())
    for full_name, old in spared.items():
        assert torch.equal(name_to_param[full_name].detach(), old), \
            f"protected term {full_name} was downscaled"


# ---------------------------------------------------------------------------
# 4. No-op cases
# ---------------------------------------------------------------------------

def test_factor_one_is_noop():
    model = _wrapped_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    report = downscale_adapters(model, factor=1.0)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n])
    # Still reports the adapters it scanned (by 1.0), just changes nothing.
    assert report.factor == 1.0 and report.n_adapters > 0


def test_no_adapters_is_safe_noop():
    model = TinyBackbone()      # never wrapped -> no MTResidualAdapter
    report = downscale_adapters(model, factor=0.5)
    assert isinstance(report, AdapterDownscaleReport)
    assert report == AdapterDownscaleReport(
        n_adapters=0, factor=0.5, n_tensors=0, pre_norm=0.0, post_norm=0.0
    )


def test_post_norm_is_factor_times_pre_norm():
    model = _wrapped_model()
    factor = 0.5
    report = downscale_adapters(model, factor=factor)
    assert report.pre_norm > 0.0
    assert abs(report.post_norm - factor * report.pre_norm) < 1e-4


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception:
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
