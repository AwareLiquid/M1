import sys
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.llama_adapter import (
    attach_mt_adapters,
    collect_adapter_aux_losses,
    count_trainable_parameters,
)


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


def test_attach_mt_adapters_freezes_base_and_trains_adapter():
    torch.manual_seed(0)
    model = TinyBackbone()
    wrapped = attach_mt_adapters(
        model,
        every=2,
        n_protofilaments=4,
        n_time_scales=2,
        map_hidden_dim=8,
        init_scale=1e-2,
    )
    assert wrapped == [1, 3]
    assert count_trainable_parameters(model) > 0
    assert all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if ".base_layer." in name
    )

    x = torch.randn(2, 6, 32)
    out = model(x)
    assert out.shape == x.shape
    out.square().mean().backward()

    adapter_grads = [
        p.grad
        for name, p in model.named_parameters()
        if "mt_adapter" in name and p.requires_grad
    ]
    assert adapter_grads and all(g is not None for g in adapter_grads)


def test_adapter_aux_losses_are_live_and_trainable():
    """Predictive-coding + Hebbian, once enabled on the adapter route, must emit
    REAL gradient-bearing losses -- i.e. they are no longer dead parameters and
    are testable without the from-scratch MTLNNModel."""
    torch.manual_seed(0)
    model = TinyBackbone()
    attach_mt_adapters(
        model,
        every=2,
        n_protofilaments=4,
        n_time_scales=2,
        map_hidden_dim=8,
        init_scale=1e-2,
        use_predictive_coding=True,
        use_hebbian=True,
    )
    model.train()

    x = torch.randn(2, 6, 32)
    _ = model(x)  # populates last_pred_error / _hebb_signal on each adapter

    aux = collect_adapter_aux_losses(model)
    # both signals are present and combined
    assert "pred_loss" in aux, "predictive-coding signal missing"
    assert "hebbian_loss" in aux, "Hebbian signal missing"
    assert "aux_loss" in aux
    assert torch.is_tensor(aux["aux_loss"]) and aux["aux_loss"].requires_grad

    # the aux loss alone must produce gradients on adapter params -> not dead
    model.zero_grad()
    aux["aux_loss"].backward()
    pred_grads = [
        p.grad
        for name, p in model.named_parameters()
        if "mt_adapter" in name and p.requires_grad
    ]
    assert any(g is not None and torch.any(g != 0) for g in pred_grads), (
        "aux loss produced no gradient on adapter params -- module is inert/dead"
    )


def test_adapter_aux_losses_empty_when_disabled():
    """With both flags off (served-graph default), no aux signal is collected --
    the generation path stays untouched."""
    torch.manual_seed(0)
    model = TinyBackbone()
    attach_mt_adapters(model, every=2, n_protofilaments=4, n_time_scales=2,
                       map_hidden_dim=8, init_scale=1e-2)
    model.train()
    _ = model(torch.randn(2, 6, 32))
    assert collect_adapter_aux_losses(model) == {}
