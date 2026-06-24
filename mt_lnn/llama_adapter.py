"""
Llama + MT-LNN residual adapters.

This module keeps the experiment deliberately surgical: load a normal
HuggingFace causal LM, freeze it, then wrap selected decoder layers with a
small MT-LNN residual adapter. The base model keeps its language ability while
the adapter tests whether MT temporal dynamics add useful long-context bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .config import MTLNNConfig
from .mt_lnn_layer import MTLNNLayer


@dataclass
class MTAdapterConfig:
    hidden_size: int
    n_protofilaments: int = 13
    n_time_scales: int = 5
    map_hidden_dim: int = 64
    dropout: float = 0.0
    init_scale: float = 1e-3
    use_scan: bool = True
    # Predictive-coding aux head (the resonance bank's W_pred). It is a
    # TRAINING-ONLY auxiliary parameter -- at inference the predictive-coding
    # branch is skipped (see MTLNNLayer.forward: `if self.training`), so it never
    # affects generation output. It is kept configurable purely so the served
    # graph can be rebuilt to EXACTLY match a checkpoint that was trained with it
    # present; otherwise its 6 tensors show up as `unexpected` on load and the
    # honest load-guard would (correctly) refuse to claim the adapter is active.
    use_predictive_coding: bool = False
    # Hebbian co-activation signal. When True, MTLNNLayer.forward writes a
    # centered-covariance scalar (_hebb_signal) per adapter; an OUTER training
    # loop must call collect_adapter_aux_losses() to fold it into the objective,
    # otherwise the Hebbian path produces a live signal that earns no gradient.
    # OFF by default so the served generation graph stays untouched.
    use_hebbian: bool = False


class MTResidualAdapter(nn.Module):
    """A pre-norm MT-LNN residual adapter for a transformer hidden stream."""

    def __init__(self, config: MTAdapterConfig):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.hidden_size)
        mt_config = MTLNNConfig(
            vocab_size=1,
            max_seq_len=4096,
            d_model=config.hidden_size,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            d_head=config.hidden_size,
            n_protofilaments=config.n_protofilaments,
            n_time_scales=config.n_time_scales,
            map_hidden_dim=config.map_hidden_dim,
            dropout=config.dropout,
            attention_dropout=0.0,
            # Disable features that require model-level loss aggregation.
            # In a standalone adapter there is no outer model.forward() to
            # collect last_pred_error / _hebb_signal, so these params would
            # never receive gradients — a silent dead-parameter bug. They are
            # therefore OFF by default; use_predictive_coding is plumbed through
            # only so a graph can be rebuilt to match a checkpoint that carries
            # the (inference-inert) W_pred tensors. See MTAdapterConfig.
            use_predictive_coding=config.use_predictive_coding,
            use_world_model=False,
            # Hebbian is now plumbed through: when enabled the layer emits a
            # _hebb_signal that collect_adapter_aux_losses() turns into a real
            # gradient-bearing loss term (see that fn). world_model stays off:
            # it needs a model-level next-state target the adapter has no access
            # to, so wiring it here would still be a dead parameter -- it remains
            # an O1 / from-scratch MTLNNModel feature on purpose.
            use_hebbian=config.use_hebbian,
        )
        self.mt_layer = MTLNNLayer(mt_config)
        self.scale = nn.Parameter(torch.tensor(float(config.init_scale)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_offset: int = 0,
        h_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mt_out, _ = self.mt_layer(
            self.norm(hidden_states),
            h_prev=h_prev,
            position_offset=position_offset,
            use_scan=self.config.use_scan,
        )
        return hidden_states + self.scale * mt_out


class DecoderLayerWithMTAdapter(nn.Module):
    """Wraps a HuggingFace decoder layer and adapts its first tuple output."""

    def __init__(self, base_layer: nn.Module, adapter: MTResidualAdapter):
        super().__init__()
        self.base_layer = base_layer
        self.mt_adapter = adapter

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get('_modules', {}).get('base_layer')
            if base is not None and hasattr(base, name):
                return getattr(base, name)
            raise

    def forward(self, *args, **kwargs):
        out = self.base_layer(*args, **kwargs)
        if isinstance(out, tuple):
            hidden_states = self.mt_adapter(out[0])
            return (hidden_states,) + out[1:]

        hidden_states = getattr(out, "last_hidden_state", None)
        if hidden_states is None:
            return self.mt_adapter(out)

        out.last_hidden_state = self.mt_adapter(hidden_states)
        return out


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Locate the ModuleList of decoder layers for Llama-like HF causal LMs.

    Supports the common paths:
      - model.model.layers        (LlamaForCausalLM, Mistral, Qwen2-style)
      - model.transformer.h       (GPT-2-style fallback)
    """
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
    ]
    for first, second in candidates:
        parent = getattr(model, first, None)
        layers = getattr(parent, second, None) if parent is not None else None
        if isinstance(layers, nn.ModuleList):
            return layers
    raise ValueError(
        "Could not find decoder layers. Expected `model.model.layers` or "
        "`model.transformer.h` on the supplied HuggingFace model."
    )


def select_layer_indices(n_layers: int, every: int = 4, last: bool = True) -> List[int]:
    if every <= 0:
        raise ValueError("every must be >= 1")
    indices = list(range(every - 1, n_layers, every))
    if last and (n_layers - 1) not in indices:
        indices.append(n_layers - 1)
    return sorted(set(indices))


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def attach_mt_adapters(
    model: nn.Module,
    hidden_size: Optional[int] = None,
    layer_indices: Optional[Iterable[int]] = None,
    every: int = 4,
    n_protofilaments: int = 13,
    n_time_scales: int = 5,
    map_hidden_dim: int = 64,
    dropout: float = 0.0,
    init_scale: float = 1e-3,
    use_scan: bool = True,
    use_predictive_coding: bool = False,
    use_hebbian: bool = False,
) -> List[int]:
    """
    Freeze `model` and wrap selected decoder layers with trainable MT adapters.

    Returns the layer indices that were wrapped.

    Set use_predictive_coding / use_hebbian to expose the corresponding
    auxiliary signals; an outer training loop must then call
    collect_adapter_aux_losses(model) to fold them into the objective.
    """
    freeze_module(model)
    layers = find_decoder_layers(model)
    if hidden_size is None:
        cfg = getattr(model, "config", None)
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if hidden_size is None:
        raise ValueError("hidden_size was not provided and could not be inferred.")

    chosen = list(layer_indices) if layer_indices is not None else select_layer_indices(
        len(layers), every=every
    )
    for idx in chosen:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"layer index {idx} out of range for {len(layers)} layers")
        if isinstance(layers[idx], DecoderLayerWithMTAdapter):
            continue
        adapter_cfg = MTAdapterConfig(
            hidden_size=hidden_size,
            n_protofilaments=n_protofilaments,
            n_time_scales=n_time_scales,
            map_hidden_dim=map_hidden_dim,
            dropout=dropout,
            init_scale=init_scale,
            use_scan=use_scan,
            use_predictive_coding=use_predictive_coding,
            use_hebbian=use_hebbian,
        )
        layers[idx] = DecoderLayerWithMTAdapter(layers[idx], MTResidualAdapter(adapter_cfg).to(getattr(model, 'dtype', torch.float32)))
    return chosen


def iter_mt_adapter_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MTResidualAdapter):
            yield from module.parameters()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_adapter_aux_losses(
    model: nn.Module,
    predictive_weight: float = 0.1,
    hebbian_lr: float = 1e-4,
) -> dict:
    """Aggregate the bio-inspired auxiliary signals emitted by MT adapters.

    This is the adapter-route analogue of MTLNNModel.forward's aux-loss block:
    a standalone residual adapter has no outer model.forward() to collect the
    predictive-coding / Hebbian signals, so without this function those modules
    run but earn no gradient (they would be silent dead parameters -- the exact
    reason they used to be hard-disabled in the adapter). Call this AFTER a
    training forward pass and add the returned ``aux_loss`` to the task loss.

    Semantics match the from-scratch model:
      - predictive coding: sum of each adapter resonance bank's last_pred_error,
        scaled by ``predictive_weight`` (cf. model.py predictive_loss_weight).
      - Hebbian: ``-hebbian_lr * mean(co-activation)`` so MINIMISING the loss
        MAXIMISES co-activation (Hebb's rule), matching HebbianRegularizer.

    Returns a dict possibly containing ``pred_loss``, ``hebbian_loss`` and the
    combined ``aux_loss``. Returns ``{}`` when no signals are present (e.g. both
    flags off, or called outside a forward pass).

    HONEST NOTE: wiring these in makes the modules trainable and TESTABLE on the
    adapter route; it does NOT by itself make them EFFECTIVE. Experiment 5 found
    Hebbian inert at this scale and predictive coding is not yet fully evaluated.
    """
    import torch

    pred_terms: List[torch.Tensor] = []
    hebb_terms: List[torch.Tensor] = []
    for module in model.modules():
        if not isinstance(module, MTResidualAdapter):
            continue
        # Gate on the adapter's own config flags: the resonance bank always
        # carries a last_pred_error buffer (0 when predictive coding is off), so
        # reading it unconditionally would fold a phantom zero term into a served
        # graph that requested neither module. Only collect what was enabled.
        if module.config.use_predictive_coding:
            pe = getattr(module.mt_layer.resonance, "last_pred_error", None)
            if torch.is_tensor(pe):
                pred_terms.append(pe.reshape(()))
        if module.config.use_hebbian:
            hs = getattr(module.mt_layer, "_hebb_signal", None)
            if torch.is_tensor(hs):
                hebb_terms.append(hs.reshape(()))

    out: dict = {}
    total: Optional[torch.Tensor] = None
    if pred_terms:
        pred_loss = predictive_weight * torch.stack(pred_terms).sum()
        out["pred_loss"] = pred_loss
        total = pred_loss if total is None else total + pred_loss
    if hebb_terms:
        hebb_loss = -hebbian_lr * torch.stack(hebb_terms).mean()
        out["hebbian_loss"] = hebb_loss
        total = hebb_loss if total is None else total + hebb_loss
    if total is not None:
        out["aux_loss"] = total
    return out


def attach_adapters_from_checkpoint(model: nn.Module, checkpoint: dict) -> List[int]:
    """Recreate the MT adapter layout recorded by train_llama_mt_adapter.py."""
    saved_args = checkpoint.get("args", {})
    return attach_mt_adapters(
        model,
        every=int(saved_args.get("mt_every", 4)),
        n_protofilaments=int(saved_args.get("mt_proto", 13)),
        n_time_scales=int(saved_args.get("mt_scales", 5)),
        map_hidden_dim=int(saved_args.get("mt_map_hidden", 64)),
        dropout=float(saved_args.get("mt_dropout", 0.0)),
        init_scale=float(saved_args.get("mt_init_scale", 1e-3)),
        use_scan=not bool(saved_args.get("mt_no_scan", False)),
    )


def load_adapter_state(model: nn.Module, checkpoint_path: str, strict: bool = False) -> dict:
    """Load saved MT adapter / LoRA weights into an already-wrapped model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {
        "checkpoint": checkpoint,
        "missing": missing,
        "unexpected": unexpected,
    }
