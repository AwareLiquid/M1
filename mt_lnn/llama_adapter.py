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
            # never receive gradients — a silent dead-parameter bug.
            use_predictive_coding=False,
            use_world_model=False,
            use_hebbian=False,
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
) -> List[int]:
    """
    Freeze `model` and wrap selected decoder layers with trainable MT adapters.

    Returns the layer indices that were wrapped.
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
        )
        layers[idx] = DecoderLayerWithMTAdapter(layers[idx], MTResidualAdapter(adapter_cfg).to(getattr(model, 'dtype', torch.float32)))
    return chosen


def iter_mt_adapter_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MTResidualAdapter):
            yield from module.parameters()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
