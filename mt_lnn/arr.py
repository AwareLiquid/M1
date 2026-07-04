"""ARR — Attention-free Recurrent Replacement (MOHAWK/LoLCATs-style).

Turn a pretrained Llama-family transformer into an ATTENTION-FREE recurrent
model by swapping every self-attention block for an MT-v2s recurrent mixer
(multi-timescale selective-decay scan + fast-weight memory + RMC slot
attention over P state slots — no token-to-token attention anywhere), while
KEEPING the pretrained MLPs, norms and embeddings frozen. The knowledge
lives mostly in those ~1B frozen weights; distillation only has to teach
the ~tens-of-M mixers to route information the way attention did.

Why replace at the DECODER-LAYER level instead of monkey-patching
`self_attn`: HF's attention forward signature/return-arity changes across
transformers versions (2-tuple vs 3-tuple, cache object mutation). An
ARRDecoderLayer re-implements the layer's forward using the layer's OWN
pretrained norms and MLP, so the only contract with the surrounding
LlamaModel is `layer(...)[0] == hidden_states` — stable across versions.

Inference is O(1)-state: no KV cache. `model.config.use_cache` must stay
False; generation runs full-sequence forwards unless the streaming-state
path (set_adapter_streaming-style, via each mixer's MTLNNLayerV2 state
contract) is wired by the server.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .mt_lnn_v2 import MTAdapterV2Config, MTLNNLayerV2, FastWeightMemoryV2


class MTRecurrentMixer(nn.Module):
    """Drop-in token mixer replacing self-attention: (B,T,D) -> (B,T,D).

    MTLNNLayerV2 (selective decay ON by default here) + FastWeightMemoryV2
    in parallel, both at FULL residual strength (scale=1.0): unlike the
    residual-adapter use case, the mixer IS the block's output, not a
    perturbation. There is no softmax attention in this module.
    """

    def __init__(self, hidden_size: int, n_protofilaments: int = 13,
                 d_proto: int = 96, n_time_scales: int = 5,
                 proj_rank: int = 384, selective_decay: bool = True,
                 fast_weight_dim: int = 96, fast_weight_heads: int = 1):
        super().__init__()
        cfg = MTAdapterV2Config(
            hidden_size=hidden_size,
            n_protofilaments=n_protofilaments,
            d_proto=d_proto,
            n_time_scales=n_time_scales,
            proj_rank=proj_rank,
            selective_decay=selective_decay,
            use_fast_weight=False,      # FW instantiated separately below
        )
        self.mt = MTLNNLayerV2(cfg)
        self.fw = FastWeightMemoryV2(
            hidden_size, d_mem=fast_weight_dim, n_heads=fast_weight_heads,
        )
        # Learnable blend between the two recall pathways, init 50/50-ish.
        self.fw_gate = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mt_out, _ = self.mt(hidden_states)
        fw_out, _ = self.fw(hidden_states)
        g = torch.sigmoid(self.fw_gate)
        return mt_out + g * fw_out


class ARRDecoderLayer(nn.Module):
    """A Llama decoder layer with the attention replaced by MTRecurrentMixer.

    Reuses the ORIGINAL layer's pretrained input_layernorm,
    post_attention_layernorm and mlp modules (frozen). Only the mixer is new.
    Forward contract: returns a 1-tuple (hidden_states,), which is all
    LlamaModel reads when use_cache=False and output_attentions=False.
    """

    def __init__(self, base_layer: nn.Module, mixer: MTRecurrentMixer):
        super().__init__()
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.mlp = base_layer.mlp
        self.mixer = mixer
        # Return convention differs across transformers versions: older
        # LlamaModel does `layer_outputs[0]` (needs a tuple), newer uses the
        # return value directly (needs a tensor). probe_return_convention()
        # flips this flag; default matches modern transformers.
        self.return_tuple = False

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.mixer(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states,) if self.return_tuple else hidden_states


def convert_to_arr(
    model: nn.Module,
    layer_indices: Optional[List[int]] = None,
    n_protofilaments: int = 13,
    d_proto: int = 96,
    n_time_scales: int = 5,
    proj_rank: int = 384,
    selective_decay: bool = True,
    fast_weight_dim: int = 96,
) -> List[int]:
    """Swap self-attention for recurrent mixers in-place; freeze everything
    else. Returns the converted layer indices (default: ALL layers — a true
    attention-free model)."""
    from .llama_adapter import find_decoder_layers, freeze_module

    freeze_module(model)
    layers = find_decoder_layers(model)
    hidden = model.config.hidden_size
    chosen = (list(layer_indices) if layer_indices is not None
              else list(range(len(layers))))
    dtype = next(model.parameters()).dtype
    for idx in chosen:
        mixer = MTRecurrentMixer(
            hidden, n_protofilaments=n_protofilaments, d_proto=d_proto,
            n_time_scales=n_time_scales, proj_rank=proj_rank,
            selective_decay=selective_decay, fast_weight_dim=fast_weight_dim,
        ).to(dtype)
        layers[idx] = ARRDecoderLayer(layers[idx], mixer)
    model.config.use_cache = False   # no KV cache exists anymore
    return chosen


def probe_return_convention(model: nn.Module) -> bool:
    """Try a tiny forward; if the host LlamaModel chokes on tensor-returning
    layers (it indexes `layer_outputs[0]`), flip all ARR layers to
    tuple-return and retry. Returns the final `return_tuple` setting.
    Call AFTER the model is on its target device/dtype."""
    arr_layers = [m for m in model.modules() if isinstance(m, ARRDecoderLayer)]
    if not arr_layers:
        return False
    device = next(model.parameters()).device
    ids = torch.zeros(1, 4, dtype=torch.long, device=device)
    for return_tuple in (False, True):
        for l in arr_layers:
            l.return_tuple = return_tuple
        try:
            with torch.no_grad():
                model(input_ids=ids)
            return return_tuple
        except Exception:
            if return_tuple:      # both conventions failed: a real bug
                raise
    return True


def iter_mixer_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MTRecurrentMixer):
            yield from module.parameters()


def count_mixer_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in iter_mixer_parameters(model))
