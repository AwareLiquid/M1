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

PARTIAL conversion (a "hybrid") is supported too and is a different product:
`convert_to_arr(model, layer_indices={...})` leaves every other layer's
pretrained attention bit-identical, which means those layers still need a
KV cache. A hybrid is NOT O(1) — its carried state is
`constant recurrent state + O(T) KV of the surviving attention layers`, and
every artifact in this repo must report those two columns separately
(see docs/ARR_RATIO_PARETO.md). Only the FULL conversion is the O-series.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

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
        # Streaming state — same contract as the adapter classes (transient
        # attributes, never in state_dict). Without this the ARR student is
        # stateless across forward calls and cannot do cross-window anything.
        self.stream_enabled: bool = False
        self.stream_in_training: bool = False
        self.stream_detach: bool = True
        self._stream_h = None
        self._stream_fw = None

    def reset_stream(self) -> None:
        self._stream_h = None
        self._stream_fw = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B = hidden_states.shape[0]
        streaming = self.stream_enabled and (
            not self.training or self.stream_in_training
        )
        h_prev, fw_state = None, None
        if streaming:
            if self._stream_h is not None and self._stream_h.shape[0] != B:
                self.reset_stream()
            h_prev, fw_state = self._stream_h, self._stream_fw

        mt_out, h_last = self.mt(hidden_states, h_prev=h_prev)
        fw_out, fw_state = self.fw(hidden_states, state=fw_state)
        if streaming:
            self._stream_h = h_last.detach() if self.stream_detach else h_last
            self._stream_fw = tuple(
                (t.detach() if self.stream_detach else t) for t in fw_state
            )
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
        # The return contract is the ONE thing that differs across
        # transformers majors, and a hybrid makes it load-bearing:
        #   >=5.x  `hidden_states = decoder_layer(...)` and each attention
        #          layer writes into a shared Cache by layer_idx (verified on
        #          the pinned 5.16.1) — so a recurrent layer must return the
        #          BARE TENSOR and is simply absent from the cache, which is
        #          fine (Cache.update appends lazily up to layer_idx).
        #   4.x   `hidden_states = layer_outputs[0]` plus a cache slot read
        #          at [1]; `probe_return_convention()` flips return_tuple for
        #          that path. If a 4.x hybrid cannot consume it, run with
        #          `--no_cache` (quality accounting is unaffected).
        return (hidden_states,) if self.return_tuple else hidden_states


def convert_to_arr(
    model: nn.Module,
    layer_indices: Optional[Iterable[int]] = None,
    n_protofilaments: int = 13,
    d_proto: int = 96,
    n_time_scales: int = 5,
    proj_rank: int = 384,
    selective_decay: bool = True,
    fast_weight_dim: int = 96,
    fast_weight_heads: int = 1,
) -> List[int]:
    """Swap self-attention for recurrent mixers in-place; freeze everything
    else. Returns the converted layer indices (sorted, de-duplicated).

    `layer_indices` is the SUBSET to convert:
      None          -> every layer (historical default, bit-for-bit unchanged)
      set/list/...  -> only those layers. Every other layer keeps its
                       pretrained attention untouched: same module object,
                       same weights, same KV path — nothing is inserted,
                       wrapped or reinitialised, so the comparison at
                       ratio>0 is against an unmodified attention layer.

    A partial conversion is a HYBRID, not the O-series: the surviving
    attention layers still need a KV cache, so `config.use_cache` stays True.
    Only the full conversion (no attention left) turns it off.
    """
    from .llama_adapter import find_decoder_layers, freeze_module

    freeze_module(model)
    layers = find_decoder_layers(model)
    chosen = normalize_layer_indices(layer_indices, len(layers))
    hidden = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    for idx in chosen:
        mixer = _build_mixer(hidden, n_protofilaments, d_proto, n_time_scales,
                             proj_rank, selective_decay, fast_weight_dim,
                             fast_weight_heads).to(dtype)
        layers[idx] = ARRDecoderLayer(layers[idx], mixer)
    model.config.use_cache = len(chosen) < len(layers)
    return chosen


def _build_mixer(hidden: int, n_protofilaments: int, d_proto: int,
                 n_time_scales: int, proj_rank: int, selective_decay: bool,
                 fast_weight_dim: int,
                 fast_weight_heads: int) -> MTRecurrentMixer:
    return MTRecurrentMixer(
        hidden, n_protofilaments=n_protofilaments, d_proto=d_proto,
        n_time_scales=n_time_scales, proj_rank=proj_rank,
        selective_decay=selective_decay, fast_weight_dim=fast_weight_dim,
        fast_weight_heads=fast_weight_heads,
    )


def normalize_layer_indices(layer_indices: Optional[Iterable[int]],
                            n_layers: int) -> List[int]:
    """Canonicalise a layer subset: sorted, de-duplicated, range-checked.

    None means "every layer" — the pre-existing default, and the path the
    bit-equivalence test pins down. A set/list passes through unchanged in
    content (only ordered), so the caller's choice of layers is preserved.
    """
    if layer_indices is None:
        return list(range(n_layers))
    chosen = sorted({int(i) for i in layer_indices})
    bad = [i for i in chosen if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"layer indices out of range [0, {n_layers}): {bad}")
    return chosen


def plan_hybrid_layers(n_layers: int,
                       ratio: float) -> Tuple[List[int], List[int]]:
    """Split a stack of `n_layers` at attention fraction `ratio`.

    Returns (layers KEEPING full attention, layers CONVERTED to recurrence).
    ratio=0 -> pure ARR (the O-series control); ratio=0.25 -> the 3:1
    linear:attention mix Qwen3-Next and Kimi Linear both landed on.
    """
    keep = select_attention_layers(n_layers, ratio)
    return keep, [i for i in range(n_layers) if i not in set(keep)]


def select_attention_layers(n_layers: int, ratio: float) -> List[int]:
    """Which layers keep full attention: `ratio` of the stack, evenly spread.

    Equal-sized bins each contribute their MIDPOINT, so the surviving
    attention layers stay distributed over the whole depth instead of
    clustering at one end (a front-loaded stack degenerates into "attention
    encoder + recurrent decoder", which is a different model). The achieved
    ratio `len(result)/n_layers` is reported by the ledger — rounding to a
    whole layer count means it is only approximately `ratio`.
    """
    n_keep = int(round(ratio * n_layers))
    if n_keep <= 0:
        return []
    if n_keep >= n_layers:
        return list(range(n_layers))
    return sorted({min(n_layers - 1, (2 * i + 1) * n_layers // (2 * n_keep))
                   for i in range(n_keep)})


_RATIO_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")


def parse_ratio(text) -> float:
    """'1:4' -> 0.25 (one attention layer per four); '0' -> 0.0; '0.25' -> 0.25."""
    m = _RATIO_RE.match(str(text))
    if not m:
        return float(text)
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        raise ValueError(f"ratio denominator is zero: {text!r}")
    return num / den


def recurrent_state_elems(n_protofilaments: int = 13, d_proto: int = 96,
                          n_time_scales: int = 5, fast_weight_dim: int = 96,
                          fast_weight_heads: int = 1) -> int:
    """Elements of CONSTANT carried state per converted layer (per sequence).

    The scan state is (P, S, d_proto) and the fast-weight carry is (F, z) =
    (H, D, D) + (H, D): both are flat in T, which is the whole O(1) claim for
    the ratio-0 stack. Analytic, so it can be computed without a GPU; the
    streaming test pins it against the tensors a real forward produces.
    """
    return (n_protofilaments * n_time_scales * d_proto
            + fast_weight_heads * fast_weight_dim * (fast_weight_dim + 1))


def measured_state_bytes(model: nn.Module, elem_bytes: int = 2) -> int:
    """Bytes actually carried by the ARR mixers' streaming state, batch-1.

    Only meaningful AFTER a streaming forward — before that the state
    attributes are None and this returns 0. Its job is to audit the analytic
    `recurrent_state_elems` ledger, not to replace it.
    """
    total = 0
    for m in model.modules():
        if not isinstance(m, MTRecurrentMixer):
            continue
        for tensor in (m._stream_h, *(m._stream_fw or ())):
            if tensor is not None:
                total += tensor[0].numel() * elem_bytes
    return total


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


def set_mixer_streaming(model: nn.Module, enabled: bool,
                        train_through: bool = False) -> int:
    """Enable/disable cross-call recurrent state on all ARR mixers.

    O-series analogue of llama_adapter.set_adapter_streaming (kept separate
    on purpose: M-series adapters and O-series mixers are distinct products).
    """
    n = 0
    for m in model.modules():
        if isinstance(m, MTRecurrentMixer):
            m.stream_enabled = enabled
            m.stream_in_training = train_through
            m.stream_detach = not train_through
            m.reset_stream()
            n += 1
    return n


def reset_mixer_streams(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, MTRecurrentMixer):
            m.reset_stream()


def count_mixer_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in iter_mixer_parameters(model))
