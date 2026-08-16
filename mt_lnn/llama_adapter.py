"""
Llama + MT-LNN residual adapters.

This module keeps the experiment deliberately surgical: load a normal
HuggingFace causal LM, freeze it, then wrap selected decoder layers with a
small MT-LNN residual adapter. The base model keeps its language ability while
the adapter tests whether MT temporal dynamics add useful long-context bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # Fast-weight associative memory (Ba et al. 2016 / gated linear attention).
    # Unlike the Hebbian REGULARIZER -- a tiny scalar side-loss whose gradient
    # (~1e-12) vanishes against CE -- this is a FORWARD-PASS mechanism: it writes
    # key->value outer products into a decaying fast-weight matrix and reads them
    # back associatively, all inside the autograd graph, so the main task loss
    # trains it directly. Its effect on the output is first-order, so associative
    # recall (e.g. needle-in-haystack) is learnable even at small scale. OFF by
    # default so the served graph stays untouched until explicitly enabled.
    use_fast_weight: bool = False
    fast_weight_dim: int = 64     # per-head key/value memory width (d_mem)
    fast_weight_heads: int = 1
    fast_weight_init_decay: float = 0.95   # initial association half-life (in (0,1))


class FastWeightMemory(nn.Module):
    """Causal fast-weight associative memory (Ba et al. 2016; gated linear attn).

    Within a sequence it WRITES key->value associations into a per-sample fast-
    weight matrix F and READS them back by content, all inside the autograd graph
    so the main task loss (CE) trains it end-to-end. This is the architectural
    answer to two findings at once:

      * Hebbian-as-side-loss is inert (gradient ~1e-12, drowned by CE). A forward-
        pass memory has a first-order effect on the output, so it is learnable at
        0.5B scale -- no need to wait for 4-8B.
      * The served adapter had no content-addressable memory, hence the honest
        needle-in-haystack 0% result. Fast weights are the principled fix.

    Recurrence (per position t, strictly causal):
        k_t, q_t = phi(W_k x_t), phi(W_q x_t)     # phi = elu+1 -> positive features
        v_t      = W_v x_t
        F_t = lam * F_{t-1} + k_t (outer) v_t      # write
        z_t = lam * z_{t-1} + k_t                  # running key normaliser
        r_t = (q_t @ F_t) / (q_t . z_t + eps)      # associative read
        out = W_o r_t
    lam in (0,1) is a learnable per-head decay (sigmoid of a raw param): how long
    a written association survives. F_t depends only on positions <= t, so the
    read is causal and safe for autoregressive LM.

    NOTE: this reference forward scans T sequentially (O(T) python steps) for
    clarity and exactness. A chunked/parallel scan is the production speed
    optimisation and does not change the math.
    """

    def __init__(self, d_model: int, d_mem: int = 64, n_heads: int = 1,
                 init_decay: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        inner = n_heads * d_mem
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)
        # Learnable decay per head, initialised near `init_decay` via logit.
        init_decay = min(max(init_decay, 1e-3), 1 - 1e-3)
        raw = math.log(init_decay / (1.0 - init_decay))
        self.decay_raw = nn.Parameter(torch.full((n_heads,), float(raw)))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, T, d_model). Returns (out (B,T,d_model), (F, z)) where the
        returned state can be fed back in to continue the memory across calls
        (used later for streaming / cross-session persistence)."""
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_mem
        k = (F.elu(self.W_k(x)) + 1.0).view(B, T, H, D)
        q = (F.elu(self.W_q(x)) + 1.0).view(B, T, H, D)
        v = self.W_v(x).view(B, T, H, D)
        decay = torch.sigmoid(self.decay_raw)              # (H,)
        dF = decay.view(1, H, 1, 1)
        dz = decay.view(1, H, 1)

        if state is None:
            Fmat = x.new_zeros(B, H, D, D)                  # (B,H,d_k,d_v)
            zvec = x.new_zeros(B, H, D)                     # (B,H,d_k)
        else:
            Fmat, zvec = state

        reads = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]          # each (B,H,D)
            Fmat = dF * Fmat + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            zvec = dz * zvec + kt
            num = torch.einsum("bhd,bhde->bhe", qt, Fmat)   # (B,H,d_v)
            den = torch.einsum("bhd,bhd->bh", qt, zvec).clamp_min(1e-6).unsqueeze(-1)
            reads.append((num / den).reshape(B, H * D))
        r = torch.stack(reads, dim=1)                       # (B,T,H*D)
        return self.W_o(r), (Fmat, zvec)


class MTResidualAdapter(nn.Module):
    """A pre-norm MT-LNN residual adapter for a transformer hidden stream.

    STREAMING STATE (opt-in via set_adapter_streaming): the MT layer and the
    fast-weight memory both have an explicit recurrent-state contract
    (h_prev in / h_last out), but the HF decoder-layer wrapper calls this
    adapter without state -- so during KV-cached generation every T=1 decode
    step used to start from ZERO state: the "recurrence" the adapter was
    TRAINED with (full-sequence scan) silently degraded to a per-token gated
    FFN at inference. With streaming enabled the adapter carries its own
    state across forward calls, restoring train-time semantics. State is
    detached (inference-only) and must be reset at each sequence start
    (reset_adapter_streams) and whenever the KV cache is re-primed.
    """

    def __init__(self, config: MTAdapterConfig):
        super().__init__()
        self.config = config
        # Streaming state -- plain attributes, NOT buffers: transient, never
        # part of state_dict, never saved to checkpoints.
        self.stream_enabled: bool = False
        # Trainers that BACKPROP through carried state (e.g. the cross-window
        # recall task: segment A -> state -> segment B loss) flip these two:
        # stream_in_training lets streaming run in train mode, stream_detach=
        # False keeps the carried state in the autograd graph.
        self.stream_in_training: bool = False
        self.stream_detach: bool = True
        self._stream_h: Optional[torch.Tensor] = None
        self._stream_fw: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._stream_pos: int = 0
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

        # Fast-weight associative memory: a forward-pass content-addressable
        # memory trained directly by the task loss (see FastWeightMemory). OFF by
        # default so the served generation graph is byte-identical until enabled.
        self.fast_weight: Optional[FastWeightMemory] = None
        if config.use_fast_weight:
            self.fast_weight = FastWeightMemory(
                config.hidden_size,
                d_mem=config.fast_weight_dim,
                n_heads=config.fast_weight_heads,
                init_decay=config.fast_weight_init_decay,
            )
            # Its own residual scale, init small so it starts as a gentle
            # correction (matching the MT branch's init_scale convention).
            self.fw_scale = nn.Parameter(torch.tensor(float(config.init_scale)))

    def reset_stream(self) -> None:
        self._stream_h = None
        self._stream_fw = None
        self._stream_pos = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_offset: int = 0,
        h_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = hidden_states.shape[0]
        streaming = self.stream_enabled and (
            not self.training or self.stream_in_training
        )
        if streaming and h_prev is None:
            # Batch-size change means a new, unrelated batch: drop stale state.
            if self._stream_h is not None and self._stream_h.shape[0] != B:
                self.reset_stream()
            h_prev = self._stream_h
            position_offset = self._stream_pos

        normed = self.norm(hidden_states)
        mt_out, h_last = self.mt_layer(
            normed,
            h_prev=h_prev,
            position_offset=position_offset,
            use_scan=self.config.use_scan,
        )
        out = hidden_states + self.scale * mt_out
        if self.fast_weight is not None:
            fw_state = self._stream_fw if streaming else None
            if fw_state is not None and fw_state[0].shape[0] != B:
                fw_state = None
            fw_out, fw_state = self.fast_weight(normed, state=fw_state)
            out = out + self.fw_scale * fw_out
            if streaming:
                self._stream_fw = tuple(
                    (t.detach() if self.stream_detach else t) for t in fw_state
                )
        if streaming:
            self._stream_h = h_last.detach() if self.stream_detach else h_last
            self._stream_pos = position_offset + hidden_states.shape[1]
        return out


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
    use_fast_weight: bool = False,
    fast_weight_dim: int = 64,
    fast_weight_heads: int = 1,
    fast_weight_init_decay: float = 0.95,
) -> List[int]:
    """
    Freeze `model` and wrap selected decoder layers with trainable MT adapters.

    Returns the layer indices that were wrapped.

    Set use_predictive_coding / use_hebbian to expose the corresponding
    auxiliary signals; an outer training loop must then call
    collect_adapter_aux_losses(model) to fold them into the objective.

    Set use_fast_weight to add a forward-pass associative memory (FastWeightMemory)
    to each adapter. Unlike the aux signals it needs NO outer collection: it is a
    differentiable mechanism in the forward pass, trained directly by the task
    loss.
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
            use_fast_weight=use_fast_weight,
            fast_weight_dim=fast_weight_dim,
            fast_weight_heads=fast_weight_heads,
            fast_weight_init_decay=fast_weight_init_decay,
        )
        layers[idx] = DecoderLayerWithMTAdapter(layers[idx], MTResidualAdapter(adapter_cfg).to(getattr(model, 'dtype', torch.float32)))
    return chosen


def iter_mt_adapter_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MTResidualAdapter):
            yield from module.parameters()


def _iter_all_adapters(model: nn.Module):
    """Yield every v1 AND v2 residual adapter in the model.

    Tolerates non-nn.Module models (callers like mt_lnn.thinking are
    model-agnostic and may pass bare callables) — no modules() means no
    adapters, an empty iteration.
    """
    from .mt_lnn_v2 import MTResidualAdapterV2  # lazy: v2 imports from this module

    modules = getattr(model, "modules", None)
    if modules is None:
        return
    for module in modules():
        if isinstance(module, (MTResidualAdapter, MTResidualAdapterV2)):
            yield module


def _iter_stream_modules(model: nn.Module):
    """Yield every module carrying the streaming-state contract — the M-series
    residual adapters AND the O-series ARR mixers (MTRecurrentMixer). All of
    them hold ``_stream_h`` / ``_stream_fw`` and a ``reset_stream``; the mixers
    lack ``_stream_pos`` (they have no GTP clock), so callers must read it with
    getattr. Used by the state-lifecycle helpers (set/reset/snapshot/restore/
    pause) so persistence covers BOTH product lines — an ARR model's fast-weight
    would otherwise be silently skipped by the adapter-only iterator."""
    from .mt_lnn_v2 import MTResidualAdapterV2
    from .arr import MTRecurrentMixer

    modules = getattr(model, "modules", None)
    if modules is None:
        return
    for module in modules():
        if isinstance(module, (MTResidualAdapter, MTResidualAdapterV2,
                               MTRecurrentMixer)):
            yield module


def set_adapter_streaming(model: nn.Module, enabled: bool) -> int:
    """Enable/disable cross-call recurrent state on all MT adapters (v1+v2).

    Returns the number of adapters touched (0 = plain model, safe no-op).
    Enabling also clears any stale state. Streaming is inference-only: the
    adapters ignore the flag in training mode.
    """
    n = 0
    for adapter in _iter_stream_modules(model):   # adapters + ARR mixers
        adapter.stream_enabled = enabled
        adapter.reset_stream()
        n += 1
    return n


def reset_adapter_streams(model: nn.Module) -> None:
    """Zero every streaming module's state (adapters AND ARR mixers). Call at
    each sequence start and whenever the KV cache is re-primed on a rebuilt
    context (a re-fed prompt would otherwise be double-written into the
    recurrent state)."""
    for adapter in _iter_stream_modules(model):
        adapter.reset_stream()


def snapshot_adapter_streams(model: nn.Module) -> dict:
    """Capture every adapter's live streaming state as a persistable dict.

    Turns the transient, request-volatile fast-weight/recurrent state
    (``_stream_h``, ``_stream_fw`` = the (F, z) pair, ``_stream_pos``) into a
    plain dict of CPU float32 tensors that can be ``torch.save``d and reloaded
    into a FRESH process — the fast->slow (hippocampus->durable) transfer the
    consolidation stack needs but never had a source for.

    Keyed by deterministic ``_iter_all_adapters`` enumeration order so a
    snapshot restores onto the same adapters it came from. The (F, z) PAIR is
    captured atomically: F alone is unusable because the associative read
    divides by q·z, so persisting one without its exactly-paired other yields
    garbage. Tensors are detached to CPU float32 (the bf16/fp16 round-trip
    would otherwise bleed low-order bits of the magnitude-heavy DxD sums each
    consolidation cycle).

    Covers M-series adapters AND O-series ARR mixers (see
    :func:`_iter_stream_modules`); mixers have no ``_stream_pos`` so it is read
    with getattr and defaults to 0.

    Returns ``{"i0": {...}, "i1": {...}}`` where each entry has ``h`` (or None),
    ``fw`` (a 2-list [F, z] or None) and ``pos``. Empty streams snapshot as
    None — a fresh, never-written module round-trips to itself.
    """
    def _cpu(t):
        return None if t is None else t.detach().to("cpu", torch.float32)

    snap: dict = {"_schema": "adapter_streams_v1"}
    for i, adapter in enumerate(_iter_stream_modules(model)):
        fw = adapter._stream_fw
        snap[f"i{i}"] = {
            "h": _cpu(adapter._stream_h),
            "fw": None if fw is None else [_cpu(fw[0]), _cpu(fw[1])],
            "pos": int(getattr(adapter, "_stream_pos", 0)),
        }
    return snap


def restore_adapter_streams(model: nn.Module, snap: dict,
                            batch: Optional[int] = None) -> int:
    """Write a :func:`snapshot_adapter_streams` dict back onto the modules
    (adapters AND ARR mixers).

    Moves each tensor to the target module's own device/dtype (a snapshot is
    stored device-agnostic in CPU fp32; here it is cast ONCE to the live dtype).
    When ``batch`` is given, entries whose snapshot batch axis ``shape[0]`` !=
    ``batch`` are DROPPED and not counted — a B=1 serve snapshot restored into a
    B>1 forward would otherwise be silently zeroed by the module's own shape
    guard on the next forward, reading as "the bridge fired" (restored>0) when
    it did not. When ``batch`` is None the caller asserts B matches (e.g. the
    B=1 eval path); the forward guard still protects correctness, but the
    returned count is only trustworthy when ``batch`` is supplied. Returns the
    number of modules whose state was actually restored; enables streaming on
    each so the restored state is read on the next forward.
    """
    restored = 0
    for i, adapter in enumerate(_iter_stream_modules(model)):
        entry = snap.get(f"i{i}")
        if entry is None:
            continue
        if batch is not None:
            h, fw = entry.get("h"), entry.get("fw")
            cand_b = (h.shape[0] if h is not None
                      else fw[0].shape[0] if fw is not None else None)
            if cand_b is not None and cand_b != batch:
                continue                          # drop B-mismatched, don't count
        p = next(adapter.parameters(), None)
        device = p.device if p is not None else torch.device("cpu")
        dtype = p.dtype if p is not None else torch.float32

        def _to(t):
            return None if t is None else t.to(device=device, dtype=dtype)

        adapter.stream_enabled = True
        adapter._stream_h = _to(entry.get("h"))
        fw = entry.get("fw")
        adapter._stream_fw = None if fw is None else (_to(fw[0]), _to(fw[1]))
        if hasattr(adapter, "_stream_pos"):
            adapter._stream_pos = int(entry.get("pos", 0))
        restored += 1
    return restored


class adapter_streaming_paused:
    """Context manager: run auxiliary forwards (probes, encoders) without
    reading or writing the generation's streaming state."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._saved = []

    def __enter__(self):
        for adapter in _iter_stream_modules(self.model):
            self._saved.append((adapter, adapter.stream_enabled,
                                adapter._stream_h, adapter._stream_fw,
                                getattr(adapter, "_stream_pos", None)))
            adapter.stream_enabled = False
        return self

    def __exit__(self, *exc):
        for adapter, enabled, h, fw, pos in self._saved:
            adapter.stream_enabled = enabled
            adapter._stream_h = h
            adapter._stream_fw = fw
            if pos is not None and hasattr(adapter, "_stream_pos"):
                adapter._stream_pos = pos
        return False


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
    """Recreate the MT adapter layout recorded by train_llama_mt_adapter.py.

    Selects V2 vs V1 by the saved --adapter flag; V2 checkpoints carry
    fast_weight / selective_decay / sel_mode params that the V1 attach path
    would otherwise report as unexpected keys.
    """
    saved_args = checkpoint.get("args", {})
    if saved_args.get("no_mt", False):
        # 纯 LoRA 对照臂：checkpoint 无 mt_adapter key，不挂任何 MT adapter
        return []
    if saved_args.get("adapter") == "v2":
        from .mt_lnn_v2 import attach_mt_v2_adapters
        return attach_mt_v2_adapters(
            model,
            every=int(saved_args.get("mt_every", 4)),
            n_protofilaments=int(saved_args.get("mt_proto", 13)),
            d_proto=int(saved_args.get("v2_d_proto", 64)),
            n_time_scales=int(saved_args.get("mt_scales", 5)),
            proj_rank=int(saved_args.get("v2_rank", 128)),
            init_scale=float(saved_args.get("mt_init_scale", 1e-3)),
            dropout=float(saved_args.get("mt_dropout", 0.0)),
            selective_decay=bool(saved_args.get("v2_selective", False)),
            selective_decay_mode=str(saved_args.get("sel_mode", "mamba")),
            use_fast_weight=not bool(saved_args.get("v2_no_fw", False)),
            fast_weight_dim=int(saved_args.get("v2_fw_dim", 64)),
            fast_weight_heads=int(saved_args.get("v2_fw_heads", 1)),
        )
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
