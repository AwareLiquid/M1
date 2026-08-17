"""
model.py — MTLNNModel: full Microtubule-Enhanced Liquid Neural Network.

Dual-state inference cache:
  - Attention sub-layer: KV cache (past_kv = (K, V)) per layer
  - LNN sub-layer:       recurrent hidden state h_prev per layer

A `LayerCache` per layer is the tuple (kv, h_prev). The full cache is a list
of LayerCache, one per block. Pass `use_cache=True` to enable.
"""

import math
from typing import Callable, List, Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig
from .embedding import MTLNNEmbedding
from .mt_attention import MicrotubuleAttention
from .mt_lnn_layer import MTLNNLayer
from .global_coherence import GlobalCoherenceLayer
from .gwtb import GWTBLayer
from .utils import init_weights, init_mt_params


# Type aliases for clarity.
# LayerCache is now a 3-tuple: (attention KV, LNN recurrent state, per-block GWTB KV).
# The third slot is None when gwtb_per_block=False so the cache structure stays
# identical for both modes — code reads cache.layers[i][2] unconditionally.
KVCache    = Tuple[torch.Tensor, torch.Tensor]                          # (K, V)
LayerCache = Tuple[
    Optional[KVCache],          # 0: attention KV
    Optional[torch.Tensor],     # 1: MT-DL recurrent h_prev
    Optional[KVCache],          # 2: per-block GWTB KV (None if not per-block)
]


class ModelCacheStruct:
    """Full inference cache for incremental decoding."""
    def __init__(self, token_count: int = 0):
        self.layers: List[LayerCache] = []
        self.gwtb_kv: Optional[KVCache] = None
        self.coherence_kv: Optional[KVCache] = None
        self.token_count = token_count

    def recurrent_only(self) -> "ModelCacheStruct":
        """Return a constant-size cache that preserves only LNN h_prev states."""
        out = ModelCacheStruct(token_count=self.token_count)
        for layer_cache in self.layers:
            h_prev = layer_cache[1] if layer_cache is not None and len(layer_cache) > 1 else None
            out.layers.append((None, h_prev, None))
        return out

    def tensor_bytes(self) -> int:
        """Approximate bytes held by tensors inside this cache."""
        total = 0

        def add_tensor(t: Optional[torch.Tensor]) -> None:
            nonlocal total
            if t is not None:
                total += t.numel() * t.element_size()

        def add_kv(kv: Optional[KVCache]) -> None:
            if kv is not None:
                add_tensor(kv[0])
                add_tensor(kv[1])

        for layer_cache in self.layers:
            if layer_cache is None:
                continue
            add_kv(layer_cache[0])
            add_tensor(layer_cache[1])
            add_kv(layer_cache[2])
        add_kv(self.gwtb_kv)
        add_kv(self.coherence_kv)
        return total


class MTLNNBlock(nn.Module):
    """One transformer-style block with microtubule attention + LNN layer.

    If `config.gwtb_per_block` is True, the block additionally contains a
    GWTBLayer that runs after the LNN sub-layer — matching the paper §4
    architecture where every block ignites its own workspace.
    """

    def __init__(self, config: MTLNNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.has_gwtb = config.gwtb_per_block

        # Hybrid thinning: attention_layers=None keeps the historical
        # every-layer attention (bit-exact). A block outside the set builds NO
        # attention machinery at all -- no norm, no projections, no KV cache
        # entry -- rather than a bypassed module, so the parameter saving and
        # the KV saving are real, not cosmetic.
        att = getattr(config, "attention_layers", None)
        self.has_attn = att is None or layer_idx in att
        if self.has_attn:
            self.attn_norm = nn.LayerNorm(config.d_model)
            self.attn: MicrotubuleAttention   # set by parent after embedding init
        else:
            self.attn_norm = None
            self.attn = None
        self.lnn_norm = nn.LayerNorm(config.d_model)
        self.lnn = MTLNNLayer(config)

        # Latent recurrent depth ("thinking steps", M2 P0). The gate parameter
        # is created ONLY when the config enables iteration, so the default
        # (core_iterations=1) model keeps a byte-identical parameter set.
        # Runtime iteration count is an attribute (not read from config at
        # forward time) so MTLNNModel.set_core_iterations() can vary depth
        # per step — train with randomized depth, evaluate at any depth.
        self.core_iterations = int(getattr(config, "core_iterations", 1))
        if self.core_iterations > 1:
            self.core_iter_gate = nn.Parameter(
                torch.tensor(float(getattr(config, "core_iter_gate_init", 0.0)))
            )
        else:
            self.core_iter_gate = None

        if self.has_gwtb:
            self.gwtb_norm = nn.LayerNorm(config.d_model)
            self.gwtb = GWTBLayer(config)
        else:
            self.gwtb_norm = None
            self.gwtb = None

        # Top-down modulation (P1 closed-loop ②): a high-level goal/context vector
        # biases this block via a ZERO-INIT-GATED residual adapter, injected after
        # attention and before the LNN sub-layer. Built only when use_top_down is
        # set, so the default model has byte-identical params (zero regression).
        # RESEARCH — NOT WIRED (architecture audit 2026-07-12): this is a complete
        # RECEIVER with no TRANSMITTER — no module in the architecture produces the
        # top_down goal vector, train.py never passes top_down=, and the demos feed
        # torch.randn. Which signal should drive it (higher-layer state vs external
        # goal vs world model) is an undecided design question. ARCHIVE until a
        # concrete goal-source module and a goal-conditioned task exist.
        #     x = x + tanh(gate) · proj(LayerNorm(top_down))
        # * top_down_norm  -- the init/runtime insurance: bounds an arbitrary-scale
        #   goal so the residual can't blow up once the gate learns to open.
        # * top_down_proj  -- small-but-nonzero std (0.02) so the gate has a live
        #   gradient at init (a zero-init proj would freeze the gate forever).
        # * top_down_gate  -- init 0 → tanh(0)=0 → identity at init (bit-exact).
        self.has_top_down = getattr(config, "use_top_down", False)
        if self.has_top_down:
            self.top_down_norm = nn.LayerNorm(config.d_model)
            self.top_down_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            nn.init.normal_(self.top_down_proj.weight, std=0.02)
            self.top_down_gate = nn.Parameter(
                torch.tensor(float(getattr(config, "top_down_gate_init", 0.0)))
            )
        else:
            self.top_down_norm = None
            self.top_down_proj = None

    def forward(
        self,
        x: torch.Tensor,                                  # (B, T_new, d_model)
        layer_cache: Optional[LayerCache] = None,
        pad_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        use_lnn_recurrence: bool = True,
        top_down: Optional[torch.Tensor] = None,    # (B, d_model) or (B, T, d_model)
    ) -> Tuple[torch.Tensor, Optional[LayerCache]]:

        past_kv      = layer_cache[0] if layer_cache is not None else None
        h_prev       = (layer_cache[1] if (layer_cache is not None and use_lnn_recurrence)
                        else None)
        past_gwtb_kv = (layer_cache[2] if (layer_cache is not None and len(layer_cache) > 2)
                        else None)

        # Attention sub-layer (pre-norm) -- absent entirely in thinned layers,
        # whose cache slot stays None and whose timing comes from the LNN.
        if self.has_attn:
            attn_out, new_kv = self.attn(
                self.attn_norm(x),
                pad_mask=pad_mask,
                past_kv=past_kv,
                position_offset=position_offset,
                use_cache=use_cache,
                h_prev=h_prev,  # Pass h_prev for position-free timing signal
            )
            x = x + attn_out
        else:
            new_kv = None

        # Top-down modulation (P1 closed-loop ②): inject the goal/context bias
        # AFTER attention, BEFORE the LNN sub-layer, so it colours the liquid
        # core's temporal integration this step. Zero-gated at init (tanh(0)=0),
        # so this is a strict no-op until the gate learns to open. top_down may be
        # (B, d_model) -- broadcast over time -- or a per-step (B, T, d_model).
        if self.has_top_down and top_down is not None:
            td = top_down.unsqueeze(1) if top_down.dim() == 2 else top_down
            x = x + torch.tanh(self.top_down_gate) * self.top_down_proj(
                self.top_down_norm(td)
            )

        # LNN sub-layer (pre-norm).
        # use_lnn_recurrence drives BOTH (a) whether to pull h_prev from the
        # cache or use zeros, and (b) whether the resonance bank runs a real
        # parallel scan (True) or the legacy broadcast-h_prev parallel mode
        # (False). The two together preserve cache parity in both modes.
        #
        # Latent recurrent depth ("thinking steps"): when core_iterations > 1
        # the SAME MTLNNLayer re-reads the SAME normed input N times. Two
        # mechanisms compose per extra iteration:
        #   1. state threading (ungated — this IS the iteration): pass k
        #      starts from pass k-1's final state, i.e. "re-scan the sequence
        #      with updated working memory", letting information propagate one
        #      more compositional hop per pass;
        #   2. gated output feedback: pass k-1's output is added to the input
        #      through a zero-init tanh gate (learnable feedback strength).
        # position_offset is held CONSTANT across iterations so the GTP
        # periodic clock (t mod gtp_period) does not drift. NOTE: resonance
        # aux buffers (last_pred_error / last_lavi_mean) are overwritten each
        # pass — aux losses therefore see the FINAL iteration only. The cache
        # stores the final iteration's h_last, so generate()/streaming are
        # unchanged. N=1 is exactly the pre-existing single-call path.
        x_normed = self.lnn_norm(x)
        lnn_pad = pad_mask[:, -x.shape[1]:] if pad_mask is not None else None
        lnn_out, h_last = self.lnn(
            x_normed, h_prev,
            position_offset=position_offset,
            use_scan=use_lnn_recurrence,
            pad_mask=lnn_pad,
        )
        for _ in range(self.core_iterations - 1):
            x_iter = x_normed + torch.tanh(self.core_iter_gate) * lnn_out
            lnn_out, h_last = self.lnn(
                x_iter, h_last,
                position_offset=position_offset,
                use_scan=use_lnn_recurrence,
                pad_mask=lnn_pad,
            )
        x = x + lnn_out

        # Per-block GWTB (pre-norm, gated residual already inside GWTBLayer)
        new_gwtb_kv = None
        if self.has_gwtb:
            x, new_gwtb_kv = self.gwtb(
                self.gwtb_norm(x),
                past_kv=past_gwtb_kv,
                position_offset=position_offset,
                use_cache=use_cache,
                pad_mask=pad_mask,
            )

        new_cache: Optional[LayerCache] = (
            (new_kv, h_last, new_gwtb_kv) if use_cache else None
        )
        return x, new_cache


class MTLNNModel(nn.Module):
    """Full MT-LNN language model (~125M params)."""

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.config = config

        # Stack-level latent recurrence pass count (runtime attribute so
        # set_stack_iterations can vary it per step; no parameters involved).
        self.stack_iterations = int(getattr(config, "stack_iterations", 1))

        self.embedding = MTLNNEmbedding(config)

        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            block = MTLNNBlock(config, layer_idx=i)
            if block.has_attn:
                block.attn = MicrotubuleAttention(config, rope=self.embedding.rope)
            self.blocks.append(block)

        # Global Workspace Theory Bottleneck.
        # If gwtb_per_block=True the bottleneck lives inside every block (paper
        # §4 default). Otherwise it's applied once after the entire stack.
        # Mutually exclusive to avoid double-broadcasting the workspace.
        # use_gwtb=False (O 系列) skips the layer entirely — no O(T²) causal
        # mask allocation.
        self.gwtb_per_block = config.gwtb_per_block
        if getattr(config, "use_gwtb", True) and not config.gwtb_per_block:
            if getattr(config, "use_competitive_gwtb", False):
                from .gwtb import CompetitiveGWTBLayer
                self.gwtb = CompetitiveGWTBLayer(config)
            else:
                self.gwtb = GWTBLayer(config)
        else:
            self.gwtb = None

        # Orch-OR collapse layer (complementary to GWTB)
        self.coherence = GlobalCoherenceLayer(config)
        self.final_norm = nn.LayerNorm(config.d_model)

        # Global rhythm controller: reads per-layer LAVI buffers after the block
        # loop and applies a small learned correction before GWTB.
        # Active only when global_rhythm=True in config (default False).
        if getattr(config, "global_rhythm", False):
            from .rhythm import GlobalRhythmController
            self.global_rhythm = GlobalRhythmController(config.n_layers, config.d_model)
        else:
            self.global_rhythm = None

        # Predictive State Head / World Model (Phase C).
        # Runs on x_normed (after final_norm), predicts h_{t+1} from h_t.
        # Training adds L_wm to total loss; inference updates last_pred_error.
        # Active only when use_world_model=True in config (default False).
        if getattr(config, "use_world_model", False):
            from .world_model import PredictiveStateHead
            self.world_model_head = PredictiveStateHead(
                config.d_model,
                hidden_ratio=getattr(config, "world_model_hidden_ratio", 0.5),
                proj_ratio=getattr(config, "world_model_proj_ratio", 0.5),
                ema_decay=getattr(config, "world_model_ema_decay", 0.99),
                use_ema_target=getattr(config, "world_model_use_ema_target", True),
                warmup_steps=getattr(config, "world_model_warmup_steps", 1000),
            )
        else:
            self.world_model_head = None

        # Physics-informed world model (Hamiltonian head, 2026-07-14). A
        # conservation-biased alternative to the PredictiveStateHead: predicts the
        # next hidden latent through a symplectic phase-space bottleneck whose
        # potential is conditioned on the liquid core. Runs on x_normed like the
        # world model and contributes an aux loss. Independent of use_world_model
        # (both may be on). Active only when use_hamiltonian_world_model=True
        # (default False → not built → bit-identical forward).
        if getattr(config, "use_hamiltonian_world_model", False):
            from .hamiltonian_head import HamiltonianWorldModelHead
            self.hamiltonian_world_model = HamiltonianWorldModelHead(
                config.d_model,
                phase_dim=getattr(config, "hamiltonian_phase_dim", 32),
                hidden_dim=getattr(config, "hamiltonian_hidden", 64),
                dt=getattr(config, "hamiltonian_dt", 0.1),
                condition_on_context=getattr(config, "hamiltonian_condition_on_context", True),
                context_dim=getattr(config, "hamiltonian_context_dim", 32),
            )
        else:
            self.hamiltonian_world_model = None

        # P3.1 multi-source GWT: when the competitive workspace AND the predictive
        # world model are both active and gwtb_external_bids is set, the world
        # model submits its *expectation* of the current state as an extra bid in
        # the workspace competition. A residual, zero-GATED adapter maps the
        # predicted latent (proj_dim) back to d_model:
        #     world_bid = x + gate · adapter(predictor(online_proj(x)))
        # gate is init 0 → world_bid == x at init → competition (all bids == x,
        # uniform scores) is unchanged → bit-identical output at init (zero
        # regression). The adapter weights are init small-but-nonzero so the gate
        # has a live gradient and the world model can *learn* to bid usefully.
        self._world_gwtb_bid = (
            getattr(config, "gwtb_external_bids", False)
            and self.world_model_head is not None
            and getattr(self.gwtb, "accept_external_bids", False)
        )
        if self._world_gwtb_bid:
            self.world_bid_adapter = nn.Linear(
                self.world_model_head.proj_dim, config.d_model, bias=False
            )
            nn.init.normal_(self.world_bid_adapter.weight, std=0.02)
            self.world_bid_gate = nn.Parameter(
                torch.tensor(float(getattr(config, "gwtb_external_bid_gate_init", 0.0)))
            )

        # Top-down -> GWT docking entry (P1 closed-loop ②): additionally offer the
        # top-down goal as an external bid in the top-level global-workspace
        # competition, reusing the same zero-gated residual-bid machinery as the
        # world model. Active only when use_top_down AND top_down_to_gwtb are set
        # AND the top-level GWTB is competitive (accepts external bids). At init
        # gate=0 → bid == x → it competes as an equal competitor, identity-at-init
        # to within the same O(1e-4) softmax artifact as the world-model bid (the
        # competition's internal bids are not exact-identity after the global init
        # pass, so this is NOT strictly bit-exact — see test_multisource_gwt). This
        # path is default-OFF; the per-block top-down residual IS bit-exact.
        self._top_down_gwtb_bid = (
            getattr(config, "use_top_down", False)
            and getattr(config, "top_down_to_gwtb", False)
            and self.gwtb is not None
            and getattr(self.gwtb, "accept_external_bids", False)
        )
        if self._top_down_gwtb_bid:
            self.top_down_gwtb_norm = nn.LayerNorm(config.d_model)
            self.top_down_gwtb_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            nn.init.normal_(self.top_down_gwtb_proj.weight, std=0.02)
            self.top_down_gwtb_gate = nn.Parameter(
                torch.tensor(float(getattr(config, "top_down_gwtb_gate_init", 0.0)))
            )

        # Synaptic memory -> GWT docking entry (Gap 1, 2026-06-25): close the
        # "spatial position code -> Hebbian/associative synaptic memory -> global
        # workspace" chain by letting a content-addressed FastWeightMemory recall
        # stream bid for the workspace, jointly trained end-to-end with the main CE.
        # Reuses the SAME zero-gated residual-bid machinery as the world model (no
        # new bid mechanism) and the FastWeightMemory primitive built for the served
        # adapter (mt_lnn.llama_adapter.FastWeightMemory -- one associative-memory
        # implementation, shared across the M1 adapter and the O1 native model, NOT
        # duplicated). FastWeightMemory already projects its read back to d_model, so
        # the bid is simply:
        #     mem_bid = x + memory_bid_gate * fast_weight_recall(x)
        # gate init 0 -> bid == x at init -> competition unchanged (zero regression);
        # the recall projection is nonzero so the gate keeps a live gradient and the
        # workspace can LEARN how much to trust the memory. Active only when the
        # competitive workspace accepts external bids (same precondition as the
        # world-model / top-down bids); default OFF.
        self._memory_gwtb_bid = (
            getattr(config, "gwtb_memory_bid", False)
            and self.gwtb is not None
            and getattr(self.gwtb, "accept_external_bids", False)
        )
        if self._memory_gwtb_bid:
            from .llama_adapter import FastWeightMemory
            self.memory_fast_weight = FastWeightMemory(
                d_model=config.d_model,
                d_mem=int(getattr(config, "gwtb_memory_bid_dim", 64)),
                n_heads=int(getattr(config, "gwtb_memory_bid_heads", 1)),
                init_decay=float(getattr(config, "gwtb_memory_bid_decay", 0.95)),
            )
            self.memory_bid_gate = nn.Parameter(
                torch.tensor(float(getattr(config, "gwtb_memory_bid_gate_init", 0.0)))
            )

        # P3.2 graceful degradation: finiteness guards on auxiliary v2 module
        # contributions. _degradation_counts tracks how often each module was
        # skipped over the whole run (a monitoring "eye"); it is process state,
        # not a checkpoint buffer.
        self.use_graceful_degradation = bool(
            getattr(config, "use_graceful_degradation", True)
        )
        self._degradation_counts: Dict[str, int] = {}

        # Hebbian Regularizer (Phase D).
        # Collects per-block co-activation signals from MTLNNLayer._hebb_signal
        # and returns -α × mean as a training loss term.
        # Active only when use_hebbian=True in config (default False).
        if getattr(config, "use_hebbian", False):
            from .plasticity import HebbianRegularizer
            self.hebbian_reg = HebbianRegularizer(
                base_lr=getattr(config, "hebbian_lr", 1e-4),
                lavi_gate=getattr(config, "hebbian_lavi_gate", True),
            )
        else:
            self.hebbian_reg = None

        # Hebbian REFACTOR (mechanism A, 2026-06-16). Fully isolated rebuild
        # gated by use_hebbian_refactor (default False -> not built -> no params,
        # no forward/loss op -> bit-identical to the legacy model). See
        # mt_lnn/hebbian_plasticity.py. Independent of the legacy hebbian_reg.
        if getattr(config, "use_hebbian_refactor", False):
            from .hebbian_plasticity import HebbianPlasticity
            self.hebbian_plasticity = HebbianPlasticity(config)
        else:
            self.hebbian_plasticity = None

        # LM head (no bias, optionally weight-tied)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.token_embed.weight

        # Auxiliary direct-extraction head. It maps the final prefix state into
        # fixed target slots, so extraction tasks can skip autoregressive decode.
        self.target_queries = nn.Parameter(
            torch.zeros(config.direct_target_max_len, config.d_model)
        )
        self.target_norm = nn.LayerNorm(config.d_model)
        self.target_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Phase 2 & 3: Causal Chain and Self-Monitor Heads.
        # RESEARCH — NOT WIRED (architecture audit 2026-07-12). Both emit
        # vocab-space logits that NO training loss consumes (grep: zero readers in
        # train.py / this forward's loss block) — enabling them only bloats the
        # checkpoint with dead vocab-sized params. The blocker is a definable
        # LABEL, not the wiring: the causal head needs a counterfactual/cause-
        # effect dataset the repo does not contain, and "what a model should say
        # about its own internal state" has no ground truth at all, so no honest
        # CE target is writable for the self-monitor head. Roadmap-only (see
        # BRAIN_INSPIRED_ROADMAP.md Phases 2–3). Do NOT enable in a trained config.
        if getattr(config, "use_causal_head", False):
            self.causal_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if getattr(config, "use_self_monitor_head", False):
            self.self_monitor_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # MTP lookahead heads (2026-06-26).
        # K shallow linear heads predict tokens t+k for k=1..K from the final
        # normed hidden state. Weight-untied from lm_head so each head can
        # specialise at a different prediction horizon. At inference they emit
        # draft logits for speculative decoding; during training they add an
        # auxiliary CE loss (mtp_loss_weight * mean_over_k).
        if getattr(config, "use_mtp_heads", False):
            K = config.mtp_lookahead
            self.mtp_heads = nn.ModuleList(
                [nn.Linear(config.d_model, config.vocab_size, bias=False)
                 for _ in range(K)]
            )
        else:
            self.mtp_heads = None

        # Weight initialisation
        self.apply(lambda m: init_weights(m, config))
        nn.init.normal_(self.target_queries, mean=0.0, std=0.02)
        init_mt_params(self, config)
        # The global init_weights pass above re-initialises EVERY nn.Linear to
        # N(0, 0.02) — including CompetitiveGWTBLayer's zero-initialised bid /
        # score projections, silently breaking its 'bid ≡ x, uniform competition
        # at init' invariant. Restore those zero-inits after the global pass.
        if self.gwtb is not None and hasattr(self.gwtb, "reset_zero_init"):
            self.gwtb.reset_zero_init()

    # ------------------------------------------------------------------
    # Graceful degradation (P3.2)
    # ------------------------------------------------------------------

    def _aux_or_skip(self, name: str, value: "Optional[torch.Tensor]"):
        """
        Finiteness guard for an AUXILIARY v2 contribution.

        Returns ``value`` unchanged when it is None, when graceful degradation is
        disabled, or when it is finite. Otherwise returns None and records a
        degradation event under ``name`` — the caller then simply skips adding
        this term, so a transient NaN/Inf in one bio-inspired module does not
        propagate into the main loss / parameters and abort a long run.
        """
        if value is None or not self.use_graceful_degradation:
            return value
        if torch.isfinite(value).all():
            return value
        self._degradation_counts[name] = self._degradation_counts.get(name, 0) + 1
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,             # (B, T_new)
        cache: Optional[ModelCacheStruct] = None,
        pad_mask: Optional[torch.Tensor] = None,              # (B, T_total) bool
        labels: Optional[torch.Tensor] = None,                # (B, T_new) for causal LM loss
        direct_target_labels: Optional[torch.Tensor] = None,  # (B, L) for direct extraction
        target_len: Optional[int] = None,
        return_target_logits: bool = False,
        use_cache: bool = False,
        position_offset: Optional[int] = None,
        use_lnn_recurrence: bool = True,
        inputs_embeds: Optional[torch.Tensor] = None,         # (B, T_new, d_model)
        top_down: Optional[torch.Tensor] = None,              # (B, d_model) or (B, T_new, d_model)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          inputs_embeds: pre-computed (B, T_new, d_model) embeddings used INSTEAD
            of looking up input_ids. This is the multimodal injection point: fuse
            text token embeddings (model.embed_tokens) with projected image/audio
            features (see mt_lnn.multimodal) along the sequence dim and pass the
            result here. Exactly one of input_ids / inputs_embeds must be given.
          use_cache: build/extend the KV+coherence cache for incremental decoding.
          use_lnn_recurrence: if True (default for inference), thread h_prev across
            steps so the LNN behaves as a true RNN. Set False to match the
            training-time parallel mode (h_prev=0 each step) — this is what makes
            cached vs full-forward outputs bit-exact equal.
          position_offset: if None, infer from cache (length of cached K) or 0.
          top_down: optional high-level goal/context (B, d_model) broadcast over
            time, or per-step (B, T_new, d_model). Requires config.use_top_down;
            each block applies a zero-init-gated residual bias (bit-exact at init).
            If config.top_down_to_gwtb is also set, the goal is additionally
            offered as an external bid in the top-level workspace competition.

        Returns dict with keys:
          - logits:   (B, T_new, vocab_size)
          - target_logits: (B, L, vocab_size) if direct extraction is requested
          - loss:     scalar (only if labels provided)
          - target_loss: scalar (only if direct_target_labels provided)
          - cache:    new ModelCacheStruct (only if use_cache=True)
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "provide exactly one of input_ids or inputs_embeds"
            )

        # Infer absolute position offset. token_count is authoritative: it is
        # maintained for every architecture, while the historical fallback --
        # reading T_past off layer 0's K tensor -- returns 0 silently whenever
        # layer 0 has no attention (attention_layers thinning), skewing RoPE
        # and the GTP clock on every subsequent decode step.
        if position_offset is None:
            if cache is not None and getattr(cache, "token_count", 0):
                position_offset = int(cache.token_count)
            elif (cache is not None and len(cache.layers) > 0
                  and cache.layers[0] is not None and cache.layers[0][0] is not None):
                position_offset = cache.layers[0][0][0].shape[2]   # T_past from first layer's K
            else:
                position_offset = 0

        if inputs_embeds is not None:
            if inputs_embeds.shape[-1] != self.config.d_model:
                raise ValueError(
                    f"inputs_embeds last dim {inputs_embeds.shape[-1]} != "
                    f"d_model {self.config.d_model}"
                )
            x = self.embedding.dropout(inputs_embeds)         # (B, T_new, d_model)
        else:
            x = self.embedding(input_ids)                     # (B, T_new, d_model)
        T_new = x.shape[1]

        # Phase C → LAVI linkage: distribute the PREVIOUS step's world-model
        # prediction error to all blocks before the loop so LAVIEstimator can
        # use it as a supplementary transience signal.
        # Uses last_pred_error (a buffer holding last forward's value — causal).
        # As of v2.1 this buffer is the NORMALISED surprise ∈ [0,1] (1-cos)/2,
        # so the LAVI correction wm_correction = tanh(scale)·pred_error stays
        # bounded regardless of representation magnitude.
        if self.world_model_head is not None:
            _wm_err = self.world_model_head.last_pred_error.item()
            # P3.2: a non-finite surprise must not corrupt the LAVI rhythm gate.
            if self.use_graceful_degradation and not math.isfinite(_wm_err):
                self._degradation_counts["lavi_surprise"] = (
                    self._degradation_counts.get("lavi_surprise", 0) + 1
                )
                _wm_err = 0.0
            for _blk in self.blocks:
                _blk.lnn._last_wm_pred_error = _wm_err

        new_cache = ModelCacheStruct(
            token_count=position_offset + T_new
        ) if use_cache else None
        _gate_period = getattr(self.config, "scale_gate_period", 1)
        # Stack-level latent recurrence (M2 P0-C′): re-apply the WHOLE block
        # stack (attention + LNN, weight-tied) stack_iterations times. Depth
        # for compositional reasoning lives in attention (P0 rounds 1–5), so
        # this — unlike core_iterations — actually multiplies the number of
        # attention applications. Single-pass (default 1) is the exact
        # pre-existing path. Caching is a single-pass concept; guard it.
        _stack_iters = self.stack_iterations
        if _stack_iters > 1 and use_cache:
            raise RuntimeError(
                "stack_iterations > 1 does not support use_cache "
                "(weight-tied multi-pass has no single KV timeline)"
            )
        for _pass in range(_stack_iters):
            _shared_active_idx = None
            for i, block in enumerate(self.blocks):
                layer_cache = (cache.layers[i] if (cache is not None and i < len(cache.layers)) else None)
                # Cross-layer scale-gate sharing: leader (i % period == 0) computes the
                # τ-scale index; followers reuse it without re-running kappa_gate.
                _is_leader = (_gate_period <= 1) or (i % _gate_period == 0)
                block.lnn.resonance._forced_active_idx = (
                    None if _is_leader else _shared_active_idx
                )
                x, new_layer_cache = block(
                    x,
                    layer_cache=layer_cache,
                    pad_mask=pad_mask,
                    position_offset=position_offset,
                    use_cache=use_cache,
                    use_lnn_recurrence=use_lnn_recurrence,
                    top_down=top_down,
                )
                if _is_leader and _gate_period > 1:
                    _shared_active_idx = block.lnn.resonance._last_computed_active_idx
                if use_cache:
                    new_cache.layers.append(new_layer_cache)

        # Global rhythm correction: aggregate per-layer LAVI means, apply residual.
        # Starts as identity (GlobalRhythmController.scale init = 0).
        if self.global_rhythm is not None:
            layer_lavi_means = torch.stack(
                [b.lnn.resonance.last_lavi_mean for b in self.blocks]
            )   # (n_layers,)
            x, _global_lavi = self.global_rhythm(layer_lavi_means, x)

        # Top-level GWTB (only if gwtb_per_block=False).
        if self.gwtb is not None:
            gwtb_past = cache.gwtb_kv if cache is not None else None
            # P3.1: build the world model's expectation bid (residual, zero-gated
            # at init → no effect until the gate learns to open).
            external_bids = None
            _bids: list = []
            if getattr(self, "_world_gwtb_bid", False):
                z_pred = self.world_model_head.predictor(
                    self.world_model_head.online_proj(x)
                )                                                 # (B, T, proj_dim)
                world_bid = x + self.world_bid_gate * self.world_bid_adapter(z_pred)
                # P3.2: a non-finite world bid must not corrupt the competition —
                # fall back to bidding the raw state x (always a valid competitor).
                if self.use_graceful_degradation and not torch.isfinite(world_bid).all():
                    self._degradation_counts["world_bid"] = (
                        self._degradation_counts.get("world_bid", 0) + 1
                    )
                    world_bid = x
                _bids.append(world_bid)
            # Top-down -> GWT docking entry: offer the goal as a workspace bid,
            # zero-gated at init (bid == x → competition unchanged → bit-exact).
            if getattr(self, "_top_down_gwtb_bid", False) and top_down is not None:
                td = top_down.unsqueeze(1) if top_down.dim() == 2 else top_down
                td_bid = x + torch.tanh(self.top_down_gwtb_gate) * self.top_down_gwtb_proj(
                    self.top_down_gwtb_norm(td)
                )
                if self.use_graceful_degradation and not torch.isfinite(td_bid).all():
                    self._degradation_counts["top_down_bid"] = (
                        self._degradation_counts.get("top_down_bid", 0) + 1
                    )
                    td_bid = x
                _bids.append(td_bid)
            # Synaptic memory bid (Gap 1): the content-addressed FastWeightMemory
            # recalls the value associated with the current state and offers it to
            # the workspace competition. Zero-gated at init (recall * 0 -> bid == x),
            # so it joins as an equal competitor without perturbing the init output.
            if getattr(self, "_memory_gwtb_bid", False):
                mem_recall, _ = self.memory_fast_weight(x)
                mem_bid = x + self.memory_bid_gate * mem_recall
                if self.use_graceful_degradation and not torch.isfinite(mem_bid).all():
                    self._degradation_counts["memory_bid"] = (
                        self._degradation_counts.get("memory_bid", 0) + 1
                    )
                    mem_bid = x
                _bids.append(mem_bid)
            if _bids:
                external_bids = _bids
            if external_bids is not None:
                # Only the CompetitiveGWTBLayer accepts external bids; the base
                # GWTBLayer keeps its original signature (zero regression).
                x, gwtb_new_kv = self.gwtb(
                    x, past_kv=gwtb_past, position_offset=position_offset,
                    use_cache=use_cache, external_bids=external_bids,
                    pad_mask=pad_mask,
                )
            else:
                x, gwtb_new_kv = self.gwtb(
                    x, past_kv=gwtb_past, position_offset=position_offset,
                    use_cache=use_cache, pad_mask=pad_mask,
                )
            if use_cache:
                new_cache.gwtb_kv = gwtb_new_kv

        coh_past = cache.coherence_kv if cache is not None else None
        x, coh_new_kv = self.coherence(
            x, past_kv=coh_past, position_offset=position_offset, use_cache=use_cache,
            pad_mask=pad_mask,
        )
        if use_cache:
            new_cache.coherence_kv = coh_new_kv

        x = self.final_norm(x)

        # Predictive State Head (Phase C): runs on normed x.
        # During training (compute_loss=True, T>1): accumulates L_wm and updates
        # the last_pred_error surprise buffer.
        # During inference (compute_loss=False): the buffer is NOT updated — the
        # LAVI linkage reads the last TRAINING-time surprise (persisted in the
        # checkpoint now that the buffer is persistent). Live inference-time
        # surprise would need next-token targets, which don't exist mid-decode.
        _world_model_loss: Optional[torch.Tensor] = None
        if self.world_model_head is not None:
            _, _world_model_loss = self.world_model_head(
                x, compute_loss=(self.training and T_new > 1)
            )

        # Physics-informed world model (Hamiltonian head): self-supervised
        # next-latent prediction through a symplectic phase-space bottleneck.
        _hamiltonian_loss: Optional[torch.Tensor] = None
        if self.hamiltonian_world_model is not None:
            _, _hamiltonian_loss = self.hamiltonian_world_model(
                x, compute_loss=(self.training and T_new > 1)
            )

        logits = self.lm_head(x)                              # (B, T_new, vocab_size)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if use_cache:
            result["cache"] = new_cache

        # MTP lookahead heads. The K draft-logit stack has TWO distinct consumers,
        # decoupled 2026-07-12 (previously both were gated behind
        # enable_speculative_decoding, which made the training regularizer
        # unreachable from any normal recipe — the mis-gating bug the audit found):
        #   (1) TRAINING aux loss — the DeepSeek-V3-style multi-token-prediction
        #       regularizer folded in below. It needs the draft stack on any
        #       training forward that has labels and a positive mtp_loss_weight,
        #       INDEPENDENT of the speculative flag. This is the wired, evidence-
        #       track use-case: enable it with use_mtp_heads=True alone.
        #   (2) INFERENCE speculative decoding — surfaces result["mtp_draft_logits"]
        #       for a draft→verify decode loop. That consumer is NOT built
        #       (RESEARCH — not wired; needs a verify loop + O(1)-cache rollback),
        #       so the draft is only exposed when enable_speculative_decoding is
        #       explicitly set.
        # use_mtp_heads=False (default) → mtp_heads is None → this block is skipped
        # entirely → forward is bit-identical to the pre-MTP model (zero regression).
        _mtp_draft: Optional[torch.Tensor] = None
        if self.mtp_heads is not None:
            _spec = getattr(self.config, "enable_speculative_decoding", False)
            _need_for_loss = (
                self.training and labels is not None
                and self.config.mtp_loss_weight > 0.0
            )
            if _spec or _need_for_loss:
                # Stack K draft logits: (B, T_new, K, vocab_size)
                drafts = [head(x) for head in self.mtp_heads]
                _mtp_draft = torch.stack(drafts, dim=2)
                if _spec:
                    result["mtp_draft_logits"] = _mtp_draft

        if direct_target_labels is not None:
            target_len = direct_target_labels.shape[1]
            return_target_logits = True

        if return_target_logits:
            if target_len is None:
                target_len = self.config.direct_target_max_len
            if target_len < 1 or target_len > self.config.direct_target_max_len:
                raise ValueError(
                    f"target_len must be in [1, {self.config.direct_target_max_len}], "
                    f"got {target_len}"
                )

            global_state = x[:, -1:, :]                        # (B, 1, d_model)
            queries = self.target_queries[:target_len].unsqueeze(0)
            target_hidden = self.target_norm(global_state + queries)
            target_logits = self.target_head(target_hidden)     # (B, L, vocab_size)
            result["target_logits"] = target_logits

            if direct_target_labels is not None:
                target_loss = F.cross_entropy(
                    target_logits.reshape(-1, self.config.vocab_size),
                    direct_target_labels.reshape(-1),
                    ignore_index=-100,
                )
                result["target_loss"] = target_loss

        # Phase 2 & 3 outputs
        if getattr(self.config, "use_causal_head", False):
            result["causal_logits"] = self.causal_head(x[:, -1:, :])
        if getattr(self.config, "use_self_monitor_head", False):
            result["self_monitor_logits"] = self.self_monitor_head(x[:, -1:, :])

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Expose the PURE next-token cross-entropy separately, BEFORE any
            # auxiliary (predictive-coding / world-model / Hebbian / ortho) term
            # is folded into `loss`. This is the honest language-modelling metric:
            # exp(lm_loss) is the true PPL. `loss` below is the training objective
            # (CE + aux) and must NOT be used to report perplexity.
            result["lm_loss"] = loss.detach()

            # P3.2: every AUXILIARY term below is finiteness-guarded via
            # _aux_or_skip — a NaN/Inf in one bio-inspired module is dropped for
            # this step (and counted) rather than poisoning `loss`. The primary
            # cross-entropy above is never guarded (a NaN there is a real fault).
            if getattr(self.config, "use_predictive_coding", False):
                pred_error_sum = sum(b.lnn.resonance.last_pred_error for b in self.blocks)
                result["pred_loss"] = pred_error_sum
                pred_ok = self._aux_or_skip("pred_loss", pred_error_sum)
                if pred_ok is not None:
                    loss = loss + self.config.predictive_loss_weight * pred_ok

            # Phase C: world model prediction loss
            if _world_model_loss is not None:
                wm_ok = self._aux_or_skip("world_model_loss", _world_model_loss)
                if wm_ok is not None:
                    wm_weight = getattr(self.config, "world_model_loss_weight", 0.01)
                    loss = loss + wm_weight * wm_ok
                    result["world_model_loss"] = wm_ok.detach()

            # Physics-informed world model (Hamiltonian head) aux loss.
            if _hamiltonian_loss is not None:
                ham_ok = self._aux_or_skip("hamiltonian_loss", _hamiltonian_loss)
                if ham_ok is not None:
                    ham_weight = getattr(self.config, "hamiltonian_loss_weight", 0.01)
                    loss = loss + ham_weight * ham_ok
                    result["hamiltonian_loss"] = ham_ok.detach()

            # Phase D: Hebbian co-activation loss (training only)
            if self.hebbian_reg is not None and self.training:
                hebb_loss = self._aux_or_skip(
                    "hebbian_loss", self.hebbian_reg.compute_loss(self)
                )
                if hebb_loss is not None:
                    loss = loss + hebb_loss
                    result["hebbian_loss"] = hebb_loss.detach()

            # Phase D' (refactor): isolated loss-term Hebbian (training only).
            # Gated by use_hebbian_refactor. compute_loss() returns None in
            # Stage 0, so this branch is a verified no-op until Stage 2.
            if self.hebbian_plasticity is not None and self.training:
                hebb_ref = self._aux_or_skip(
                    "hebbian_refactor_loss", self.hebbian_plasticity.compute_loss(self)
                )
                if hebb_ref is not None:
                    loss = loss + hebb_ref
                    result["hebbian_refactor_loss"] = hebb_ref.detach()

            # Phase A: orthogonality penalty for CompetitiveGWTBLayer (training only)
            # Pushes bid projectors apart in weight space (DeepSeekMoE-style).
            from .gwtb import CompetitiveGWTBLayer
            if (self.gwtb is not None
                and isinstance(self.gwtb, CompetitiveGWTBLayer)
                and self.gwtb._last_ortho_penalty is not None):
                ortho = self._aux_or_skip("ortho_penalty", self.gwtb._last_ortho_penalty)
                if ortho is not None:
                    loss = loss + self.gwtb.ortho_penalty_weight * ortho
                    result["ortho_penalty"] = ortho.detach()

            # MTP auxiliary loss: CE at each lookahead step k=1..K.
            # DeepSeek-V3 convention: the MAIN head at position t owns t+1, so
            # MTP head k predicts token t+1+k. (The old target labels[t+k] made
            # head k=1 an exact duplicate of the main CE — zero lookahead value
            # for drafting and a wasted 1/K of the MTP parameters/loss.)
            # Labels come from the input sequence itself (unsupervised), so no
            # extra annotations required. Only computed when T_new > K + 1.
            if (_mtp_draft is not None
                and self.config.mtp_loss_weight > 0.0
                and T_new > self.config.mtp_lookahead + 1):
                K = self.config.mtp_lookahead
                mtp_losses = []
                for k in range(1, K + 1):
                    # Predict position t+1+k from position t: valid for t in [0, T-k-2]
                    pred = _mtp_draft[:, :T_new - k - 1, k - 1, :]  # (B, T-k-1, V)
                    tgt = labels[:, k + 1:T_new]                     # (B, T-k-1)
                    if tgt.numel() == 0:
                        continue
                    mtp_k = F.cross_entropy(
                        pred.reshape(-1, self.config.vocab_size),
                        tgt.reshape(-1),
                        ignore_index=-100,
                    )
                    mtp_losses.append(mtp_k)
                if mtp_losses:
                    mtp_loss = torch.stack(mtp_losses).mean()
                    mtp_ok = self._aux_or_skip("mtp_loss", mtp_loss)
                    if mtp_ok is not None:
                        loss = loss + self.config.mtp_loss_weight * mtp_ok
                        result["mtp_loss"] = mtp_ok.detach()

            result["loss"] = loss

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embedding.token_embed.weight.numel()
        return total

    @classmethod
    def from_config(cls, config: MTLNNConfig) -> "MTLNNModel":
        return cls(config)

    def set_core_iterations(self, n: int) -> None:
        """Set the latent-recurrent-depth ("thinking steps") of every block.

        Used by the P0 depth study (docs/ROADMAP_M2.md): train with a
        randomized per-step depth, then evaluate the SAME weights at any
        depth. Requires the model to have been BUILT with
        config.core_iterations > 1 when n > 1 (the feedback gate parameter is
        only created at construction time); raises otherwise rather than
        silently running gateless.
        """
        n = int(n)
        if n < 1:
            raise ValueError(f"core_iterations must be >= 1; got {n}")
        for block in self.blocks:
            if n > 1 and block.core_iter_gate is None:
                raise RuntimeError(
                    "Model was built with core_iterations=1 (no feedback gate). "
                    "Construct with MTLNNConfig(core_iterations>1) to enable "
                    "thinking-step iteration."
                )
            block.core_iterations = n

    def set_stack_iterations(self, n: int) -> None:
        """Set the stack-level pass count (whole block stack, weight-tied).
        Unlike core iterations, no parameters are involved, so any depth is
        valid on any model. use_cache is rejected at forward time when n>1."""
        n = int(n)
        if n < 1:
            raise ValueError(f"stack_iterations must be >= 1; got {n}")
        self.stack_iterations = n

    def set_workspace_iterations(self, n: int) -> None:
        """J-Space J1: set the workspace reverberation pass count on every
        GWTB instance (top-level and/or per-block). No parameters involved —
        valid on any model at any depth."""
        n = int(n)
        if n < 1:
            raise ValueError(f"workspace_iterations must be >= 1; got {n}")
        targets = []
        if getattr(self, "gwtb", None) is not None:
            targets.append(self.gwtb)
        for block in self.blocks:
            if getattr(block, "gwtb", None) is not None:
                targets.append(block.gwtb)
        if not targets:
            raise RuntimeError("model has no GWTB layer to set")
        for g in targets:
            g.workspace_iterations = n

    def get_mt_diagnostics(self) -> Dict[str, float]:
        """Return a dict of MT health metrics for monitoring during training."""
        diag: Dict[str, float] = {}
        tau_vals, gamma_vals, pol_vals = [], [], []
        lat_norms, rmc_gates = [], []
        scale_gate_means, active_scale_ratios, nonzero_scale_ratios = [], [], []
        sparse_scale_ratios, sparse_selected = [], []

        for block in self.blocks:
            # τ tensor for this block's resonance bank: shape (P, S)
            tau = (F.softplus(block.lnn.resonance.log_tau)
                   + self.config.tau_min).clamp(self.config.tau_min,
                                                self.config.tau_max)
            tau_vals.extend(tau.detach().flatten().cpu().tolist())

            gamma_vals.append(block.lnn.gtp_gamma.item())
            if block.attn is not None:
                pol_vals.extend(block.attn.polarity_direction.detach().cpu().tolist())
            W = block.lnn.lateral.W_lat
            off_diag = W - torch.eye(W.shape[0], device=W.device)
            lat_norms.append(off_diag.norm().item())
            rmc_gates.append(torch.sigmoid(block.lnn.lateral.rmc_gate).item())
            resonance = block.lnn.resonance
            if hasattr(resonance, "last_scale_gate_mean"):
                scale_gate_means.append(resonance.last_scale_gate_mean.detach().cpu())
                active_scale_ratios.append(float(resonance.last_active_scale_ratio.item()))
                nonzero_scale_ratios.append(float(resonance.last_nonzero_scale_ratio.item()))
                sparse_scale_ratios.append(float(resonance.last_sparse_scale_ratio.item()))
                sparse_selected.append(resonance.last_sparse_selected_scales.detach().cpu())

        if tau_vals:
            t = torch.tensor(tau_vals)
            diag["tau_mean"] = t.mean().item()
            diag["tau_std"] = t.std().item()
            diag["tau_min"] = t.min().item()
            diag["tau_max"] = t.max().item()
        if gamma_vals:
            diag["gamma_mean"] = sum(gamma_vals) / len(gamma_vals)
        if pol_vals:
            p = torch.tensor(pol_vals)
            diag["polarity_mean"] = p.mean().item()
            diag["polarity_std"] = p.std().item()
        if lat_norms:
            diag["lat_coupling_mean_off_diag_norm"] = sum(lat_norms) / len(lat_norms)
        if rmc_gates:
            diag["rmc_gate_mean"] = sum(rmc_gates) / len(rmc_gates)
        if scale_gate_means:
            gate_mean = torch.stack(scale_gate_means).mean(dim=0)
            diag["scale_gate_mean"] = gate_mean.mean().item()
            diag["scale_gate_active_ratio"] = sum(active_scale_ratios) / len(active_scale_ratios)
            diag["scale_gate_nonzero_ratio"] = sum(nonzero_scale_ratios) / len(nonzero_scale_ratios)
            diag["sparse_resonance_scale_ratio"] = sum(sparse_scale_ratios) / len(sparse_scale_ratios)
            sparse_mean = torch.stack(sparse_selected).mean(dim=0)
            for idx, value in enumerate(gate_mean.tolist()):
                diag[f"scale_gate_s{idx}_mean"] = float(value)
            for idx, value in enumerate(sparse_mean.tolist()):
                diag[f"sparse_resonance_s{idx}_selected"] = float(value)

        # Rhythm diagnostics (only populated when use_rhythm=True)
        lavi_vals = [b.lnn.resonance.last_lavi_mean.item() for b in self.blocks
                     if getattr(b.lnn, "use_rhythm", False)]
        if lavi_vals:
            diag["lavi_mean"] = sum(lavi_vals) / len(lavi_vals)
            diag["lavi_min"]  = min(lavi_vals)
            diag["lavi_max"]  = max(lavi_vals)
        rhythm_scales = []
        for b in self.blocks:
            rs = getattr(b.lnn.resonance, "rhythm_scale", None)
            if rs is not None:
                rhythm_scales.append(torch.tanh(rs).item())
        if rhythm_scales:
            diag["rhythm_scale_mean"] = sum(rhythm_scales) / len(rhythm_scales)
        if self.global_rhythm is not None:
            diag["global_rhythm_scale"] = torch.tanh(self.global_rhythm.scale).item()

        # Phase C: world model diagnostics
        #   world_model_pred_error      : normalised surprise ∈ [0,1] (LAVI input)
        #   world_model_pred_error_raw  : raw latent MSE magnitude
        if self.world_model_head is not None:
            diag["world_model_pred_error"] = self.world_model_head.last_pred_error.item()
            diag["world_model_pred_error_raw"] = self.world_model_head.last_pred_error_raw.item()

        # Phase D: Hebbian diagnostics
        if self.hebbian_reg is not None:
            hebb_sigs = [
                b.lnn._hebb_signal.item()
                for b in self.blocks
                if getattr(b.lnn, "_hebb_signal", None) is not None
            ]
            if hebb_sigs:
                diag["hebbian_signal_mean"] = sum(hebb_sigs) / len(hebb_sigs)
            diag["hebbian_lavi_temperature"] = self.hebbian_reg.lavi_temperature.item()

        diag["coherence_scale"] = self.coherence.coherence_scale.item()
        diag["collapse_threshold"] = self.coherence.collapse_threshold.item()
        diag["collapse_gate_last"] = self.coherence.last_gate.item()

        if self.gwtb is not None:
            # Single top-level GWTB (standard or competitive)
            diag["gwtb_broadcast_gate"] = self.gwtb.broadcast_gate.item()
            diag["gwtb_d_gw"] = float(self.gwtb.d_gw)
            # Dynamic workspace bandwidth diagnostics (only when enabled): the
            # effective active bandwidth is the fraction of bottleneck channels
            # that ignited on the last forward — the "fixed d_gw" upper bound is
            # gwtb_d_gw, and this reports how much of it is actually in use.
            if getattr(self.gwtb, "dynamic_bandwidth", False):
                diag["gwtb_active_bandwidth"] = self.gwtb.last_active_bandwidth.item()
                diag["gwtb_bandwidth_gate_mean"] = self.gwtb.last_bandwidth_gate_mean.item()
            # Competitive GWTB diagnostics
            from .gwtb import CompetitiveGWTBLayer
            if isinstance(self.gwtb, CompetitiveGWTBLayer):
                diag["gwtb_competition_entropy"] = self.gwtb.last_competition_entropy.item()
                for k, w in enumerate(self.gwtb.last_winner_weights.tolist()):
                    diag[f"gwtb_bid{k}_weight"] = w
                # P3.1: external (multi-source) bid mass + the world-model bid gate.
                diag["gwtb_external_bid_weight"] = self.gwtb.last_external_weight.item()
                if getattr(self, "_world_gwtb_bid", False):
                    diag["gwtb_world_bid_gate"] = self.world_bid_gate.item()
                if getattr(self, "_memory_gwtb_bid", False):
                    diag["gwtb_memory_bid_gate"] = self.memory_bid_gate.item()
        else:
            # Per-block GWTB — report mean / spread across blocks
            gates = [b.gwtb.broadcast_gate.item() for b in self.blocks if b.has_gwtb]
            if gates:
                gates_t = torch.tensor(gates)
                diag["gwtb_broadcast_gate_mean"] = gates_t.mean().item()
                diag["gwtb_broadcast_gate_std"]  = gates_t.std().item()
                diag["gwtb_d_gw"] = float(self.blocks[0].gwtb.d_gw)

        # P3.2: graceful-degradation event counters (cumulative over the run).
        # A nonzero value here during a long pre-train run flags a misbehaving
        # module without crashing it — an early-warning "eye".
        if getattr(self, "_degradation_counts", None):
            diag["degradation_events_total"] = float(
                sum(self._degradation_counts.values())
            )
            for _name, _cnt in self._degradation_counts.items():
                diag[f"degradation_{_name}"] = float(_cnt)
        return diag

    def get_mt_histograms(self) -> Dict[str, torch.Tensor]:
        """Return raw tensors for W&B histogram logging."""
        taus, gammas, pols = [], [], []
        for block in self.blocks:
            tau = F.softplus(block.lnn.resonance.log_tau).detach() + self.config.tau_min
            taus.append(tau.flatten())
            gammas.append(block.lnn.gtp_gamma.detach().reshape(-1))
            if block.attn is not None:
                pols.append(block.attn.polarity_direction.detach())
        out = {
            "tau": torch.cat(taus).cpu().float(),
            "gamma": torch.cat(gammas).cpu().float(),
        }
        if pols:
            out["polarity"] = torch.cat(pols).cpu().float()
        return out

    # ------------------------------------------------------------------
    # Persistent memory (SQLite-backed cross-session h_prev storage)
    # ------------------------------------------------------------------

    def save_state(
        self,
        session_id: str,
        cache: "ModelCacheStruct",
        token_count: int = 0,
        db_path: str = ".mt_lnn_state.db",
    ) -> None:
        """Persist the recurrent h_prev states in *cache* to SQLite.

        Convenience wrapper around :class:`~mt_lnn.memory.SessionMemory`.
        Call this after each inference turn to save conversational state.

        Parameters
        ----------
        session_id:
            Unique key for the conversation (e.g. ``"user_42_chat_7"``).
        cache:
            The ``ModelCacheStruct`` returned by a ``use_cache=True`` forward.
        token_count:
            Cumulative tokens in this session (for bookkeeping).
        db_path:
            Path to the SQLite database file.
        """
        from .memory import SessionMemory
        with SessionMemory(db_path) as mem:
            mem.save(session_id, cache, token_count=token_count)

    def load_state(
        self,
        session_id: str,
        db_path: str = ".mt_lnn_state.db",
        device: Union[str, torch.device] = "cpu",
    ) -> Optional["ModelCacheStruct"]:
        """Load persisted h_prev states and return a seed ``ModelCacheStruct``.

        Returns ``None`` if *session_id* is not found in the database.

        Parameters
        ----------
        session_id:
            Same key used in :meth:`save_state`.
        db_path:
            Path to the SQLite database file.
        device:
            Device to move tensors onto after loading.

        Example
        -------
        >>> cache = model.load_state("my_session") or ModelCacheStruct()
        >>> out = model(input_ids, cache=cache, use_cache=True)
        >>> model.save_state("my_session", out["cache"], token_count=42)
        """
        from .memory import SessionMemory
        with SessionMemory(db_path) as mem:
            h_states = mem.load(session_id, device=device)
            if h_states is None:
                return None
            return mem.restore_cache(h_states)

    # ------------------------------------------------------------------
    # Multimodal helper: token embeddings for fusion with other modalities
    # ------------------------------------------------------------------

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw token embeddings (B, T, d_model) WITHOUT dropout.

        Use this to build a fused multimodal sequence: concatenate these with
        projected image/audio features (see :mod:`mt_lnn.multimodal`) along the
        sequence axis, then pass the result to ``forward(inputs_embeds=...)``.
        The embedding dropout is applied inside ``forward`` so fused and
        text-only paths are treated identically.
        """
        return self.embedding.token_embed(input_ids)

    # ------------------------------------------------------------------
    # Autoregressive generation (native, KV-cache incremental decoding)
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Keep only the top-k logits per row; rest → -inf. logits: (B, V)."""
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]   # (B, 1)
        return logits.masked_fill(logits < kth, float("-inf"))

    @staticmethod
    def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus filtering: drop the low-probability tail beyond cumulative
        mass top_p, always keeping at least one token. logits: (B, V)."""
        if not (0.0 < top_p < 1.0):
            return logits
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Mark tokens whose *preceding* cumulative mass already exceeded top_p.
        remove = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        remove_scattered = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        return logits.masked_fill(remove_scattered, float("-inf"))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,                   # (B, T_prompt)
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        cache: Optional["ModelCacheStruct"] = None,
        return_cache: bool = False,
        step_callback: Optional[Callable[["ModelCacheStruct", int], None]] = None,
    ):
        """Batch autoregressive decoding with O(1)-per-step KV-cache reuse.

        Returns the full token tensor ``(B, T_prompt + n_generated)``. With
        ``return_cache=True`` returns ``(tokens, cache)`` so a conversation can
        be continued (or persisted via :meth:`save_state`).

        Per-sequence EOS: once a row emits ``eos_token_id`` it is "finished" and
        all subsequent positions are filled with ``pad_token_id`` (defaults to
        ``eos_token_id``); decoding stops early once every row has finished.

        ``step_callback`` is an optional, fully generic decode hook of the form
        ``fn(cache, step) -> None`` invoked **after every forward** (``step=0``
        after the prompt prefill, then ``1, 2, …`` after each incremental step).
        It receives the just-updated :class:`ModelCacheStruct` and may **mutate it
        in place** — e.g. correct a layer's recurrent state
        ``cache.layers[i] = (kv, steered_h, gwtb_kv)`` so the edit conditions the
        *next* token (this is how L3 causal activation steering is applied inside
        real generation; see :class:`mt_lnn.causal_decoding.CausalDecodeSteerer`).
        The hook is policy-free and imports nothing model-specific, so the model
        stays decoupled from any particular monitor/actuator. ``None`` (default)
        leaves decoding bit-for-bit unchanged.
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B = input_ids.shape[0]
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        # 1. Prefill the prompt (continuing from a prior cache if supplied).
        out = self.forward(input_ids, cache=cache, use_cache=True)
        cache = out["cache"]
        if step_callback is not None:
            step_callback(cache, 0)                            # may steer h_prev
        logits = out["logits"][:, -1, :]                      # (B, V)
        generated = input_ids
        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        # 2. Incremental decode.
        step_idx = 0
        for _ in range(max_new_tokens):
            if do_sample:
                logits = logits / max(temperature, 1e-6)
                logits = self._filter_top_k(logits, top_k)
                logits = self._filter_top_p(logits, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)   # (B, 1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)       # (B, 1)

            # Finished rows emit pad and are frozen.
            next_tok = torch.where(
                unfinished.unsqueeze(1), next_tok,
                torch.full_like(next_tok, pad_token_id),
            )
            generated = torch.cat([generated, next_tok], dim=1)

            if eos_token_id is not None:
                unfinished = unfinished & (next_tok.squeeze(1) != eos_token_id)
                if not unfinished.any():
                    break

            out = self.forward(next_tok, cache=cache, use_cache=True)
            cache = out["cache"]
            step_idx += 1
            if step_callback is not None:
                step_callback(cache, step_idx)                # may steer h_prev
            logits = out["logits"][:, -1, :]

        if was_training:
            self.train()
        return (generated, cache) if return_cache else generated
