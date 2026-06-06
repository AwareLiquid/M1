"""
model.py — MTLNNModel: full Microtubule-Enhanced Liquid Neural Network.

Dual-state inference cache:
  - Attention sub-layer: KV cache (past_kv = (K, V)) per layer
  - LNN sub-layer:       recurrent hidden state h_prev per layer

A `LayerCache` per layer is the tuple (kv, h_prev). The full cache is a list
of LayerCache, one per block. Pass `use_cache=True` to enable.
"""

from typing import List, Optional, Tuple, Dict, Union
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

        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn: MicrotubuleAttention   # set by parent after embedding init
        self.lnn_norm = nn.LayerNorm(config.d_model)
        self.lnn = MTLNNLayer(config)

        if self.has_gwtb:
            self.gwtb_norm = nn.LayerNorm(config.d_model)
            self.gwtb = GWTBLayer(config)
        else:
            self.gwtb_norm = None
            self.gwtb = None

    def forward(
        self,
        x: torch.Tensor,                                  # (B, T_new, d_model)
        layer_cache: Optional[LayerCache] = None,
        pad_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        use_lnn_recurrence: bool = True,
    ) -> Tuple[torch.Tensor, Optional[LayerCache]]:

        past_kv      = layer_cache[0] if layer_cache is not None else None
        h_prev       = (layer_cache[1] if (layer_cache is not None and use_lnn_recurrence)
                        else None)
        past_gwtb_kv = (layer_cache[2] if (layer_cache is not None and len(layer_cache) > 2)
                        else None)

        # Attention sub-layer (pre-norm)
        attn_out, new_kv = self.attn(
            self.attn_norm(x),
            pad_mask=pad_mask,
            past_kv=past_kv,
            position_offset=position_offset,
            use_cache=use_cache,
            h_prev=h_prev,  # Pass h_prev for position-free timing signal
        )
        x = x + attn_out

        # LNN sub-layer (pre-norm).
        # use_lnn_recurrence drives BOTH (a) whether to pull h_prev from the
        # cache or use zeros, and (b) whether the resonance bank runs a real
        # parallel scan (True) or the legacy broadcast-h_prev parallel mode
        # (False). The two together preserve cache parity in both modes.
        lnn_out, h_last = self.lnn(
            self.lnn_norm(x), h_prev,
            position_offset=position_offset,
            use_scan=use_lnn_recurrence,
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

        self.embedding = MTLNNEmbedding(config)

        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            block = MTLNNBlock(config, layer_idx=i)
            block.attn = MicrotubuleAttention(config, rope=self.embedding.rope)
            self.blocks.append(block)

        # Global Workspace Theory Bottleneck.
        # If gwtb_per_block=True the bottleneck lives inside every block (paper
        # §4 default). Otherwise it's applied once after the entire stack.
        # Mutually exclusive to avoid double-broadcasting the workspace.
        self.gwtb_per_block = config.gwtb_per_block
        if not config.gwtb_per_block:
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
            )
        else:
            self.world_model_head = None

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
        
        # Phase 2 & 3: Causal Chain and Self-Monitor Heads
        if getattr(config, "use_causal_head", False):
            self.causal_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if getattr(config, "use_self_monitor_head", False):
            self.self_monitor_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight initialisation
        self.apply(lambda m: init_weights(m, config))
        nn.init.normal_(self.target_queries, mean=0.0, std=0.02)
        init_mt_params(self, config)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,                              # (B, T_new)
        cache: Optional[ModelCacheStruct] = None,
        pad_mask: Optional[torch.Tensor] = None,              # (B, T_total) bool
        labels: Optional[torch.Tensor] = None,                # (B, T_new) for causal LM loss
        direct_target_labels: Optional[torch.Tensor] = None,  # (B, L) for direct extraction
        target_len: Optional[int] = None,
        return_target_logits: bool = False,
        use_cache: bool = False,
        position_offset: Optional[int] = None,
        use_lnn_recurrence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          use_cache: build/extend the KV+coherence cache for incremental decoding.
          use_lnn_recurrence: if True (default for inference), thread h_prev across
            steps so the LNN behaves as a true RNN. Set False to match the
            training-time parallel mode (h_prev=0 each step) — this is what makes
            cached vs full-forward outputs bit-exact equal.
          position_offset: if None, infer from cache (length of cached K) or 0.

        Returns dict with keys:
          - logits:   (B, T_new, vocab_size)
          - target_logits: (B, L, vocab_size) if direct extraction is requested
          - loss:     scalar (only if labels provided)
          - target_loss: scalar (only if direct_target_labels provided)
          - cache:    new ModelCacheStruct (only if use_cache=True)
        """
        # Infer absolute position offset
        if position_offset is None:
            if (cache is not None and len(cache.layers) > 0
                and cache.layers[0] is not None and cache.layers[0][0] is not None):
                position_offset = cache.layers[0][0][0].shape[2]   # T_past from first layer's K
            else:
                position_offset = 0

        x = self.embedding(input_ids)                         # (B, T_new, d_model)

        # Phase C → LAVI linkage: distribute the PREVIOUS step's world-model
        # prediction error to all blocks before the loop so LAVIEstimator can
        # use it as a supplementary transience signal.
        # Uses last_pred_error (a buffer holding last forward's value — causal).
        # As of v2.1 this buffer is the NORMALISED surprise ∈ [0,1] (1-cos)/2,
        # so the LAVI correction wm_correction = tanh(scale)·pred_error stays
        # bounded regardless of representation magnitude.
        if self.world_model_head is not None:
            _wm_err = self.world_model_head.last_pred_error.item()
            for _blk in self.blocks:
                _blk.lnn._last_wm_pred_error = _wm_err

        new_cache = ModelCacheStruct(
            token_count=position_offset + input_ids.shape[1]
        ) if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = (cache.layers[i] if (cache is not None and i < len(cache.layers)) else None)
            x, new_layer_cache = block(
                x,
                layer_cache=layer_cache,
                pad_mask=pad_mask,
                position_offset=position_offset,
                use_cache=use_cache,
                use_lnn_recurrence=use_lnn_recurrence,
            )
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
            x, gwtb_new_kv = self.gwtb(
                x, past_kv=gwtb_past, position_offset=position_offset,
                use_cache=use_cache,
            )
            if use_cache:
                new_cache.gwtb_kv = gwtb_new_kv

        coh_past = cache.coherence_kv if cache is not None else None
        x, coh_new_kv = self.coherence(
            x, past_kv=coh_past, position_offset=position_offset, use_cache=use_cache
        )
        if use_cache:
            new_cache.coherence_kv = coh_new_kv

        x = self.final_norm(x)

        # Predictive State Head (Phase C): runs on normed x.
        # During training (compute_loss=True, T>1): accumulates L_wm.
        # During inference (T=1): updates last_pred_error buffer only.
        _world_model_loss: Optional[torch.Tensor] = None
        if self.world_model_head is not None:
            _, _world_model_loss = self.world_model_head(
                x, compute_loss=(self.training and input_ids.shape[1] > 1)
            )

        logits = self.lm_head(x)                              # (B, T_new, vocab_size)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if use_cache:
            result["cache"] = new_cache

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
            
            if getattr(self.config, "use_predictive_coding", False):
                pred_error_sum = sum(b.lnn.resonance.last_pred_error for b in self.blocks)
                result["pred_loss"] = pred_error_sum
                loss = loss + self.config.predictive_loss_weight * pred_error_sum

            # Phase C: world model prediction loss
            if _world_model_loss is not None:
                wm_weight = getattr(self.config, "world_model_loss_weight", 0.01)
                loss = loss + wm_weight * _world_model_loss
                result["world_model_loss"] = _world_model_loss.detach()

            # Phase D: Hebbian co-activation loss (training only)
            if self.hebbian_reg is not None and self.training:
                hebb_loss = self.hebbian_reg.compute_loss(self)
                if hebb_loss is not None:
                    loss = loss + hebb_loss
                    result["hebbian_loss"] = hebb_loss.detach()

            # Phase A: orthogonality penalty for CompetitiveGWTBLayer (training only)
            # Pushes bid projectors apart in weight space (DeepSeekMoE-style).
            from .gwtb import CompetitiveGWTBLayer
            if (self.gwtb is not None
                and isinstance(self.gwtb, CompetitiveGWTBLayer)
                and self.gwtb._last_ortho_penalty is not None):
                ortho = self.gwtb._last_ortho_penalty
                loss = loss + self.gwtb.ortho_penalty_weight * ortho
                result["ortho_penalty"] = ortho.detach()

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
            # Competitive GWTB diagnostics
            from .gwtb import CompetitiveGWTBLayer
            if isinstance(self.gwtb, CompetitiveGWTBLayer):
                diag["gwtb_competition_entropy"] = self.gwtb.last_competition_entropy.item()
                for k, w in enumerate(self.gwtb.last_winner_weights.tolist()):
                    diag[f"gwtb_bid{k}_weight"] = w
        else:
            # Per-block GWTB — report mean / spread across blocks
            gates = [b.gwtb.broadcast_gate.item() for b in self.blocks if b.has_gwtb]
            if gates:
                gates_t = torch.tensor(gates)
                diag["gwtb_broadcast_gate_mean"] = gates_t.mean().item()
                diag["gwtb_broadcast_gate_std"]  = gates_t.std().item()
                diag["gwtb_d_gw"] = float(self.blocks[0].gwtb.d_gw)
        return diag

    def get_mt_histograms(self) -> Dict[str, torch.Tensor]:
        """Return raw tensors for W&B histogram logging."""
        taus, gammas, pols = [], [], []
        for block in self.blocks:
            tau = F.softplus(block.lnn.resonance.log_tau).detach() + self.config.tau_min
            taus.append(tau.flatten())
            gammas.append(block.lnn.gtp_gamma.detach().reshape(-1))
            pols.append(block.attn.polarity_direction.detach())
        return {
            "tau": torch.cat(taus).cpu().float(),
            "gamma": torch.cat(gammas).cpu().float(),
            "polarity": torch.cat(pols).cpu().float(),
        }

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
