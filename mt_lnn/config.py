import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MTLNNConfig:
    # Vocabulary and sequence
    vocab_size: int = 50257          # GPT-2 BPE default
    max_seq_len: int = 1024
    pad_token_id: int = 0

    # Model dimensions
    # d_model = 832 = 13 × 64 → d_proto = 64 (Tensor-Core aligned).
    # n_heads = 13 chosen so each head naturally corresponds to one
    # protofilament, with d_head = 832 / 13 = 64.
    d_model: int = 832
    n_layers: int = 12               # 12 layers → ~125M params
    n_heads: int = 13
    n_kv_heads: int = 1              # GQA: 1 KV head, 13 Q heads → 13× KV-cache savings
    d_head: int = 64                 # d_model // n_heads

    # Microtubule protofilament settings
    # Biological constant is 13; vectorised forward scales near-flat so going
    # higher (32 / 64 / 128) only costs more parameters, not more wall-clock.
    n_protofilaments: int = 13
    map_hidden_dim: int = 64

    # LTC / ODE parameters
    tau_init: float = 1.0
    tau_min: float = 0.01
    tau_max: float = 10.0
    dt: float = 1.0

    # Polarity Attention mode
    #   "scalar"   — per-head learned scalar polarity (cheap, current default)
    #   "low_rank" — content-aware low-rank bilinear σ(X W_A (X W_B)^T) bias.
    #                Mimics α/β-tubulin pair interactions; rank-r adds 2·d·r
    #                params per head.
    polarity_mode: str = "scalar"
    polarity_rank: int = 16  # Increased from 8 for better content-based attention capacity

    # GTP hydrolysis (lateral coupling in MTLNNLayer)
    gamma_init: float = 0.1
    # GTP cap renewal period — lateral coupling refreshes every gtp_period
    # positions. Without this, the exp(-γ·t) gate vanishes at large t and
    # microtubule mixing silently dies in long contexts.
    gtp_period: int = 256

    # Continuous τ spectrum. With n_time_scales > 3 the resonance frequencies
    # are chosen as a geometric sweep spanning tau_min → tau_max.
    n_time_scales: int = 5
    resonance_freqs: Optional[Tuple[float, ...]] = None

    # Global Workspace Theory Bottleneck (compress → workspace SA → broadcast)
    # Workspace dim d_gw = d_model // gwtb_compression_ratio.
    # gwtb_broadcast_init is the initial value of the gated residual scalar;
    # small so the layer starts as near-identity and ramps up during training.
    gwtb_compression_ratio: int = 8
    gwtb_n_heads: int = 4
    gwtb_broadcast_init: float = 0.01
    # If True, every MTLNNBlock contains its own GWTB instance (paper §4).
    # If False (default), GWTB is applied once after the block stack.
    # Per-block adds ~3% parameters and ~10% wall-clock per layer at d_gw=104,
    # in exchange for the layer-wise ignition semantics the paper prescribes.
    gwtb_per_block: bool = False

    # Global coherence (Orch-OR collapse, complementary to GWTB)
    coherence_sparsity: float = 0.1  # keep top 10% of attention scores
    coherence_heads: int = 4
    
    # Working memory / exponential decay in Global Coherence KV.
    # Set to True to replace persistent KV cache with an O(1) state buffer.
    use_decay_wm: bool = True
    wm_decay_rate_init: float = 0.99

    # Endogenous Compute Skipping (Hard gating threshold)
    # Automatically zeros out dynamic_kappa connections < threshold.
    compute_skip_threshold: float = 0.0

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Misc
    tie_embeddings: bool = True

    # Predictive Coding across tau channels
    use_predictive_coding: bool = True
    predictive_loss_weight: float = 0.05

    # Direct target extraction head. This auxiliary head reads the final
    # contextual state once and emits a fixed number of target slots, so tasks
    # such as Selective Copy can be evaluated without autoregressive decoding.
    direct_target_max_len: int = 16

    # Position-Free Architecture (Experimental - Default OFF)
    # WARNING: Only enable after validating on 200K parameter model first!
    use_position_free_attention: bool = False    # Replace RoPE with h_prev-based timing
    h_prev_position_weight: float = 0.05         # h_prev position signal strength (0.05 = 5% of content)
    keep_relative_bias: bool = True              # Keep GTP-cap relative distance bias
    position_free_mode: str = "hybrid"           # "hybrid"(KV+h_prev) or "state_only"(h_prev only)
    
    # Optional Causal Chain and Self-Monitor Extraction Heads (Phase 2 & 3)
    use_causal_head: bool = False
    use_self_monitor_head: bool = False

    # Dynamic multi-scale tau gates. The first phase only gates the blend over
    # already-computed tau scales; compute skipping is kept behind a later flag.
    dynamic_scale_gates: bool = True
    scale_gate_init_bias: float = 2.0
    scale_gate_active_threshold: float = 0.5
    scale_gate_skip_threshold: float = 0.0
    sparse_resonance_kernel: bool = False
    sparse_resonance_top_k: int = 1

    # EEG-inspired rhythm gate (LAVI). Default OFF — no impact on existing code.
    # use_rhythm: attach LAVIEstimator to each MTLNNLayer; modulates the τ-scale
    #   blend toward slow scales (persistent mode) or fast scales (transient mode)
    #   based on cosine similarity between h_prev and the current input.
    # rhythm_scale_init: initial tanh-gated influence of LAVI on blend weights.
    #   Small value (0.1) means rhythm starts nearly disabled; training grows it.
    # global_rhythm: attach GlobalRhythmController to MTLNNModel to aggregate
    #   cross-layer LAVI means and apply a small residual correction before GWTB.
    use_rhythm: bool = False
    rhythm_scale_init: float = 0.1
    global_rhythm: bool = False

    # Derived (set in __post_init__)
    d_proto: int = field(init=False)
    d_proto_total: int = field(init=False)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_head == self.d_model // self.n_heads, "d_head must equal d_model // n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads (GQA)"
        # Pad to next multiple of n_protofilaments so each proto gets equal width
        self.d_proto = math.ceil(self.d_model / self.n_protofilaments)
        self.d_proto_total = self.d_proto * self.n_protofilaments
        # e.g. d_model=1024, P=13: d_proto=79, d_proto_total=1027

        # Tensor-Core alignment warning: protofilament-level einsums see best
        # GPU throughput when d_proto is a multiple of 8 (fp16/bf16) or 16. The
        # closest aligned d_model values for the current P, n_heads are listed
        # by recommended_aligned_d_model().
        if self.d_proto % 8 != 0:
            import warnings
            aligned = self.recommended_aligned_d_model(self.d_model)
            warnings.warn(
                f"d_proto={self.d_proto} is not a multiple of 8 — protofilament "
                f"einsums won't hit Tensor Cores optimally. Nearest aligned "
                f"d_model values (n_protofilaments={self.n_protofilaments}, "
                f"n_heads={self.n_heads}): {aligned}",
                RuntimeWarning, stacklevel=2,
            )

        # Continuous τ spectrum: geometric sweep tau_min → tau_max.
        # Each scale s in [0, n_time_scales) gets τ_s = tau_min * (tau_max/tau_min)^(s/(S-1))
        if self.resonance_freqs is None:
            if self.n_time_scales == 1:
                self.resonance_freqs = (self.tau_init,)
            else:
                ratio = self.tau_max / self.tau_min
                freqs = tuple(
                    self.tau_min * (ratio ** (s / (self.n_time_scales - 1)))
                    for s in range(self.n_time_scales)
                )
                self.resonance_freqs = freqs
        else:
            assert len(self.resonance_freqs) == self.n_time_scales, \
                f"resonance_freqs length {len(self.resonance_freqs)} != n_time_scales {self.n_time_scales}"

    def recommended_aligned_d_model(self, target: int, n: int = 5) -> list:
        """
        Return up to `n` d_model values near `target` that satisfy:
          - divisible by n_heads (so d_head is integral)
          - d_proto = d_model / n_protofilaments is a multiple of 8 (Tensor-Core
            friendly for the per-protofilament einsums)
        """
        results = []
        for delta in range(0, 4096, 8):
            for candidate in {target - delta, target + delta}:
                if candidate <= 0:
                    continue
                if candidate % self.n_heads != 0:
                    continue
                # d_proto must be an integer multiple of 8 → d_model = P * 8k
                if candidate % (self.n_protofilaments * 8) != 0:
                    continue
                if candidate not in results:
                    results.append(candidate)
                if len(results) >= n:
                    return sorted(results)
        return sorted(results)
