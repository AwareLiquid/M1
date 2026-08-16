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
    #
    # Attention heads are DECOUPLED from the protofilament count. The original
    # "one head per protofilament" pairing (n_heads = 13) was naming aesthetics,
    # not mechanism — nothing in the attention path consumes n_protofilaments,
    # and the multi-timescale resonance semantics live entirely in the LNN's
    # 13 protofilaments. The pairing had a real cost: 13 is prime, so the only
    # expressible GQA ratios were 13:1 (MQA — P0 round 4 measured acc 0.248 on
    # the relational probe) and 1:1 (full MHA — acc 1.0000 but 13× the KV
    # cache). Every production middle ratio (8:1, 4:1, 2:1) was arithmetically
    # impossible. See ABLATIONS.md "Design-coupling audit".
    #
    # Defaults stay 13/1 — bit-exact with every existing checkpoint. A
    # decoupled shape such as n_heads=16, d_head=52, n_kv_heads=4 (keeping
    # d_model=832 and d_proto=64) is verified end-to-end by
    # tests/test_head_decoupling.py; flipping the default awaits the probe
    # sweep (HANDOFF §3.8).
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
    # M2 architecture principle #1 (docs/ROADMAP_M2.md §4.5): reserve this many
    # TRULY global attention heads per layer (γ init ≈ 1e-3, no effective
    # distance decay); remaining heads keep the biological GTP-decay schedule.
    # P0 measured the old all-decaying init blocking in-context relational
    # lookup entirely (0.25 → 1.00 once freed). Default 0 = historical init,
    # bit-exact. Flip the default only after the --n_global_heads sweep on the
    # (shortcut-sealed) pointer-chase probe lands.
    n_global_heads: int = 0
    # Hybrid thinning (M2 review 2026-07-30, HANDOFF 3.8 item 1): which layers
    # keep an attention sub-layer. None = every layer (bit-exact historical
    # default). The current hybrid is ADDITIVE -- a full 12-layer Transformer
    # with liquid layers stacked on top -- so it pays full KV growth and full
    # attention activations, which is why "hybrid is O(1)" had to be retracted
    # (RESULTS.md) and training memory measured strictly worse. Production
    # hybrids REPLACE: LFM2 runs attention in 6/16 layers. attention_layers=
    # (2,5,8,11) on a 12-layer model cuts the KV cache and attention compute
    # to a third; the other layers become pure LNN+FFN blocks.
    # Placement matters and is unswept -- late-ish layers follow the hybrid
    # literature, but treat any specific placement as a hypothesis.
    attention_layers: "Optional[Tuple[int, ...]]" = None
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
    # Full GWTB on/off (2026-08-16). O 系列（attention-free 端侧）应设 False：
    # GWTBLayer 预分配 O(T²) causal mask（128k ≈ 17GB），是 O 系列无注意力
    # 之后唯一的 O(T²) 残留。默认 True = 历史路径。
    use_gwtb: bool = True
    # J-Space J1 (docs/JSPACE_DESIGN.md): workspace reverberation. The
    # workspace self-attention iterates this many weight-tied passes per
    # forward — content "reverberates on the stage" before broadcast. At
    # d_gw = d_model/8 this is the cheapest thinking-depth knob in the model
    # (~1/64 the cost of iterating the backbone). Default 1 = the existing
    # single-pass path, bit-exact. No new parameters.
    workspace_iterations: int = 1
    # If True, every MTLNNBlock contains its own GWTB instance (paper §4).
    # If False (default), GWTB is applied once after the block stack.
    # Per-block adds ~3% parameters and ~10% wall-clock per layer at d_gw=104,
    # in exchange for the layer-wise ignition semantics the paper prescribes.
    gwtb_per_block: bool = False

    # Latent recurrent depth — "thinking steps" (M2 P0, docs/ROADMAP_M2.md §4).
    # The liquid-core sub-layer of every block is applied core_iterations times
    # per forward pass: iteration k re-reads the SAME normed input through the
    # SAME MTLNNLayer weights, but (a) starts from iteration k-1's final
    # recurrent state ("re-scan with updated working memory" — this ungated
    # state threading IS the iteration mechanism), and (b) adds iteration
    # k-1's output to its input through a zero-init tanh gate (learnable
    # feedback strength). Weight-tied depth in the Universal-Transformer /
    # latent-recurrent-depth family, scoped to the LNN sub-layer so attention
    # KV is computed once. Default 1 = exactly the pre-existing single-pass
    # code path with a byte-identical parameter set (the gate parameter is
    # only created when core_iterations > 1). position_offset is held constant
    # across iterations so the GTP periodic clock does not drift.
    core_iterations: int = 1
    core_iter_gate_init: float = 0.0

    # Signed decay — negative-eigenvalue extension (M2 separation study,
    # ABLATIONS.md / memory 2026-08-03/04). The stock liquid update's strictly
    # positive diagonal transition provably cannot express parity in finite
    # precision (Sarrof et al. NeurIPS 2024 Thm 2; measured: parity acc 0.492
    # ≈ chance). Following Grazzi et al. (ICLR 2025), True extends the state
    # eigenvalue to λ = decay · tanh(sign_raw) ∈ (−decay, decay) with a
    # learnable (P,S) sign parameter (init 3.0 → tanh≈0.995, near-stock start).
    # Input coefficient stays (1 − decay). Default False = no parameter, exact
    # historical path.
    signed_decay: bool = False
    # Init mode for the sign parameter. "stock": constant 3.0 → tanh≈0.995,
    # near-historical dynamics — BUT tanh'(3)=0.0099, so flipping a channel
    # negative must crawl through the saturation plateau; measured 2026-08-04:
    # parity stayed at chance for 10k steps under this init (both arms
    # collapsed to constant prediction). "mixed": s ~ U(−2, 2) — half the
    # channels start negative and every channel sits in the live-gradient
    # region; the oscillation basis parity needs exists from step 0.
    signed_decay_init: str = "stock"

    # Selective decay — INPUT-DEPENDENT signed transition (the actual parity
    # fix). Measured 2026-08-04: signed_decay alone (constant λ, both stock
    # and mixed init) leaves parity at exact chance — as it must: a
    # constant-λ diagonal core computes fixed-weight linear sums
    # Σ λ^(T−t)·b(x_t) regardless of sign. Parity needs the TRANSITION to
    # depend on the token (flip on 1, hold on 0): λ_t = decay · tanh(W_sel·x_t
    # + b_sel). This is Mamba-class selectivity — and is faithful to the
    # ORIGINAL LTC theory, whose time constant τ(x) is input-dependent (the
    # implementation dropped that for pscan_constant_A speed; the general
    # pscan supports per-step multipliers). Supersedes signed_decay when both
    # are set. Default False = no parameters, exact historical path.
    selective_decay: bool = False

    # Transition parameterisation for selective_decay (E5e, 2026-08-15).
    # "tanh" (default, historical): λ_t = decay·tanh(W_sel·x_t + b_sel) —
    # saturating, |λ|<decay<1, per-step state leak; E5d measured it DESTROYS
    # length extrapolation (extrap 0.023 in the minimal A/B).
    # "exp": λ_t = 2·exp(−softplus(W_d·x_t + b_d)/τ) − 1 — input inside the
    # exponential, reaches ±1 exactly; E5d measured extrap 0.953 with 2/3
    # seeds at PERFECT 1.000, reproducing the branch's both_khavari result.
    selective_decay_mode: str = "tanh"
    # 推理翻转硬化（2026-08-16）：soft λ_t 的 ±1 偏离在超长序列累积误差
    # （实测 parity 外推 64 满分、1k 崩 0.000）。True 时推理（eval）阶段
    # 把 λ_t snap 到精确 ±1（sign），训练保持连续可导。parity 的
    # flip/hold 语义因此精确，无累积误差。
    selective_decay_snap: bool = False
    # STE 训练时离散化（2026-08-16）：训练 forward 用 sign(λ)（精确 ±1），
    # backward 用 soft 梯度（straight-through）。训练/推理语义一致，修复
    # 推理 snap 的 OOD 问题（snap 使 in-dist 都崩）。parity 翻转决策在训练
    # 中学会精确 ±1，外推不再受 soft 累积误差限制。
    selective_decay_ste: bool = False

    # Householder NDIT — NON-DIAGONAL input-dependent transition
    # (docs/NONDIAGONAL_TRANSITION.md, 2026-08-15). A5 (NC1-complete) needs a
    # non-diagonal state transition (Merrill ICML 2024 Cor 4.7); the diagonal
    # selective_decay — regardless of parameterisation — provably cannot.
    # Q_t = I - 2 v_t v_t^T per protofilament, v_t = normalize(W_h x_t + b_h),
    # unitary → spectral radius exactly 1 → no state explosion. Off by default
    # (zero regression); requires selective_decay=True to compose with λ_t.
    use_householder_transition: bool = False
    # Number of Householder reflections per token (NDIT rank). 1 = single
    # involution (only D degrees of freedom — measured INSUFFICIENT for A5,
    # 2026-08-15 diag: rotation active but A5 fails); k reflections compose to
    # a richer unitary subgroup (rank-k correction, k·D degrees of freedom).
    householder_rank: int = 2

    # DeltaProduct NDIT — NON-INVOLUTORY dense low-rank transition
    # (docs/NONDIAGONAL_TRANSITION.md §4.5, 2026-08-15). Householder
    # involutions failed A5 (rank-1 = rank-2 = 0.130, diagnosis: active but
    # wrong inductive bias — Q²=I cannot encode 60 distinct group-multiply
    # semantics). DeltaProduct-style update instead:
    #   h_t = (I + Σ_r δ_r(x_t) u_r(x_t) v_r(x_t)ᵀ) h_{t-1} + (1-decay) ⊙ B_t
    # Non-involutory, non-diagonal, input-dependent; δ_r starts tiny so the
    # transition is ≈ I at init (stable) and opens up as training proceeds.
    use_deltaproduct_transition: bool = False
    deltaproduct_rank: int = 2
    deltaproduct_scale: float = 0.1

    # Component ablation switches (E5c, 2026-08-15). The length-extrapolation
    # gap between the branch's minimal probe (extrap 1.000) and main's full
    # MTLNNLayer (extrap ~0.3) must be attributed to one of these components.
    # Default True = exact historical path (zero regression); False removes
    # the component entirely (no params built/used, bit-identical to a
    # stripped layer).
    use_lateral_coupling: bool = True
    use_map_gate: bool = True

    # Stack-level latent recurrence (M2 P0-C′, docs/ROADMAP_M2.md §4.5).
    # Re-apply the ENTIRE block stack (attention + LNN, weight-tied) N times
    # per forward — Universal-Transformer-style depth. P0 rounds 1–5 showed
    # compositional lookup lives in ATTENTION, so iterating only the LNN
    # sub-layer (core_iterations) buys no reasoning depth; this knob loops the
    # attention too. Default 1 = existing single-pass path (zero regression).
    # No parameters are added; use_cache is not supported when > 1.
    stack_iterations: int = 1

    # Competitive Global Workspace (Phase A, 2026-06-06)
    # When True, the top-level GWTBLayer is replaced with CompetitiveGWTBLayer.
    # K specialist bid projectors (residual init → all bids start as x → zero-
    # impact at init) compete via a score head. Winner's representation enters
    # the workspace bottleneck; broadcast is added back to the *original* x.
    # Only applies when gwtb_per_block=False (top-level GWTB mode).
    use_competitive_gwtb: bool = False
    n_competitive_bids: int = 3          # K concurrent workspace bids
    competitive_hard_winner: bool = False  # True: argmax at inference, soft in training
    # Symmetry-breaking mechanisms (2026-06-07, P0 fix per docs/reviews/V2_REVIEW.md):
    # Without these, all K bids start identical → score gradient is zero by symmetry →
    # competition stays at uniform 1/K forever (DeepSeekMoE "routing collapse" failure).
    competitive_score_noise: float = 0.5  # Gaussian noise σ on scores during training
    competitive_ortho_weight: float = 0.01  # weight of bid-projector orthogonality penalty
    # P3.1 multi-source GWT (2026-06-07): let *external* modules (e.g. the
    # predictive world model) submit their own bids into the workspace
    # competition alongside the K internal BidProjectors. When ON, the model
    # builds a residual, zero-init-gated adapter per external source so that at
    # init each external bid == x (identical-to-internal) → competition stays
    # uniform → output is bit-identical to the no-external-bid path (zero
    # regression). The competition then learns how much to trust each source.
    gwtb_external_bids: bool = False
    gwtb_external_bid_gate_init: float = 0.0  # residual gate init for external bids

    # Dynamic workspace bandwidth (2026-06-15): a per-channel "arousal" gate over
    # the d_gw workspace bottleneck. The current GWTB bottleneck has a FIXED width
    # (d_gw channels are always fully active); this makes it input-dependent.
    #   g_t = sigmoid(W_g · z_t + b_g)   ∈ (0,1)^d_gw
    #   z_gated = z_t * g_t
    # Brain analog: the global workspace allocates how many bottleneck channels
    # "ignite" per token — few cortical units active on calm/redundant input, many
    # on surprising input (Dehaene's all-or-none ignition is graded here so it stays
    # differentiable). This is the "fixed-bandwidth → dynamic-bandwidth" upgrade.
    #
    # Zero-regression contract:
    #   • Default OFF → no gate parameters are built → forward is bit-identical to
    #     the pre-existing fixed-bandwidth path.
    #   • When ON, the gate weight is zero-init and the bias is large-positive
    #     (gwtb_bandwidth_gate_bias) → g ≈ sigmoid(bias) ≈ "mostly open" at init →
    #     the workspace starts at near-full bandwidth and *learns* to close
    #     redundant channels, rather than starting sparse and risking dead units.
    gwtb_dynamic_bandwidth: bool = False
    gwtb_bandwidth_gate_bias: float = 4.0    # b_g: sigmoid(4)≈0.982 → mostly-open at init
    gwtb_bandwidth_threshold: float = 0.1    # channels with g < this count as "dormant"
    # Eval-time compute saving: hard-mask dormant channels to exactly zero (a
    # genuine compute-skip, like cortical neurons staying silent). Training always
    # uses the soft multiply so gradients flow to every channel. Default OFF keeps
    # train/eval behaviour identical when you only want the soft gate.
    gwtb_bandwidth_hard_mask: bool = False

    # Top-down modulation (P1 closed-loop ②, 2026-06-14): a high-level goal /
    # context vector biases EVERY block's representation (cortical top-down
    # feedback). Each block gets a zero-init-gated residual adapter:
    #     x = x + tanh(gate) · proj(LayerNorm(top_down))
    # use_top_down=False (default) → no params built, no behaviour change → all
    # existing checkpoints/tests bit-identical. Even when True, gate init 0 →
    # tanh(0)=0 → bit-exact at init, but the gate keeps a live gradient (proj is
    # small-but-nonzero) so the model can LEARN to open it. The LayerNorm is the
    # init/runtime "insurance": it bounds an arbitrary-scale goal vector so the
    # residual can't destabilise training once the gate opens.
    use_top_down: bool = False
    top_down_gate_init: float = 0.0           # per-block residual gate init (0 → identity)
    # GWT docking entry: additionally offer the top-down goal as an external bid
    # in the top-level global-workspace competition (reuses the world-model bid
    # pathway; needs a CompetitiveGWTBLayer, i.e. gwtb_external_bids=True). The bid
    # is zero-gated, so at init it equals x and competes as one of K+1 equal
    # competitors -- identity-at-init to within the same O(1e-4) softmax-normalisation
    # artifact as the world-model bid (NOT strictly bit-exact, because the model's
    # global init_weights pass perturbs the internal bid projectors off exact
    # identity; see tests/test_multisource_gwt.py). Default OFF, so the default
    # model is entirely unaffected; the per-block path above IS bit-exact at init.
    top_down_to_gwtb: bool = False
    top_down_gwtb_gate_init: float = 0.0      # GWT-bid gate init (0 → ~identity)

    # Synaptic-memory -> GWT docking (Gap 1 closed-loop, 2026-06-25): complete the
    # "spatial/Hebbian associative memory -> global workspace" chain end-to-end by
    # letting a content-addressed FastWeightMemory recall stream submit its own bid
    # into the top-level workspace competition, jointly trained with the main CE.
    # This reuses (no duplication): the FastWeightMemory primitive built for the
    # served adapter (mt_lnn.llama_adapter.FastWeightMemory) AND the same zero-init-
    # gated residual-bid machinery as the world model:
    #     mem_bid = x + gate * fast_weight_recall(x)
    # Needs a CompetitiveGWTBLayer that accepts external bids, i.e.
    # use_competitive_gwtb=True AND gwtb_external_bids=True (same precondition as the
    # world-model / top-down bids). gate init 0 -> bid == x -> competition unchanged
    # at init (zero regression); the recall projection is nonzero so the gate keeps a
    # live gradient and the workspace can LEARN to trust the memory. Default OFF.
    gwtb_memory_bid: bool = False
    gwtb_memory_bid_dim: int = 64             # per-head fast-weight key/value width
    gwtb_memory_bid_heads: int = 1
    gwtb_memory_bid_decay: float = 0.95       # initial association half-life in (0,1)
    gwtb_memory_bid_gate_init: float = 0.0    # residual bid gate init (0 → identity)

    # P3.2 graceful degradation (2026-06-07): wrap the AUXILIARY v2 module
    # contributions (world-model loss, GWTB orthogonality penalty, Hebbian loss,
    # the world-model workspace bid, and the LAVI surprise signal) in finiteness
    # guards. If a bio-inspired module emits NaN/Inf on some step, that module's
    # contribution is dropped for that step (and counted) instead of poisoning
    # the main LM loss and killing a multi-day pre-training run. Zero-regression
    # by construction: on finite values the guards are no-ops. The PRIMARY
    # cross-entropy loss is never masked (a NaN there is a real failure that must
    # surface). Set False to let faults crash loudly during debugging.
    use_graceful_degradation: bool = True

    # Global coherence (Orch-OR collapse, complementary to GWTB)
    coherence_sparsity: float = 0.1  # keep top 10% of attention scores
    coherence_heads: int = 4
    
    # Working memory / exponential decay in Global Coherence KV.
    # Set to True to replace persistent KV cache with an O(1) state buffer.
    # DEFAULT OFF since 2026-07-15 (E4, ITERATION_PRINCIPLES.md): the decay-WM
    # branch runs a per-token Python loop inside every training forward (T
    # small-op dispatches), the prime suspect in the measured 1.6x training
    # slowdown, and its O(1)-memory benefit only matters for very long
    # streaming decode. Opt back in explicitly for streaming deployments.
    use_decay_wm: bool = False
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
    # DEFAULT OFF since 2026-07-15 (E4, ITERATION_PRINCIPLES.md): the O1
    # switch-matrix ablation (BENCHMARKS.md) measured it PPL-NEUTRAL trending
    # NEGATIVE (+0.65 val PPL) at 48M while costing throughput on every
    # training step. A default must earn its cost; this one measurably didn't.
    use_predictive_coding: bool = False
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

    # Multi-Token Prediction (MTP) lookahead heads (2026-06-26).
    # Biological grounding: Friston's predictive coding / free-energy principle
    # posits that the brain simultaneously predicts multiple future states across
    # time scales — not just the next step. These K shallow heads predict tokens
    # t+1 … t+K from the final normed hidden state, trained with a joint CE loss.
    # At inference: draft logits enable speculative decoding. M1's O(1) recurrent
    # state (~4 KB) makes it an exceptionally cheap draft model — no KV cache to
    # re-compute for each draft step.
    #
    # ⚠ VALIDATED NULL (2026-07-12, see BENCHMARKS.md "MTP aux loss — honest
    # null"): matched-budget 3-seed A/B at 125M/2000steps/WikiText-2 gave
    # 267.65±2.63 (MTP) vs 268.40±2.61 (core) — a −0.75 PPL delta well inside
    # seed noise, not a resolvable win. The K heads add 125.4M params (+99% of
    # the base model) for a training-only term. DO NOT enable by default;
    # keep opt-in for research only.
    #
    # Zero-regression contract:
    #   use_mtp_heads=False (default) → no MTP parameters built, forward unchanged.
    #   use_mtp_heads=True with mtp_loss_weight=0.0 → heads present, no aux loss.
    #   The main CE loss (result["lm_loss"]) is never affected — MTP is additive.
    #
    # Training: loss += mtp_loss_weight * mean(CE(draft_k[t], label[t+1+k]) for k in 1..K)
    # (head k predicts t+1+k — the main head owns t+1, DeepSeek-V3 convention;
    #  labels[t+k] would make head k=1 duplicate the main CE exactly)
    # Inference: result["mtp_draft_logits"] → Tensor(B, T, K, vocab_size)
    use_mtp_heads: bool = False
    mtp_lookahead: int = 3            # K: number of future tokens to predict
    mtp_loss_weight: float = 0.1     # λ: aux loss weight (won't affect lm_loss PPL)
    # Master throttle for the INFERENCE speculative-decoding path ONLY. The
    # serving-side draft→verify consumer is a separate, future iteration and is
    # NOT built yet (RESEARCH — not wired; needs a verify loop + O(1)-recurrent-
    # cache rollback). When True, forward exposes result["mtp_draft_logits"] for
    # that (absent) consumer; leaving False keeps every inference forward free of
    # the unused draft compute even when use_mtp_heads=True builds the params.
    # NOTE (2026-07-12): the MTP TRAINING aux loss NO LONGER depends on this flag
    # (that gating was a bug — it made the regularizer unreachable). The aux loss
    # now runs whenever use_mtp_heads=True, mtp_loss_weight>0, and the forward is
    # a training pass with labels. Enable MTP-as-regularizer with use_mtp_heads
    # alone; flip this True only for the (unwired) speculative-decoding path.
    enable_speculative_decoding: bool = False

    # Dynamic multi-scale tau gates. The first phase only gates the blend over
    # already-computed tau scales; compute skipping is kept behind a later flag.
    dynamic_scale_gates: bool = True
    scale_gate_init_bias: float = 2.0
    scale_gate_active_threshold: float = 0.5
    scale_gate_skip_threshold: float = 0.0
    sparse_resonance_kernel: bool = False
    sparse_resonance_top_k: int = 1

    # Cross-layer scale-gate sharing (IndexShare analog, 2026-06-26).
    # Every `scale_gate_period` MTLNNLayers, one "leader" layer computes the
    # τ-scale selection index (which top-k time scales to activate); the next
    # period-1 "follower" layers reuse that index without re-running kappa_gate.
    # Biological rationale: cortical columns maintain the same dominant
    # oscillatory frequency preference across 2-4 laminar integration steps;
    # re-evaluating the τ-selection on every layer is biologically redundant
    # and computationally wasteful.
    # period=1 (default) → current behaviour, no sharing.
    # period=4 → mirrors GLM-5.2 IndexShare (every 4 layers share 1 indexer).
    # Only active when sparse_resonance_kernel=True (gate sharing without
    # sparse compute skip has negligible benefit).
    # CONSTRAINT: period must be in {1, 2, 4}. Values >4 risk τ-routing
    # rigidity — a single group spanning >4 layers suppresses the dynamic
    # multi-scale diversity that is architecturally central to MT-LNN.
    scale_gate_period: int = 1

    # Hebbian Regularizer (Phase D, 2026-06-06)
    # Loss-level Hebbian co-activation term: L_hebb = -α × mean(out ⊙ x_in)
    # α is modulated by global LAVI mean (persistent mode → stronger consolidation).
    # Does NOT change forward() — purely a training loss term.
    # use_hebbian=False (default) → zero impact on anything.
    use_hebbian: bool = False
    hebbian_lr: float = 1e-4          # base co-activation weight
    hebbian_lavi_gate: bool = True    # gate α by LAVI (False → constant α)

    # ---- Hebbian REFACTOR (mechanism A: loss-term式), 2026-06-16 -------------
    # A staged rebuild of the Hebbian branch that fixes the three verified
    # failure modes of the legacy `use_hebbian` path:
    #   (1) the LAVI gate sat at sigmoid(0)=0.5 because LAVI is only produced
    #       when use_rhythm=True (a separate, off-by-default module) -> the new
    #       path owns a DEDICATED lightweight LAVI estimator (Stage 1) so the
    #       gate has a live, non-trivial input regardless of use_rhythm;
    #   (2) the term's gradient share was ~8e-5 even at hebbian_lr=1e-1
    #       (see experiments/report_ablation_hebbian_lr.md) -> the new path
    #       decouples its base lr AND adaptively aligns its gradient norm to the
    #       main gradient, capped at `hebbian_grad_frac_cap` (Stage 2);
    #   (3) the legacy signal was a same-timestep co-activation only -> Stage 1+
    #       adds a within-sequence lagged term (h_t · h_{t-1}) along the seq dim.
    #
    # Zero-regression contract: use_hebbian_refactor=False (default) builds NO
    # module and adds NO forward/loss op -> bit-identical to the current model.
    # STATUS (corrected 2026-07-12): Stages 1+2 have LANDED — when ON,
    # HebbianPlasticity.compute_loss() returns a REAL loss term (NOT None; the
    # old "Stage 0 no-op" note was stale). ⚠ It is ACTIVE, currently UNVALIDATED
    # (the ablation shows it HURTS val PPL: report_ablation_hebbian_refactor.md),
    # and its grad-fraction safety cap only engages if the training loop calls
    # HebbianPlasticity.recalibrate() — which train.py does NOT. Do not enable on
    # a real training run without wiring recalibrate() first.
    use_hebbian_refactor: bool = False
    hebbian_base_lr: float = 1e-2          # independent base lr (decoupled from main BP lr)
    hebbian_window: int = 32               # LAVI sliding-window length (Stage 1)
    hebbian_grad_frac_cap: float = 0.05    # max share of per-step grad norm (Stage 2)
    hebbian_refactor_mode: str = "weak"    # "weak" (residual-only) | "strong" (all-layer)

    # Predictive State Head / World Model (Phase C, 2026-06-06)
    # When True, a PredictiveStateHead is mounted after final_norm.
    # Training: L_total = L_lm + world_model_loss_weight × L_wm
    # Inference: last_pred_error buffer updated for monitoring + LAVI linkage.
    use_world_model: bool = False
    world_model_loss_weight: float = 0.01   # small: LM loss always dominates
    world_model_hidden_ratio: float = 0.5   # predictor bottleneck width rel. to proj_dim
    # v2.1 — BYOL / V-JEPA EMA target encoder (prevents representational collapse).
    world_model_proj_ratio: float = 0.5     # latent proj dim relative to d_model
    world_model_ema_decay: float = 0.99     # EMA momentum for the target projector
    world_model_use_ema_target: bool = True # False → trainable stop-grad target (ablation)
    world_model_warmup_steps: int = 1000    # EMA warm-up (gentler decay early in training)

    # Physics-informed world model (Hamiltonian head, 2026-07-14). A structured,
    # conservation-biased ALTERNATIVE to PredictiveStateHead: predicts the next
    # hidden latent through a symplectic phase-space bottleneck (see
    # mt_lnn/hamiltonian_head.py HamiltonianWorldModelHead). It reads the liquid
    # core's hidden state, decodes a physical phase state (q,p), evolves it with a
    # velocity-Verlet symplectic step whose potential is CONDITIONED on the liquid
    # core (the "liquid ODE is the physics substrate" claim, realized in code),
    # then re-encodes the predicted next latent. Self-supervised (trains on any
    # sequence), folded into the aux loss like the world model.
    #   Honest scope: a structural inductive bias; expected PPL-neutral on
    #   language (like every physics/bio module in the switch-matrix), with real
    #   value only on continuous-state / trajectory prediction (physics metrics,
    #   see benchmarks/physics_rollout_eval.py). Does NOT reduce hallucination.
    # Zero-regression: default False → head not built → forward bit-identical.
    use_hamiltonian_world_model: bool = False
    hamiltonian_loss_weight: float = 0.01    # small: LM loss always dominates
    hamiltonian_phase_dim: int = 32          # dim of each of q, p decoded from d_model
    hamiltonian_hidden: int = 64             # energy-MLP hidden width
    hamiltonian_dt: float = 0.1              # symplectic step size
    hamiltonian_condition_on_context: bool = True   # liquid core sets the potential landscape
    hamiltonian_context_dim: int = 32        # width of the context that conditions V

    # Causal consistency checker (Phase B). The checker is a stateless
    # inference-time monitor (not part of the model graph), but its defaults
    # live here so a single config drives the whole pipeline. Construct with
    # CausalConsistencyChecker.from_config(cfg).
    #   method="cosine"  : cheap, back-compat default. Saturates on anisotropic
    #                      hidden states (rarely fires on real breaks).
    #   method="subspace": principal-subspace residual, anisotropy-robust.
    #                      Recommended for real hidden states. Tuning tip: pair
    #                      with a higher consistency_floor (subspace scores sit
    #                      higher in stable regimes, drop sharply on a break).
    causal_check_method: str = "cosine"
    causal_check_window: int = 5
    causal_check_threshold: float = 0.3

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
        # RoPE pairs dimensions, so an odd d_head fails deep inside
        # RotaryEmbedding with a bare assert. Fail here with the reason.
        if self.d_head % 2 != 0:
            raise ValueError(
                f"d_head must be even for rotary embeddings; got d_head="
                f"{self.d_head} (d_model={self.d_model} / n_heads={self.n_heads}). "
                f"Pick n_heads so that d_model/n_heads is even."
            )
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

        # core_iterations: weight-tied latent depth must be a positive int.
        if int(self.core_iterations) < 1:
            raise ValueError(
                f"core_iterations must be >= 1; got {self.core_iterations}"
            )
        if int(self.stack_iterations) < 1:
            raise ValueError(
                f"stack_iterations must be >= 1; got {self.stack_iterations}"
            )
        if int(self.workspace_iterations) < 1:
            raise ValueError(
                f"workspace_iterations must be >= 1; got {self.workspace_iterations}"
            )
        if not (0 <= int(self.n_global_heads) <= int(self.n_heads)):
            raise ValueError(
                f"n_global_heads must be in [0, n_heads={self.n_heads}]; "
                f"got {self.n_global_heads}"
            )

        # attention_layers: None means all layers; otherwise a de-duplicated
        # tuple of valid indices. An EMPTY tuple is legal -- a pure-LNN stack --
        # and position tracking must then come from cache.token_count, never
        # from a layer-0 KV that does not exist.
        if self.attention_layers is not None:
            idx = tuple(int(i) for i in self.attention_layers)
            if len(set(idx)) != len(idx):
                raise ValueError(f"attention_layers has duplicates: {idx}")
            bad = [i for i in idx if not (0 <= i < self.n_layers)]
            if bad:
                raise ValueError(
                    f"attention_layers indices {bad} out of range for "
                    f"n_layers={self.n_layers}"
                )
            self.attention_layers = tuple(sorted(idx))

        # scale_gate_period constraint: values >4 risk τ-routing rigidity.
        if self.scale_gate_period not in (1, 2, 4):
            raise ValueError(
                f"scale_gate_period must be 1, 2, or 4; got {self.scale_gate_period}. "
                "Values >4 suppress multi-scale τ diversity across too many layers."
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
