# MT-LNN Technical Specification

**Version:** 1.1  
**Date:** 2026-05-12  
**Repo:** https://github.com/everest-an/M1

---

## 1. System Architecture & Biological Mapping / 系统架构与生物学映射

To make this specification accessible, we map the mathematical components directly to their biological inspirations before presenting the compute graph:
> *为了让这份技术规范更易读，我们在呈现计算图之前，先将晦涩的数学组件与它们所对应的生物学灵感进行直观的映射：*

*   **Microtubule Dynamic Layer (MT-DL) ⇔ Microtubules (细胞微管)**: Replaces standard dense networks with 13 parallel liquid time-constant channels, representing the 13 protofilaments making up a biological microtubule. *(将标准的密集网络层替换为 13 个平行的液态时间常数通道，这完全对应了构成生物微管的 13 根原纤维。)*
*   **Vectorized Multi-Scale Resonance ⇔ Brain wave frequencies (脑波频率动态)**: Each protofilament computes at 5 geometrical time-scales (from rapid spikes to slow drifts), capturing the temporal dynamics of biological neurons. *(每根原纤维在 5 个几何时间尺度上进行计算（从快速的脉冲到极慢的信息漂移），精准捕捉了生物神经元的时间动态特性。)*
*   **Lateral Coupling ⇔ B-Lattice Bonds (横向耦合与B晶格键)**: Forces information to interact with neighboring protofilaments instead of being independent. *(强制信息在相邻的原纤维之间进行交互，对应了生物微管结构中的 B晶格键横向作用力。)*
*   **GWP & Global Coherence ⇔ Global Workspace Theory (全局工作区与意识广播)**: Creates a central "bottleneck" that forces the model to synthesize disparate information streams into a single "conscious" broadcast vector. *(创建了一个集中的“瓶颈”，迫使模型将各个局部信息流合成为一个单一的、类似于“意识”的广播向量。)*

### 1.1 Forward pass (default config)

```
input_ids  (B, T)
  │
  ▼  MTLNNEmbedding
token_embed(input_ids)  →  x  (B, T, d_model)
  │
  ▼  × n_layers  [MTLNNBlock]
  │    ├─ pre-norm + MicrotubuleAttention  →  +residual
  │    └─ pre-norm + MTLNNLayer            →  +residual
  │       [returns (out, h_last) where h_last cached as h_prev]
  │
  ▼  GWTBLayer  (if gwtb_per_block=False)
compress  →  workspace SA  →  broadcast  →  +γ·residual
  │
  ▼  GlobalCoherenceLayer
sparse top-k causal SA  →  collapse gate  →  +residual
  │
  ▼  LayerNorm  →  lm_head (Linear, no bias, weight-tied)
  │
  ▼
logits  (B, T, vocab_size)
```

### 1.2 Inference cache (`ModelCacheStruct`)

```python
class ModelCacheStruct:
    layers: List[LayerCache]       # one per MTLNNBlock
    gwtb_kv: Optional[KVCache]     # top-level GWTB
    coherence_kv: Optional[KVCache]

LayerCache = Tuple[
    Optional[KVCache],       # [0] attention KV  (K, V)
    Optional[Tensor],        # [1] LNN h_prev     (B, P, D)
    Optional[KVCache],       # [2] per-block GWTB (K, V) or None
]
```

### 1.3 Two forward modes / 两种前向传播模式

| Mode | `use_lnn_recurrence` | Semantics |
|---|---|---|
| Training / prefill (训练或预填充) | `False` | h_prev = 0 for every token; parallel; cached decode matches within `1e-4` (FP reduction order) |
| Inference (default) (流式推理) | `True` | h_prev threaded across decode steps; true RNN memory |

---

## 2. Module Specifications / 核心模块规格

### 2.1 `MTLNNConfig` (`mt_lnn/config.py`)

Complete parameter reference: (完整参数配置表：)

| Parameter (参数) | Type (类型) | Default (默认值) | Description (描述) |
|---|---|---|---|
| `vocab_size` | int | 50257 | GPT-2 BPE vocab |
| `max_seq_len` | int | 1024 | Max sequence length; RoPE table size |
| `pad_token_id` | int | 0 | Padding token ID |
| `d_model` | int | **832** | Model width. Must satisfy `d_model % n_protofilaments == 0` after ceiling; best = 13k×64 |
| `n_layers` | int | 12 | Transformer depth |
| `n_heads` | int | **13** | Q heads. Default = n_protofilaments (1 head per protofilament) |
| `n_kv_heads` | int | **1** | KV heads (MQA). `n_heads % n_kv_heads == 0` required |
| `d_head` | int | 64 | `d_model // n_heads`. Must be set consistently |
| `n_protofilaments` | int | 13 | MT protofilament count. Biological constant; vectorised so P=64 ≈ free |
| `map_hidden_dim` | int | 64 | Hidden dim of per-protofilament MAP gate MLP |
| `tau_init` | float | 1.0 | Base time constant initial value |
| `tau_min` | float | 0.01 | Minimum τ after softplus + clamp |
| `tau_max` | float | 10.0 | Maximum τ after softplus + clamp |
| `dt` | float | 1.0 | Discrete timestep Δt for CfLTC update |
| `gamma_init` | float | 0.1 | Base GTP decay γ₀ for attention ALiBi schedule |
| `gtp_period` | int | 256 | GTP cap renewal period (tokens); lateral coupling resets every T_period |
| `n_time_scales` | int | 5 | Number of τ scales per protofilament (S in P×S LTC banks) |
| `resonance_freqs` | tuple\|None | None | Manual τ schedule; auto-computed as geometric sweep if None |
| `polarity_mode` | str | `"scalar"` | `"scalar"` or `"low_rank"` (bilinear polarity bias) |
| `polarity_rank` | int | 8 | Rank r for low-rank bilinear polarity (W_A, W_B ∈ ℝ^{d×r}) |
| `gwtb_compression_ratio` | int | 8 | d_gw = d_model // r |
| `gwtb_n_heads` | int | 4 | Heads in GWTB workspace self-attention |
| `gwtb_broadcast_init` | float | 0.01 | Initial value of gated broadcast scalar γ_bcast |
| `gwtb_per_block` | bool | False | If True, GWTB lives inside every block (vs. once after stack) |
| `coherence_sparsity` | float | 0.1 | Fraction of attention scores retained in GlobalCoherenceLayer |
| `coherence_heads` | int | 4 | Heads in GlobalCoherenceLayer |
| `dropout` | float | 0.1 | General dropout rate |
| `attention_dropout` | float | 0.1 | Attention weight dropout (SDPA `dropout_p`) |
| `tie_embeddings` | bool | True | Tie token embedding and lm_head weights |
| `d_proto` | int | *derived* | `ceil(d_model / n_protofilaments)` — auto-computed |
| `d_proto_total` | int | *derived* | `d_proto × n_protofilaments` — auto-computed; may exceed d_model by up to P-1 |

**Tensor-Core alignment rule:**  
Choose `d_model = n_protofilaments × k × 8` for some integer k. Default: 832 = 13 × 8 × 8.

---

### 2.2 `MTLNNEmbedding` (`mt_lnn/embedding.py`)

```
Input:  input_ids  (B, T)
Output: x          (B, T, d_model)

Components:
  token_embed: nn.Embedding(vocab_size, d_model)
  rope:        RotaryEmbedding(d_head, max_seq_len)
  dropout:     nn.Dropout(dropout)
```

`RotaryEmbedding.forward(x, position_offset)` applies RoPE at absolute positions `[offset, offset+T)`. Buffers `cos_table` and `sin_table` are precomputed in `__init__`; no allocation per forward call.

---

### 2.3 `MicrotubuleAttention` (`mt_lnn/mt_attention.py`)

```
Input:  x               (B, T_new, d_model)
        pad_mask        (B, T_total) bool, optional
        past_kv         ((K, V) tuple) or None
        position_offset int
        use_cache       bool
Output: out             (B, T_new, d_model)
        new_kv          (K, V) or None
```

**Attention bias formula** (applied as additive logit bias to SDPA):

```
bias[h, i, j] = polarity_h × (j - i) / L          [scalar polarity]
              + gate_h × σ(xWₐ)(xW_b)ᵀ[i,j]       [low-rank bilinear, if enabled]
              - γ_h × max(i - j, 0)                 [GTP log-decay / ALiBi]
              + −∞                                   [if j > i (causal) or j is pad]
```

**γ initialisation:**
```
γ_h = γ₀ × 2^(linspace(+3, −3, H))    [geometric; head 0 = local, head H-1 = global]
γ stored in raw (softplus⁻¹) space; actual γ = softplus(raw_γ)
```

**KV cache:** K/V stored with shape `(B, n_kv_heads, T_total, d_head)`. GQA repeats K/V heads `n_rep = n_heads // n_kv_heads` times before SDPA. Distance buffers `_delta` and `_causal` precomputed as buffers of size `(max_seq_len, max_seq_len)`.

---

### 2.4 `MTLNNLayer` (`mt_lnn/mt_lnn_layer.py`)

```
Input:  x        (B, T, d_model)
        h_prev   (B, P, D) or None
        position_offset  int
Output: out      (B, T, d_model)
        h_last   (B, P, D)   ← recurrent state for next call
```

**Internal pipeline:**

```
x  →  in_proj  →  x_split  (B, T, P, D)
                    │
                    ▼  VectorizedMultiScaleResonance
                 h_stack   (B, T, P, D)
                 [P × S CfLTC banks; einsum; blend via softmax(blend_weights)]
                    │
                    ▼  LateralCoupling + GTP gate
                 h_coupled (B, T, P, D)
                 [W_lat static + torch.roll NN + σ(rmc_gate)×RMC]
                 [× exp(-γ × (t mod T_period))]
                    │
                    ▼  VectorizedMAPGate
                 h_gated   (B, T, P, D)
                 [P parallel 2-layer MLPs; fc2_bias=+2 init]
                    │
                    ▼  reshape + out_proj + dropout
                 out       (B, T, d_model)
                 h_last = h_gated[:, -1, :, :]
```

**CfLTC update equation (per protofilament p, scale s):**

```
decay_{p,s}  = exp(-dt / τ_{p,s})
A_{b,t,p,s}  = σ(W_in[p,s] @ x_split[b,t,p] + b_in[p,s])
h_{p,s}      = h_prev[p] × decay + A × (1 − decay)
h_blended[p] = Σ_s softmax(blend_weights[p])_s × h_{p,s}
```

**Lateral coupling (three contributions, summed):**

```
residual = einsum("btpd,pq->btqd", h, W_lat)                   # static
nearest  = η × tanh(W_L(roll(h,+1)) + W_R(roll(h,−1)))        # NN ring
rmc      = σ(rmc_gate) × SDPA(Q=q_proj(h), K=k_proj(h), V=v_proj(h))
h_lateral = residual + nearest + rmc
h_coupled = h + exp(-γ × (t mod T_period)) × (h_lateral − h)
```

---

### 2.5 `GWTBLayer` (`mt_lnn/gwtb.py`)

```
Input:  x           (B, T_new, d_model)
        past_kv     KVCache or None
        position_offset  int
        use_cache   bool
Output: x_out       (B, T_new, d_model)
        new_kv      KVCache or None
```

**Three-step pipeline:**

```
1. Compression:
   z = LN(W_compress @ x)          # (B, T, d_gw);  d_gw = d_model // r

2. Workspace SA:
   Q, K, V = q_proj(z), k_proj(z), v_proj(z)
   [K/V extended with past_kv if present]
   z_attn = SDPA(Q, K, V, causal_mask)   # (B, H_gw, T, d_gw/H_gw)
   z' = LN(z + attn_out_proj(z_attn))

3. Broadcast:
   Δh = W_broadcast @ z'           # (B, T, d_model)
   x_out = x + broadcast_gate × Δh
   [broadcast_gate: scalar param, init=gwtb_broadcast_init=0.01]
```

---

### 2.6 `GlobalCoherenceLayer` (`mt_lnn/global_coherence.py`)

```
Input:  x           (B, T_new, d_model)
        past_kv     KVCache or None
        position_offset  int
        use_cache   bool
Output: x_out       (B, T_new, d_model)
        new_kv      KVCache or None
```

**Mechanism:**

```
scores = (Q @ K.T) / sqrt(d_head)      # (B, H, T_q, T_k)
scores = causal_mask(scores)
scores = sparse_top_k(scores, k=ceil(T_k × sparsity))

# Collapse gate (Orch-OR)
mean_energy = mean(raw_scores[causal_positions])
gate = σ((mean_energy − collapse_threshold) × 10)   # ∈ (0, 1)
model.coherence.last_gate = gate   # diagnostic buffer

attn = softmax(scores)
out = attn @ V
coherence_out = coherence_scale × gate × out_proj(out)
x_out = LN(x + coherence_out)
```

---

### 2.7 `AnesthesiaController` (`mt_lnn/anesthesia.py`)

```python
# Attach
ctrl = AnesthesiaController().attach_to(model)
ctrl.set(0.7)           # level ∈ [0.0, 1.0]

# Context manager
with anesthetize(model, 0.7):
    logits = model(ids)["logits"]
```

**Hook effects at level ℓ:**

| Module (模块) | Effect (效果) |
|---|---|
| `MTLNNLayer` (post-hook) | `out *= (1 - ℓ);  h_last *= (1 - ℓ)` |
| `GlobalCoherenceLayer` (post-hook) | `x_out = x_in + (1 - ℓ)(x_out - x_in)` |

No weight modification; hooks detach automatically at context exit.

---

### 2.8 `phi_hat.py`

**`knn_entropy_chebyshev(X, k=3) → float`**

KSG estimator, L∞ metric:
```
Ĥ(X) = −ψ(k) + ψ(N) + d × ⟨log(2 × ε_i)⟩
```
where ε_i = L∞ distance to k-th nearest neighbour. O(N²d) memory; N ≤ 2048.

**`compute_phi_hat(hidden, K=4, k_nn=3) → float`**
```
Φ̂(h) = Σ_k Ĥ(s_k) − Ĥ(h)      [K-way contiguous partition]
```

**`compute_phi_hat_from_model(model, input_ids, K=4, k_nn=3, n_batches=10) → float`**  
Runs `n_batches` forward passes (first uses provided ids, rest random); averages Φ̂ estimates. Handles `gwtb_per_block=True/False` automatically.

**`phi_hat_anesthesia_sweep(model, ids, kappas, K, k_nn) → Dict[float, float]`**  
κ → ℓ = (κ − 1) / 9; runs sweep; returns {κ: Φ̂}.

**`anesthesia_test_result(sweep, delta=0.7) → dict`**  
```
passed = Φ̂(κ_max) / Φ̂(κ=1) ≤ 1 − δ
```
Returns `{phi_clean, phi_full, ratio, collapse_pct, delta_threshold, passed}`.

---

## 3. Initialisation Protocol / 初始化协议

Applied by `init_mt_params(model, config)` after standard weight init:

| Parameter (参数) | Init value (初始值) | Reason (原因) |
|---|---|---|
| `polarity_direction` | Uniform(−0.05, 0.05) | Near-symmetric at start |
| `W_lat` | Identity + N(0, 0.005) | Protofilaments start independent |
| `rmc_gate` | −3.0 | σ(−3) ≈ 0.05; RMC near-off at init |
| `coherence_scale` | 0.05 | Near-identity global coherence |
| `collapse_threshold` | 0.5 | Centred gate |
| `blend_weights` | 0.0 | softmax(0) = uniform over S scales |
| `gtp_gamma` (LNN) | `gamma_init` | Moderate lateral decay |
| `lnn.out_proj.weight` | N(0, 0.01) | Small residual for training stability |
| `gwtb.broadcast_gate` | `gwtb_broadcast_init = 0.01` | GWTB near-identity at start |
| `VectorizedMAPGate.fc2_bias` | +2.0 | σ(2) ≈ 0.88; gates near-open |

---

## 4. Training Specification / 训练规格

### 4.1 Optimiser / 优化器配置

```python
# Four separate LR groups (make_param_groups in utils.py)
{"params": main_params,     "lr": base_lr,          "weight_decay": 0.1}
{"params": ode_params,      "lr": base_lr × 0.33,   "weight_decay": 0.0}  # log_tau, gtp_gamma
{"params": polarity_params, "lr": base_lr × 1.67,   "weight_decay": 0.0}  # polarity_direction
{"params": lateral_params,  "lr": base_lr × 0.33,   "weight_decay": 0.01} # W_lat

optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
```

### 4.2 Schedule / 调度器

```
Linear warmup: steps 0 → warmup_steps  (default 2000)
Cosine decay:  steps warmup → total    (min_lr = 0.1 × peak_lr)
Peak LR:  6e-4 (125M)
```

### 4.3 Default training config (125M) / 默认预训练流超参

| Parameter | Value |
|---|---|
| `d_model` | 832 |
| `n_layers` | 12 |
| `n_heads` | 13, `n_kv_heads`=1 |
| `seq_len` | 512 |
| `batch` | 8 |
| `grad_accum` | 64 |
| `global_batch` | 512 sequences = ~262K tokens |
| `steps` | 50,000 |
| `grad_clip` | 1.0 |
| `dropout` | 0.1 |
| `precision` | BF16 (A100) / FP16 (fallback) |

### 4.4 Data format / 喂表数据格式

```
prepare_data.py → data/train.bin, data/validation.bin, data/meta.json
Format: flat uint16 token stream (numpy.memmap)
meta.json: {"vocab_size": int, ...}
```

`BinDataset.__getitem__` returns a random-offset window within each stride bucket — mild data augmentation without full shuffle.

---

## 5. Evaluation Specification / 评测规格

### 5.1 Perplexity

```bash
python eval.py --ckpt checkpoints/final.pt --eval_data \
               --dataset wikitext --dataset_config wikitext-103-raw-v1 \
               --tokenizer gpt2 --batch 8
```

Outputs: standard PPL at training seq_len.

### 5.2 Long-context PPL

```bash
python eval.py --ckpt checkpoints/final.pt --eval_data \
               --long_ctx 2048 4096 --chunk_size 256
```

Uses sliding-window decoding with KV cache + h_prev threading.

### 5.3 MT diagnostics

```bash
python eval.py --ckpt checkpoints/final.pt --diagnostics
```

Outputs: τ mean/std/min/max, γ mean, polarity std, W_lat off-diag norm, rmc_gate mean, coherence_scale, collapse_threshold, collapse_gate_last.

### 5.4 W_lat heatmaps

```bash
python eval.py --ckpt checkpoints/final.pt --heatmap_dir analysis/
```

Renders one PNG per layer showing the 13×13 W_lat coupling matrix.

### 5.5 Anesthesia Validation Protocol

```bash
python eval.py --ckpt checkpoints/final.pt \
               --anesthesia_test \
               --anesthesia_kappas 1 2 5 10 \
               --anesthesia_delta 0.7 \
               --phi_K 4 --phi_k_nn 3 --phi_batch 512
```

Outputs: Φ̂ at each κ, collapse %, pass/fail verdict.

### 5.6 Generation demo

```bash
python demo.py --ckpt checkpoints/final.pt \
               --prompt "The human brain" \
               --temperature 0.8 --top_p 0.9 --max_tokens 200
```

Uses prefill + incremental decode with dual cache; streams output token by token.

---

## 6. Test Suite / 单元测试套件

```bash
python tests/test_model.py          # run all 17 tests
python -m pytest tests/             # pytest runner
```

Tests require no GPU and complete in < 2 minutes on CPU. Small config used:  
`d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, vocab_size=200, dropout=0.0`.

---

## 7. File Map / 代码文件树映射

```
E:\O1\
├── mt_lnn/
│   ├── config.py           MTLNNConfig dataclass
│   ├── embedding.py        TokenEmbedding + RotaryEmbedding
│   ├── mt_attention.py     MicrotubuleAttention (GQA, KV cache, polarity bias)
│   ├── mt_lnn_layer.py     VectorizedMultiScaleResonance, LateralCoupling,
│   │                       VectorizedMAPGate, MTLNNLayer
│   ├── gwtb.py             GWTBLayer (compress → workspace SA → broadcast)
│   ├── global_coherence.py GlobalCoherenceLayer (sparse top-k + collapse gate)
│   ├── anesthesia.py       AnesthesiaController, anesthetize()
│   ├── phi_hat.py          knn_entropy_chebyshev, compute_phi_hat,
│   │                       phi_hat_anesthesia_sweep, anesthesia_test_result
│   ├── model.py            MTLNNBlock, MTLNNModel, ModelCacheStruct
│   └── utils.py            init_weights, init_mt_params, WarmupCosineScheduler,
│                           save/load_checkpoint, make_param_groups
├── prepare_data.py         Tokenise dataset → uint16 .bin files
├── train.py                Full training loop (AMP, torch.compile, W&B)
├── eval.py                 PPL, long-ctx PPL, diagnostics, AVP CLI
├── demo.py                 Streaming KV-cached generation
├── tests/test_model.py     17-test suite
├── PRD.md                  This product requirements document
├── SPEC.md                 This technical specification
├── mt_lnn_arxiv.md         Arxiv paper (Markdown, most current)
├── mt_lnn_arxiv.tex        Arxiv paper (LaTeX, for submission)
├── README.md               Quick-start guide + architecture + inspiration
└── requirements.txt        Python dependencies
```

---

## 8. Known Limitations / 已知架构限制

| Limitation (限制点) | Mitigation (缓解/对策) |
|---|---|
| Low-rank bilinear polarity skips past tokens in cached decode (only new×new block computed) | Acceptable approximation; scalar polarity covers all positions |
| Φ̂ estimator can be negative for small N or random-init models | Use n_batches=10; use trained model; interpret sign relative to baseline |
| GTP-γ modulation not implemented in anesthesia hooks (only output damping) | Two implemented effects sufficient for anesthesia test; γ modification unsafe under torch.compile |
| d_proto_total may exceed d_model by up to P-1 tokens | Handled by ceiling in config.__post_init__; slight parameter overhead |
| torch.compile strips module names; use `_orig_mod` to access diagnostics | eval.py and train.py already do `getattr(model, "_orig_mod", model)` |
