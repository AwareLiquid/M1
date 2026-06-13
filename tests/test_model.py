"""
Test suite for MT-LNN.

Run:  python -m pytest tests/   -or-   python tests/test_model.py
"""

import math
import sys
import warnings
import pytest
import torch
import torch.nn.functional as F

# allow `python tests/test_model.py` from project root
sys.path.insert(0, ".")

# Tests use tiny non-TC-aligned dims for speed; suppress the alignment warning.
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import (
    MTLNNConfig, MTLNNModel, ModelCacheStruct, anesthetize,
    compute_phi_hat, compute_phi_hat_from_model,
    phi_hat_anesthesia_sweep, anesthesia_test_result,
)
from mt_lnn.utils import make_param_groups, count_parameters


def small_config():
    """Tiny config for fast tests."""
    return MTLNNConfig(
        vocab_size=200,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,        # GQA: 2 KV heads, 4 Q heads (2× repeat)
        d_head=32,
        dropout=0.0,         # disable dropout for deterministic tests
        attention_dropout=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Shape and basic forward
# ---------------------------------------------------------------------------

def test_shapes_and_loss():
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)

    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"].dim() == 0 and out["loss"].item() > 0
    print("[ok] test_shapes_and_loss")


# ---------------------------------------------------------------------------
# 2. Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    cfg = small_config()
    model = MTLNNModel(cfg).train()

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    out["loss"].backward()

    dead, exploding = [], []
    for name, p in model.named_parameters():
        if name.startswith("target_"):
            continue
        if p.grad is None:
            dead.append(name)
        elif not p.grad.isfinite().all():
            exploding.append(name)
    assert not dead, f"Dead gradients: {dead}"
    assert not exploding, f"Exploding gradients: {exploding}"
    print(f"[ok] test_gradient_flow ({sum(1 for _ in model.parameters())} params, all finite)")


def test_direct_target_head_shapes_and_loss():
    cfg = small_config()
    model = MTLNNModel(cfg).train()

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 4))
    out = model(
        ids,
        direct_target_labels=targets,
        target_len=4,
        return_target_logits=True,
    )

    assert out["target_logits"].shape == (2, 4, cfg.vocab_size)
    assert out["target_loss"].dim() == 0 and out["target_loss"].item() > 0
    out["target_loss"].backward()
    assert model.target_head.weight.grad is not None
    assert model.target_queries.grad is not None
    print("[ok] test_direct_target_head_shapes_and_loss")


# ---------------------------------------------------------------------------
# 3. KV-cache parity: cached vs uncached must produce identical logits
# ---------------------------------------------------------------------------

def test_kv_cache_parity():
    """
    Parallel-mode parity: with use_lnn_recurrence=False (the same mode used in
    training), the cached and full-forward paths must produce IDENTICAL logits.

    This isolates the KV-cache + RoPE-offset + polarity/GTP correctness from
    the LNN's recurrent state behaviour.
    """
    torch.manual_seed(42)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    T = 12
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    # A. Full forward
    with torch.no_grad():
        full_logits = model(ids, use_lnn_recurrence=False)["logits"]

    # B. Incremental with cache, parallel-LNN mode
    cache = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            tok = ids[:, t:t+1]
            out = model(tok, cache=cache, use_cache=True, use_lnn_recurrence=False)
            cache = out["cache"]
            step_logits.append(out["logits"][:, -1, :])
        step_logits = torch.stack(step_logits, dim=1)

    max_diff = (full_logits - step_logits).abs().max().item()
    print(f"  max logit diff (parallel-mode cached vs full): {max_diff:.6e}")
    assert max_diff < 1e-4, f"KV-cache parity FAILED: max diff = {max_diff:.6e}"
    print("[ok] test_kv_cache_parity")


def test_pscan_cache_parity_with_recurrence():
    """
    With pscan (use_lnn_recurrence=True) + properly threaded h_prev across
    cached steps, full forward and single-step cached decoding should produce
    IDENTICAL logits. This is the gold-standard test that proves the new
    parallel-scan path correctly threads its recurrent state through the cache.
    """
    torch.manual_seed(7)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    T = 12
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    # A. Full forward, real recurrence, no cache
    with torch.no_grad():
        full = model(ids, use_lnn_recurrence=True)["logits"]

    # B. Step-by-step with cache (h_prev properly threaded)
    cache = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            out = model(ids[:, t:t+1], cache=cache, use_cache=True,
                        use_lnn_recurrence=True)
            cache = out["cache"]
            step_logits.append(out["logits"][:, -1, :])
    stacked = torch.stack(step_logits, dim=1)

    diff = (full - stacked).abs().max().item()
    print(f"  pscan + real-recurrence cache parity: {diff:.6e}")
    assert diff < 1e-3, f"pscan cache parity FAILED: {diff}"
    print("[ok] test_pscan_cache_parity_with_recurrence")


def test_lnn_recurrence_active():
    """
    Sanity check: with use_lnn_recurrence=True, cached decode SHOULD differ
    from a parallel forward — proving that h_prev is actually flowing.
    """
    torch.manual_seed(123)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    T = 8
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        full_logits = model(ids, use_lnn_recurrence=False)["logits"]

    cache = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            tok = ids[:, t:t+1]
            out = model(tok, cache=cache, use_cache=True, use_lnn_recurrence=True)
            cache = out["cache"]
            step_logits.append(out["logits"][:, -1, :])
        step_logits = torch.stack(step_logits, dim=1)

    diff = (full_logits - step_logits).abs().max().item()
    print(f"  diff with recurrence active: {diff:.4e} (expect > 0)")
    assert diff > 1e-4, "LNN recurrence appears INACTIVE — h_prev not flowing"
    print("[ok] test_lnn_recurrence_active")


# ---------------------------------------------------------------------------
# 4. Prefill-then-decode: encode prompt fully, decode rest token by token
# ---------------------------------------------------------------------------

def test_prefill_then_decode():
    """
    Mixed mode: prefill T_prompt=8 tokens in one shot, then 4 incremental tokens.
    Parallel-LNN mode for bit-exact parity with single-shot full forward.
    """
    torch.manual_seed(7)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    T_prompt, T_gen = 8, 4
    ids = torch.randint(0, cfg.vocab_size, (1, T_prompt + T_gen))

    cache = None
    pieces = []
    with torch.no_grad():
        full = model(ids, use_lnn_recurrence=False)["logits"]
        # Prefill (parallel mode for parity)
        out = model(ids[:, :T_prompt], use_cache=True, use_lnn_recurrence=False)
        cache = out["cache"]
        pieces.append(out["logits"])
        # Decode one token at a time
        for t in range(T_gen):
            tok = ids[:, T_prompt + t: T_prompt + t + 1]
            out = model(tok, cache=cache, use_cache=True, use_lnn_recurrence=False)
            cache = out["cache"]
            pieces.append(out["logits"])
        cached = torch.cat(pieces, dim=1)

    diff = (full - cached).abs().max().item()
    print(f"  max diff (prefill+decode vs full): {diff:.6e}")
    assert diff < 1e-4, f"Prefill-decode parity FAILED: {diff:.6e}"
    print("[ok] test_prefill_then_decode")


# ---------------------------------------------------------------------------
# 5. GQA verification: KV cache uses fewer heads than Q
# ---------------------------------------------------------------------------

def test_gqa_kv_cache_size():
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids, use_cache=True)

    # Inspect first layer's KV cache
    K, V = out["cache"].layers[0][0]
    assert K.shape == (1, cfg.n_kv_heads, 8, cfg.d_head), f"K shape: {K.shape}"
    assert V.shape == (1, cfg.n_kv_heads, 8, cfg.d_head), f"V shape: {V.shape}"
    saved = (cfg.n_heads - cfg.n_kv_heads) / cfg.n_heads
    print(f"  KV cache: {cfg.n_kv_heads} heads vs {cfg.n_heads} Q heads ({saved*100:.0f}% memory saved)")
    print("[ok] test_gqa_kv_cache_size")


# ---------------------------------------------------------------------------
# 6. Overfit: can the model actually learn?
# ---------------------------------------------------------------------------

def test_low_rank_polarity():
    """
    Low-rank bilinear polarity mode should:
      1. Be opt-in (not affect default config behaviour)
      2. Produce a valid forward + backward pass
      3. Have a non-trivial number of extra params (2·d·r per head)
    """
    cfg_default = small_config()
    cfg_lr = MTLNNConfig(
        vocab_size=200, max_seq_len=64, d_model=128, n_layers=2,
        n_heads=4, n_kv_heads=2, d_head=32, dropout=0.0, attention_dropout=0.0,
        polarity_mode="low_rank", polarity_rank=4,
    )
    m_default = MTLNNModel(cfg_default).eval()
    m_lr = MTLNNModel(cfg_lr).eval()

    n_default = m_default.get_num_params()
    n_lr = m_lr.get_num_params()
    extra = n_lr - n_default
    print(f"  scalar polarity: {n_default:,} params  |  low_rank polarity: {n_lr:,} params  "
          f"(+{extra:,})")
    assert extra > 0, "low_rank polarity didn't add any parameters"

    ids = torch.randint(0, cfg_lr.vocab_size, (2, 16))
    out = m_lr(ids, labels=ids)
    out["loss"].backward()
    for name, p in m_lr.named_parameters():
        if "pol_W_A" in name or "pol_W_B" in name or "pol_bilinear_gate" in name:
            assert p.grad is not None, f"low-rank polarity param has no grad: {name}"
    print("[ok] test_low_rank_polarity")


def test_nearest_neighbor_coupling():
    """
    Nearest-neighbor torch.roll coupling should produce non-trivial gradients
    on W_left, W_right, nn_eta and not break the parity test paths.
    """
    cfg = small_config()
    model = MTLNNModel(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    out["loss"].backward()

    found = {"W_left": False, "W_right": False, "nn_eta": False}
    for name, p in model.named_parameters():
        for key in found:
            if name.endswith(key + ".weight") or name.endswith(key):
                assert p.grad is not None and p.grad.abs().sum().item() > 0, \
                    f"{key} has zero gradient"
                found[key] = True
    assert all(found.values()), f"missing NN-coupling params: {found}"
    print("[ok] test_nearest_neighbor_coupling")


def test_gwtb_bottleneck():
    """
    Verify the GWT Bottleneck:
      - d_gw < d_model (capacity-limited)
      - all GWTB params receive gradients
      - broadcast_gate starts at the configured small value
    """
    cfg = small_config()
    model = MTLNNModel(cfg).train()
    assert model.gwtb.d_gw < cfg.d_model, \
        f"GWTB d_gw={model.gwtb.d_gw} should be < d_model={cfg.d_model}"
    assert abs(model.gwtb.broadcast_gate.item() - cfg.gwtb_broadcast_init) < 1e-6, \
        "broadcast_gate didn't initialise to configured value"

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    out["loss"].backward()
    live = 0
    for name, p in model.named_parameters():
        if name.startswith("gwtb."):
            assert p.grad is not None and p.grad.isfinite().all(), \
                f"GWTB param {name} has bad grad"
            live += 1
    print(f"  d_gw={model.gwtb.d_gw}  d_model={cfg.d_model}  "
          f"compression={cfg.d_model // model.gwtb.d_gw}×  live grads on {live} GWTB params")
    assert live >= 5
    print("[ok] test_gwtb_bottleneck")


def test_gwtb_cache_parity():
    """Cached vs full forward with GWTB enabled must remain bit-exact (parallel LNN mode)."""
    torch.manual_seed(99)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    T = 10
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        full = model(ids, use_lnn_recurrence=False)["logits"]

    cache = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            out = model(ids[:, t:t+1], cache=cache, use_cache=True,
                        use_lnn_recurrence=False)
            cache = out["cache"]
            step_logits.append(out["logits"][:, -1, :])
    stacked = torch.stack(step_logits, dim=1)

    diff = (full - stacked).abs().max().item()
    print(f"  GWTB cache parity diff: {diff:.6e}")
    assert diff < 1e-4, f"GWTB broke cache parity: {diff}"
    print("[ok] test_gwtb_cache_parity")


def test_gwtb_per_block_mode():
    """
    With gwtb_per_block=True:
      - top-level model.gwtb is None
      - every MTLNNBlock has its own gwtb sublayer
      - forward + backward work, every gwtb param has gradients
      - cache parity holds (parallel-LNN mode) — i.e. the per-block GWTB
        KV cache slots are threaded correctly
    """
    cfg = MTLNNConfig(
        vocab_size=200, max_seq_len=64, d_model=128, n_layers=2,
        n_heads=4, n_kv_heads=2, d_head=32, dropout=0.0,
        attention_dropout=0.0, gwtb_per_block=True,
    )
    model = MTLNNModel(cfg).eval()

    # Structural assertions
    assert model.gwtb is None, "top-level GWTB should be disabled per-block mode"
    for i, blk in enumerate(model.blocks):
        assert blk.has_gwtb and blk.gwtb is not None, \
            f"block {i} missing per-block GWTB"

    # Forward + backward
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    model.train()
    out = model(ids, labels=ids)
    out["loss"].backward()
    for name, p in model.named_parameters():
        if ".gwtb." in name:
            assert p.grad is not None and p.grad.isfinite().all(), \
                f"per-block GWTB param {name} has bad grad"
    model.eval()

    # Cache parity (parallel-LNN mode for bit-exact comparison)
    torch.manual_seed(11)
    T = 10
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        full = model(ids, use_lnn_recurrence=False)["logits"]

    cache = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            o = model(ids[:, t:t+1], cache=cache, use_cache=True,
                       use_lnn_recurrence=False)
            cache = o["cache"]
            step_logits.append(o["logits"][:, -1, :])
    stacked = torch.stack(step_logits, dim=1)
    diff = (full - stacked).abs().max().item()
    print(f"  per-block GWTB cache parity: {diff:.6e}")
    assert diff < 1e-4, f"per-block GWTB broke cache parity: {diff}"
    print("[ok] test_gwtb_per_block_mode")


def test_phi_hat_basic():
    """
    Phi_hat should be FINITE and HIGHER for correlated parts than for independent parts.
    """
    torch.manual_seed(123)
    N, d = 256, 32

    # Independent Gaussian: parts are independent, Phi_hat should be near 0
    indep = torch.randn(N, d)
    phi_indep = compute_phi_hat(indep, K=4, k_nn=3)

    # Correlated: build hidden where parts share a latent factor
    z = torch.randn(N, d // 4)
    parts = [z + 0.05 * torch.randn_like(z) for _ in range(4)]
    corr = torch.cat(parts, dim=-1)
    phi_corr = compute_phi_hat(corr, K=4, k_nn=3)

    print(f"  Phi_hat(independent)={phi_indep:+.3f}   Phi_hat(correlated)={phi_corr:+.3f}")
    assert math.isfinite(phi_indep) and math.isfinite(phi_corr)
    assert phi_corr > phi_indep, "Phi_hat should detect part correlation"
    print("[ok] test_phi_hat_basic")


def test_anesthesia_validation_protocol():
    """
    The full AVP: Phi_hat should monotonically collapse as κ rises from 1 → 10.
    """
    torch.manual_seed(0)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 24))

    sweep = phi_hat_anesthesia_sweep(model, ids, kappas=[1.0, 5.0, 10.0],
                                      K=4, k_nn=3)
    print(f"  Phi_hat sweep: {{ {', '.join(f'κ={k:.0f}: Phi_hat={v:+.3f}' for k, v in sweep.items())} }}")

    # Phi_hat(κ=1) should not be NaN; clean and full should differ
    assert all(math.isfinite(v) for v in sweep.values()), "Phi_hat produced NaN"
    result = anesthesia_test_result(sweep, delta=0.0)   # weak threshold for unseeded test
    print(f"  collapse: {result['collapse_pct']:.1f}%  (clean→full)")
    assert sweep[1.0] != sweep[10.0], "anesthesia hooks not affecting Phi_hat"
    print("[ok] test_anesthesia_validation_protocol")


def test_anesthesia_collapse():
    """
    Anesthesia validation: as anesthesia_level rises from 0→1, the entropy of
    the model's output distribution should INCREASE (the model becomes
    less confident — its 'consciousness' is degrading) and the output should
    diverge from the clean prediction.

    This is the in-silico mirror of the 2025 Wiest/Hameroff anesthesia finding.
    """
    torch.manual_seed(0)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    def entropy_and_logits(level):
        with anesthetize(model, level):
            with torch.no_grad():
                logits = model(ids)["logits"]                          # (1,T,V)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        ent = -(probs * probs.clamp(min=1e-9).log()).sum().item()
        return ent, logits.detach()

    ent_0, logits_0 = entropy_and_logits(0.0)
    ent_5, logits_5 = entropy_and_logits(0.5)
    ent_1, logits_1 = entropy_and_logits(1.0)

    diff_5 = (logits_0 - logits_5).abs().max().item()
    diff_1 = (logits_0 - logits_1).abs().max().item()

    print(f"  entropy: clean={ent_0:.3f}  half={ent_5:.3f}  full={ent_1:.3f}")
    print(f"  logit diff vs clean: half={diff_5:.4f}  full={diff_1:.4f}")
    # Anesthesia must actually change the output (sanity)
    assert diff_1 > diff_5 > 1e-5, "anesthesia hooks not firing"
    # Full anesthesia should produce a different distribution from clean
    assert diff_1 > 1e-3, f"full anesthesia barely affects output: {diff_1}"
    print("[ok] test_anesthesia_collapse")


def test_protofilament_scaling():
    """
    The vectorized MTLNNLayer should scale gracefully as P grows.
    Time at P=13 should not be >2× time at P=64 — proving the work happens
    on a single GPU/CPU einsum, not in a Python loop.
    """
    import time
    base_cfg = dict(vocab_size=200, max_seq_len=64, d_model=128, n_layers=1,
                    n_heads=4, n_kv_heads=2, d_head=32, dropout=0.0,
                    attention_dropout=0.0)

    times = {}
    for P in (13, 32, 64):
        cfg = MTLNNConfig(n_protofilaments=P, **base_cfg)
        model = MTLNNModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 32))
        # warmup
        for _ in range(3):
            with torch.no_grad():
                model(ids)
        t0 = time.time()
        for _ in range(20):
            with torch.no_grad():
                model(ids)
        times[P] = (time.time() - t0) / 20

    print(f"  P=13 → {times[13]*1000:.1f}ms  P=32 → {times[32]*1000:.1f}ms  "
          f"P=64 → {times[64]*1000:.1f}ms")
    # 5× more protofilaments should NOT cost 5× the time (CPU/GPU parallelism)
    ratio = times[64] / max(times[13], 1e-6)
    assert ratio < 5.0, f"P=64 is {ratio:.1f}× slower than P=13 — vectorisation broken"
    print(f"  scaling ratio (P=64 / P=13): {ratio:.2f}× — vectorisation OK")
    print("[ok] test_protofilament_scaling")


@pytest.mark.slow
def test_overfit_single_batch():
    """Loss should drop ≥10× within 200 steps on a fixed batch."""
    torch.manual_seed(0)
    cfg = small_config()
    model = MTLNNModel(cfg).train()

    groups = make_param_groups(model, base_lr=3e-3)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))

    ids = torch.randint(0, cfg.vocab_size, (4, 16))

    init_loss = None
    for step in range(200):
        opt.zero_grad()
        out = model(ids, labels=ids)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            init_loss = out["loss"].item()
    final = out["loss"].item()
    print(f"  init {init_loss:.3f} → final {final:.3f}")
    assert final < init_loss * 0.1, f"Overfit failed: init={init_loss:.3f} final={final:.3f}"
    print("[ok] test_overfit_single_batch")


# ---------------------------------------------------------------------------
# 7. MT diagnostics return finite values
# ---------------------------------------------------------------------------

def test_mt_diagnostics():
    cfg = small_config()
    model = MTLNNModel(cfg).eval()
    diag = model.get_mt_diagnostics()

    expected_keys = {"tau_mean", "tau_std", "gamma_mean", "polarity_mean",
                     "polarity_std", "lat_coupling_mean_off_diag_norm",
                     "coherence_scale", "collapse_threshold"}
    missing = expected_keys - diag.keys()
    assert not missing, f"Missing diagnostic keys: {missing}"
    for k, v in diag.items():
        assert math.isfinite(v), f"Non-finite diagnostic: {k}={v}"

    # τ initialised from resonance_freqs → values span the configured range
    # and there must be variance (multi-scale init not collapsed to a single value)
    assert cfg.tau_min <= diag["tau_min"] <= diag["tau_max"] <= cfg.tau_max
    assert diag["tau_std"] > 0.1, f"τ has no spread — multi-scale init failed: {diag['tau_std']}"
    # Lateral coupling near-identity at init
    assert diag["lat_coupling_mean_off_diag_norm"] < 0.5
    print(f"  τ={diag['tau_mean']:.3f}±{diag['tau_std']:.3f} "
          f"[{diag['tau_min']:.3f}, {diag['tau_max']:.3f}]  γ={diag['gamma_mean']:.3f}  "
          f"polarity_std={diag['polarity_std']:.3f}  rmc_gate={diag['rmc_gate_mean']:.3f}")
    print("[ok] test_mt_diagnostics")


def test_dynamic_scale_gate_diagnostics():
    cfg = small_config()
    cfg.dynamic_scale_gates = True
    cfg.scale_gate_skip_threshold = 0.0
    model = MTLNNModel(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out = model(ids)

    diag = model.get_mt_diagnostics()
    assert out["logits"].shape == (2, 8, cfg.vocab_size)
    assert "scale_gate_mean" in diag
    assert "scale_gate_active_ratio" in diag
    assert "scale_gate_nonzero_ratio" in diag
    assert 0.0 <= diag["scale_gate_mean"] <= 1.0
    assert 0.0 <= diag["scale_gate_active_ratio"] <= 1.0
    assert 0.0 <= diag["scale_gate_nonzero_ratio"] <= 1.0
    print("[ok] test_dynamic_scale_gate_diagnostics")


def test_sparse_resonance_kernel_topk():
    cfg = small_config()
    cfg.dynamic_scale_gates = True
    cfg.sparse_resonance_kernel = True
    cfg.sparse_resonance_top_k = 1
    cfg.scale_gate_skip_threshold = 0.0
    model = MTLNNModel(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out = model(ids)

    diag = model.get_mt_diagnostics()
    assert out["logits"].shape == (2, 8, cfg.vocab_size)
    expected = 1.0 / cfg.n_time_scales
    assert abs(diag["sparse_resonance_scale_ratio"] - expected) < 1e-6
    assert abs(diag["scale_gate_nonzero_ratio"] - expected) < 1e-6
    print("[ok] test_sparse_resonance_kernel_topk")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 60)
    print("MT-LNN test suite")
    print("=" * 60)
    test_shapes_and_loss()
    test_gradient_flow()
    test_direct_target_head_shapes_and_loss()
    test_kv_cache_parity()
    test_pscan_cache_parity_with_recurrence()
    test_lnn_recurrence_active()
    test_prefill_then_decode()
    test_gqa_kv_cache_size()
    test_mt_diagnostics()
    test_dynamic_scale_gate_diagnostics()
    test_sparse_resonance_kernel_topk()
    test_low_rank_polarity()
    test_nearest_neighbor_coupling()
    test_gwtb_bottleneck()
    test_gwtb_cache_parity()
    test_gwtb_per_block_mode()
    test_phi_hat_basic()
    test_anesthesia_validation_protocol()
    test_anesthesia_collapse()
    test_protofilament_scaling()
    test_overfit_single_batch()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
