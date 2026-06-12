"""
test_long_context_memory.py -- pins the O(1) streaming-memory headline claim.

The product's flagship performance claim is that state-only recurrent streaming
holds a CONSTANT working-memory footprint as the stream length ``T`` grows,
while an ordinary KV cache grows ``O(T)``. These regression tests pin BOTH
halves of that contrast at a stream length that far exceeds the model's finite
RoPE/mask window (``T >> max_seq_len``) -- exactly the regime the benchmark
script sizes its tables to avoid, and the regime the prior memory test never
crossed (it stopped at ``max_seq_len + 8``, less than one wrap).

Run:  python -m pytest tests/test_long_context_memory.py -v
"""
import sys
import warnings

import torch

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel                     # noqa: E402
from mt_lnn.streaming import streaming_inference               # noqa: E402


def tiny_cfg(max_seq_len: int = 16) -> MTLNNConfig:
    """A deliberately small RoPE window so a CPU run can cross many wraps fast."""
    return MTLNNConfig(
        vocab_size=64, max_seq_len=max_seq_len, d_model=64,
        n_layers=2, n_heads=4, n_kv_heads=2, d_head=16,
        dropout=0.0, attention_dropout=0.0,
    )


def _state_only_curve(model, cfg, n_steps, *, seed=0):
    """Stream ``n_steps`` single tokens in state-only mode; return byte curve."""
    torch.manual_seed(seed)
    cache = None
    sizes = []
    for _ in range(n_steps):
        tok = torch.randint(0, cfg.vocab_size, (1, 1))
        logits, cache = streaming_inference(model, tok, cache, state_only=True)
        assert logits.shape == (1, 1, cfg.vocab_size)
        sizes.append(cache.tensor_bytes())
    return sizes, cache


def _kv_curve(model, cfg, n_steps, *, seed=1):
    """Stream ``n_steps`` single tokens keeping full KV history; byte curve."""
    torch.manual_seed(seed)
    cache = None
    sizes = []
    for _ in range(n_steps):
        tok = torch.randint(0, cfg.vocab_size, (1, 1))
        out = model(tok, cache=cache, use_cache=True, use_lnn_recurrence=True)
        cache = out["cache"]
        sizes.append(cache.tensor_bytes())
    return sizes, cache


# ---------------------------------------------------------------------------
# O(1): state-only memory is flat, even far past the finite RoPE window
# ---------------------------------------------------------------------------
def test_state_only_memory_is_flat_far_beyond_the_rope_window():
    cfg = tiny_cfg(max_seq_len=16)
    model = MTLNNModel(cfg).eval()

    n_steps = cfg.max_seq_len * 20                  # 320 steps == 20 RoPE wraps
    sizes, cache = _state_only_curve(model, cfg, n_steps)

    # The whole point of O(1): every step after the seed holds the IDENTICAL
    # byte count, no matter how large T grows.
    assert len(set(sizes[1:])) == 1
    # It genuinely streamed past many wrap boundaries (not just one short run).
    assert cache.token_count == n_steps
    assert n_steps > cfg.max_seq_len * 8


def test_state_only_constant_is_independent_of_T():
    """The flat value itself must not depend on how long we have streamed."""
    cfg = tiny_cfg(max_seq_len=16)
    model = MTLNNModel(cfg).eval()

    short, _ = _state_only_curve(model, cfg, cfg.max_seq_len * 2)
    long, _ = _state_only_curve(model, cfg, cfg.max_seq_len * 20)

    # Same constant footprint whether measured early or 10x deeper into the stream.
    assert len(set(short[1:])) == 1
    assert len(set(long[1:])) == 1
    assert short[-1] == long[-1]


# ---------------------------------------------------------------------------
# O(T): the KV cache grows linearly -- and state-only does not
# ---------------------------------------------------------------------------
def test_kv_cache_grows_linearly_while_state_only_stays_constant():
    cfg = tiny_cfg(max_seq_len=16)
    model = MTLNNModel(cfg).eval()

    # A full-history KV stream can only run within the finite RoPE window;
    # measure its growth across that window.
    kv_sizes, _ = _kv_curve(model, cfg, cfg.max_seq_len)

    # Strictly monotonic: every token appends fresh key/value positions.
    assert all(b > a for a, b in zip(kv_sizes, kv_sizes[1:]))
    # And the growth is LINEAR -- a constant per-step increment is the O(T)
    # signature (allowing tiny slack for any compressed side caches).
    increments = [b - a for a, b in zip(kv_sizes, kv_sizes[1:])]
    assert max(increments) - min(increments) <= 1e-6 * max(increments) or len(set(increments)) == 1
    # By the end of the window the KV cache already dwarfs the first step.
    assert kv_sizes[-1] > kv_sizes[0] * 2

    # State-only over the SAME window is flat ...
    st_sizes, _ = _state_only_curve(model, cfg, cfg.max_seq_len)
    assert len(set(st_sizes[1:])) == 1
    # ... and the bounded recurrent state is strictly smaller than the KV cache,
    # a gap that only widens as T grows.
    assert kv_sizes[-1] > st_sizes[-1]


def test_the_memory_gap_widens_without_bound_as_T_grows():
    """The two paths are both runnable past the window, but their footprints
    diverge: KV keeps accumulating O(T) while state-only stays pinned, so the
    KV/state ratio is strictly larger at 8 wraps than at 2."""
    cfg = tiny_cfg(max_seq_len=16)
    model = MTLNNModel(cfg).eval()

    near = cfg.max_seq_len * 2                       # 32 steps  (2 wraps)
    far = cfg.max_seq_len * 8                        # 128 steps (8 wraps)

    kv_near, _ = _kv_curve(model, cfg, near)
    kv_far, _ = _kv_curve(model, cfg, far)
    st_far, _ = _state_only_curve(model, cfg, far)

    # KV keeps growing the whole way -- no plateau, no cap.
    assert kv_far[-1] > kv_near[-1]
    assert all(b > a for a, b in zip(kv_far, kv_far[1:]))
    # State-only is flat across the same long stream.
    assert len(set(st_far[1:])) == 1

    # The decisive headline number: the relative blow-up of KV over the bounded
    # recurrent state grows with T (here ~2x -> ~8x as the stream octuples).
    ratio_near = kv_near[-1] / st_far[-1]
    ratio_far = kv_far[-1] / st_far[-1]
    assert ratio_far > ratio_near * 3                # widening, not converging
    assert ratio_far > 10                            # KV already an order larger


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
