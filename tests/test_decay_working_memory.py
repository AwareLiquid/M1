"""
tests/test_decay_working_memory.py -- behavioural contract for the O(1)
working-memory decay path in the Global Coherence layer.

This mechanism (BRAIN_INSPIRED_ROADMAP.md, Phase 1 item "工作记忆缓冲区 / GWTB
升级") is ON by default (``use_decay_wm=True``, ``wm_decay_rate_init=0.99`` in
``MTLNNConfig``) and is the architecture's headline "infinite-turn, O(1) memory"
claim. Prior to this file it had NO test: the existing long-context memory test
streams via ``ModelCacheStruct.recurrent_only()``, which DROPS ``coherence_kv``
entirely, so it never exercises this path. We close that gap.

What the layer does when ``use_decay_wm=True`` (``GlobalCoherenceLayer.forward``):
instead of concatenating K/V over history (the O(T) cache), it keeps a single
rolling working-memory vector and updates it with a gated exponential moving
average,

    curr_wm = curr_wm * decay * (1 - u_t) + o_t * u_t,   u_t = sigmoid(update_gate(x_t)),

re-packing the rolling state into the cache as ``(wm_seq, wm_seq)`` of shape
``(B, T_new, d_model)``. Under single-token streaming that is ``(B, 1, d_model)``
for every step -- constant in the number of tokens seen.

We pin:
- the flag actually builds (or omits) ``update_gate`` + ``decay_rate``;
- the coherence cache is CONSTANT size under streaming in decay mode, whereas
  the legacy mode's K/V cache grows linearly -- this is the O(1)-vs-O(T) claim,
  tested both at the layer level and end-to-end through ``MTLNNModel``;
- the working memory forgets geometrically on a zero input (the "decay" half);
- a non-zero input writes into the working memory (the "update" half).

Run:  python -m pytest tests/test_decay_working_memory.py -v
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import GlobalCoherenceLayer, MTLNNConfig, MTLNNModel    # noqa: E402


def _config(**overrides):
    base = dict(
        vocab_size=200,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        dropout=0.0,            # deterministic
        attention_dropout=0.0,
    )
    base.update(overrides)
    return MTLNNConfig(**base)


# --------------------------------------------------------------------------- #
# flag wiring                                                                  #
# --------------------------------------------------------------------------- #
def test_decay_flag_builds_update_gate_and_decay_rate():
    on = GlobalCoherenceLayer(_config(use_decay_wm=True))
    assert hasattr(on, "update_gate") and hasattr(on, "decay_rate")
    assert on.decay_rate.item() == pytest.approx(0.99)

    off = GlobalCoherenceLayer(_config(use_decay_wm=False))
    assert not hasattr(off, "update_gate")
    assert not hasattr(off, "decay_rate")


# --------------------------------------------------------------------------- #
# O(1) cache at the layer level (vs O(T) for the legacy path)                  #
# --------------------------------------------------------------------------- #
def test_decay_cache_is_constant_size_while_legacy_grows():
    torch.manual_seed(0)
    B, d = 2, 128
    decay = GlobalCoherenceLayer(_config(use_decay_wm=True)).eval()
    legacy = GlobalCoherenceLayer(_config(use_decay_wm=False)).eval()

    decay_kv = legacy_kv = None
    decay_sizes, legacy_sizes = [], []
    with torch.no_grad():
        for step in range(8):
            x = torch.randn(B, 1, d)
            _, decay_kv = decay(x, past_kv=decay_kv, position_offset=step, use_cache=True)
            _, legacy_kv = legacy(x, past_kv=legacy_kv, position_offset=step, use_cache=True)
            # decay path: rolling WM (B, 1, d_model) -- the token axis never grows
            decay_sizes.append(decay_kv[0].shape[1])
            # legacy path: K is (B, H, T_total, D) -- the time axis grows each step
            legacy_sizes.append(legacy_kv[0].shape[2])

    assert decay_sizes == [1] * 8                    # O(1): constant
    assert legacy_sizes == [1, 2, 3, 4, 5, 6, 7, 8]  # O(T): linear growth


# --------------------------------------------------------------------------- #
# O(1) cache end-to-end through the model                                      #
# --------------------------------------------------------------------------- #
def test_model_coherence_cache_is_o1_under_streaming():
    torch.manual_seed(0)
    model = MTLNNModel(_config(use_decay_wm=True)).eval()

    cache = None
    sizes = []
    with torch.no_grad():
        for step in range(8):
            ids = torch.randint(0, 200, (1, 1))
            out = model(ids, cache=cache, use_cache=True, position_offset=step)
            cache = out["cache"]
            sizes.append(cache.coherence_kv[0].shape[1])

    # The coherence working memory stays a single rolling state regardless of
    # how many tokens have streamed through -- the O(1) memory claim.
    assert sizes == [1] * 8


def test_model_coherence_cache_grows_without_decay():
    # Control: the legacy coherence cache genuinely grows, so the O(1) result
    # above is a property of the decay path, not of the harness.
    torch.manual_seed(0)
    model = MTLNNModel(_config(use_decay_wm=False)).eval()

    cache = None
    sizes = []
    with torch.no_grad():
        for step in range(6):
            ids = torch.randint(0, 200, (1, 1))
            out = model(ids, cache=cache, use_cache=True, position_offset=step)
            cache = out["cache"]
            sizes.append(cache.coherence_kv[0].shape[2])  # (B,H,T_total,D)

    assert sizes == [1, 2, 3, 4, 5, 6]


# --------------------------------------------------------------------------- #
# EMA semantics: forget on zero input, write on non-zero input                #
# --------------------------------------------------------------------------- #
def _wm_norm(kv):
    # rolling WM is packed as (wm_seq, wm_seq); the last row is the current state
    return kv[0][:, -1, :].norm().item()


def test_working_memory_forgets_geometrically_on_zero_input():
    torch.manual_seed(0)
    layer = GlobalCoherenceLayer(_config(use_decay_wm=True)).eval()
    B, d = 1, 128

    with torch.no_grad():
        # prime the memory with a strong write
        _, kv = layer(torch.randn(B, 1, d) * 5.0, past_kv=None, position_offset=0, use_cache=True)
        norms = [_wm_norm(kv)]
        # then feed zeros: with x=0 the bias-free projections give o_t=0, so
        # curr_wm <- curr_wm * decay * (1 - u_t), a strict geometric contraction.
        for step in range(1, 7):
            _, kv = layer(torch.zeros(B, 1, d), past_kv=kv, position_offset=step, use_cache=True)
            norms.append(_wm_norm(kv))

    assert norms[0] > 0.0
    for a, b in zip(norms, norms[1:]):
        assert b < a                                  # strictly forgetting
    assert norms[-1] < norms[0] * 0.5                 # and clearly decayed


def test_working_memory_writes_on_nonzero_input():
    torch.manual_seed(0)
    layer = GlobalCoherenceLayer(_config(use_decay_wm=True)).eval()
    B, d = 1, 128
    with torch.no_grad():
        _, kv = layer(torch.zeros(B, 1, d), past_kv=None, position_offset=0, use_cache=True)
        empty_norm = _wm_norm(kv)
        _, kv = layer(torch.randn(B, 1, d) * 3.0, past_kv=kv, position_offset=1, use_cache=True)
        written_norm = _wm_norm(kv)
    assert empty_norm == pytest.approx(0.0, abs=1e-6)  # zero in -> nothing written
    assert written_norm > 1e-3                          # real input -> a write


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
