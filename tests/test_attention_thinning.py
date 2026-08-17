"""attention_layers: replace attention, don't stack on top of it.

The current hybrid is additive -- a full N-layer Transformer with liquid layers
bolted on -- which is why "hybrid is O(1)" was retracted and hybrid training
memory measured strictly worse than the Transformer baseline (RESULTS.md).
Production hybrids replace: LFM2 keeps attention in 6/16 layers. This knob
makes a layer outside the set a pure LNN+FFN block: no norm, no projections,
no KV cache entry. The saving must be real, not a bypassed module.

Also pinned here: position tracking survives thinning. The historical offset
inference read T_past off layer 0's K tensor; with layer 0 thinned that path
silently returns 0 and every decode step gets a skewed RoPE/GTP phase. The fix
routes through cache.token_count, and the test that would have caught the bug
is the decode-parity one with layer 0 in the thinned set.
"""

import pytest
import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def _cfg(**kw):
    base = dict(vocab_size=128, max_seq_len=64, d_model=104, n_layers=4,
                n_heads=4, n_kv_heads=2, d_head=26, gwtb_n_heads=1,
                dropout=0.0, attention_dropout=0.0)
    base.update(kw)
    return MTLNNConfig(**base)


def _logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return out["logits"]
    return out[0] if isinstance(out, tuple) else out


def _cache(out):
    if isinstance(out, dict):
        return out.get("cache")
    return out[-1] if isinstance(out, tuple) else getattr(out, "cache", None)


class TestDefaultIsBitExact:
    def test_none_means_every_layer(self):
        assert MTLNNConfig().attention_layers is None

    def test_default_model_is_bit_identical(self):
        torch.manual_seed(0)
        a = MTLNNModel(_cfg())
        torch.manual_seed(0)
        b = MTLNNModel(_cfg(attention_layers=None))
        assert set(a.state_dict()) == set(b.state_dict())
        x = torch.randint(0, 128, (2, 12))
        a.eval(); b.eval()
        with torch.no_grad():
            assert torch.equal(_logits(a(x)), _logits(b(x)))

    def test_explicit_full_set_matches_none(self):
        """(0,1,2,3) on a 4-layer model is the same parameter set as None."""
        torch.manual_seed(0)
        a = MTLNNModel(_cfg())
        torch.manual_seed(0)
        b = MTLNNModel(_cfg(attention_layers=(0, 1, 2, 3)))
        assert set(a.state_dict()) == set(b.state_dict())


class TestThinningIsReal:
    def test_parameters_actually_drop(self):
        full = MTLNNModel(_cfg())
        thin = MTLNNModel(_cfg(attention_layers=(1, 3)))
        pf = sum(p.numel() for p in full.parameters())
        pt = sum(p.numel() for p in thin.parameters())
        assert pt < pf
        # and the dropped names are exactly the attention machinery of 0 and 2
        gone = set(full.state_dict()) - set(thin.state_dict())
        assert gone
        assert all(("blocks.0." in k or "blocks.2." in k) and
                   ("attn" in k) for k in gone), sorted(gone)[:6]

    def test_thinned_layers_have_no_kv_cache_entry(self):
        """The KV saving is the point -- None slots, not empty tensors."""
        m = MTLNNModel(_cfg(attention_layers=(1, 3))).eval()
        x = torch.randint(0, 128, (2, 8))
        with torch.no_grad():
            cache = _cache(m(x, use_cache=True))
        assert cache is not None
        has_kv = [layer is not None and layer[0] is not None
                  for layer in cache.layers]
        assert has_kv == [False, True, False, True]

    def test_forward_and_generate_stay_finite(self):
        m = MTLNNModel(_cfg(attention_layers=(1, 3))).eval()
        x = torch.randint(0, 128, (2, 12))
        with torch.no_grad():
            out = _logits(m(x))
            g = m.generate(x[:, :6], max_new_tokens=4)
        assert torch.isfinite(out).all()
        assert g.shape == (2, 10)

    def test_gradients_flow_through_thinned_blocks(self):
        """The pure-LNN block must still train -- a dead layer would be a
        silent 25% parameter cut."""
        m = MTLNNModel(_cfg(attention_layers=(1, 3)))
        x = torch.randint(0, 128, (2, 10))
        _logits(m(x)).sum().backward()
        lnn0 = [p for n, p in m.named_parameters()
                if n.startswith("blocks.0.lnn.") and p.grad is not None]
        assert lnn0
        assert all(torch.isfinite(p.grad).all() for p in lnn0)


class TestPositionSurvivesThinning:
    def test_decode_parity_with_layer0_thinned(self):
        """THE regression test for the offset bug. Layer 0 carries no KV, so
        the old inference read 0 and skewed RoPE/GTP on every decode step --
        producing plausible-looking but position-shifted logits. token_count
        is authoritative now; prefill+decode must match the full forward."""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(attention_layers=(1, 3))).eval()
        x = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            full = _logits(m(x))
            pre = m(x[:, :8], use_cache=True)
            cache = _cache(pre)
            assert cache.token_count == 8
            post = m(x[:, 8:], cache=cache, use_cache=True)
        assert torch.allclose(full[:, 8:], _logits(post), atol=1e-4)

    def test_pure_lnn_stack_decodes(self):
        """attention_layers=() is legal: zero attention anywhere. Position can
        only come from token_count -- there is no K tensor in the model."""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(attention_layers=())).eval()
        x = torch.randint(0, 128, (1, 10))
        with torch.no_grad():
            full = _logits(m(x))
            pre = m(x[:, :6], use_cache=True)
            post = m(x[:, 6:], cache=_cache(pre), use_cache=True)
        assert torch.allclose(full[:, 6:], _logits(post), atol=1e-4)


class TestValidation:
    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            _cfg(attention_layers=(0, 7))

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError, match="duplicates"):
            _cfg(attention_layers=(1, 1, 3))

    def test_indices_are_normalised_sorted(self):
        assert _cfg(attention_layers=(3, 1)).attention_layers == (1, 3)
