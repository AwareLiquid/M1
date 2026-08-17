"""Attention heads are independent of the protofilament count.

The historical default pairs n_heads with n_protofilaments (both 13), which is
naming aesthetics with a real cost: 13 is prime, so GQA could only be 13:1
(acc 0.248 on the relational probe) or 1:1 (acc 1.0000 at 13x KV). These tests
pin down that (a) the default stays bit-identical, and (b) a decoupled shape --
16 heads, d_head 52, 4:1 GQA on the same d_model=832 -- runs the full model
end to end, so the middle ratios exist when the sweep needs them.

See ABLATIONS.md "Design-coupling audit" and HANDOFF section 3.8.
"""

import pytest
import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def _tiny(**kw):
    base = dict(vocab_size=256, max_seq_len=64, d_model=832, n_layers=2,
                dropout=0.0, attention_dropout=0.0)
    base.update(kw)
    return MTLNNConfig(**base)


def _logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return out["logits"]
    return out[0] if isinstance(out, tuple) else out


class TestDefaultUnchanged:
    def test_defaults_are_still_coupled_13_to_1(self):
        cfg = MTLNNConfig()
        assert (cfg.n_heads, cfg.n_kv_heads, cfg.d_head) == (13, 1, 64)

    def test_default_parameter_set_is_untouched(self):
        """Same names AND shapes as the historical config -- checkpoint-compatible."""
        torch.manual_seed(0)
        m = MTLNNModel(_tiny())
        sd = {k: tuple(v.shape) for k, v in m.state_dict().items()}
        # Attention projections at the historical sizes: q 13*64, kv 1*64.
        q = [v for k, v in sd.items() if k.endswith("attn.q_proj.weight")]
        k_ = [v for k, v in sd.items() if k.endswith("attn.k_proj.weight")]
        assert q and all(v == (832, 832) for v in q)
        assert k_ and all(v == (64, 832) for v in k_)


class TestDecoupledShapes:
    def test_16_heads_4to1_constructs_and_forwards(self):
        """The sweep configuration: 832 kept, heads decoupled, LNN untouched."""
        cfg = _tiny(n_heads=16, n_kv_heads=4, d_head=52)
        assert cfg.n_protofilaments == 13     # biology stays where it belongs
        assert cfg.d_proto == 64              # LNN width unchanged by the head change
        m = MTLNNModel(cfg).eval()
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            logits = _logits(m(x))
        assert logits.shape == (2, 16, 256)
        assert torch.isfinite(logits).all()

    def test_generate_exercises_the_gqa_cache(self):
        """4:1 KV is the point of decoupling -- the cache path must work, not
        just the parallel forward."""
        cfg = _tiny(n_heads=16, n_kv_heads=4, d_head=52)
        m = MTLNNModel(cfg).eval()
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            out = m.generate(x, max_new_tokens=4)
        assert out.shape == (2, 12)

    def test_composes_with_the_global_head_quota(self):
        """The two dials the prime lock froze must turn together: middle-ratio
        GQA and n_global_heads (architecture principle #1)."""
        cfg = _tiny(n_heads=16, n_kv_heads=4, d_head=52, n_global_heads=2)
        m = MTLNNModel(cfg).eval()
        x = torch.randint(0, 256, (1, 12))
        with torch.no_grad():
            logits = _logits(m(x))
        assert torch.isfinite(logits).all()

    @pytest.mark.parametrize("kv", [1, 2, 4, 8, 16])
    def test_every_divisor_ratio_is_expressible(self, kv):
        cfg = _tiny(n_heads=16, n_kv_heads=kv, d_head=52)
        assert cfg.n_heads % cfg.n_kv_heads == 0


class TestInvalidShapesStillRejected:
    def test_non_divisor_gqa_raises(self):
        with pytest.raises(AssertionError, match="divisible"):
            _tiny(n_heads=16, n_kv_heads=3, d_head=52)

    def test_wrong_d_head_raises(self):
        with pytest.raises(AssertionError, match="d_head"):
            _tiny(n_heads=16, n_kv_heads=4, d_head=64)

    def test_odd_d_head_fails_at_config_with_the_reason(self):
        """RoPE pairs dimensions; before this check the failure was a bare
        `assert d_head % 2 == 0` deep inside RotaryEmbedding."""
        with pytest.raises(ValueError, match="rotary"):
            _tiny(d_model=104, n_heads=8, n_kv_heads=2, d_head=13)

    def test_prime_13_still_only_offers_the_two_extremes(self):
        """Documents WHY the default is stuck: with 13 heads, anything between
        MQA and full MHA is arithmetically impossible."""
        for kv in (2, 4, 8):
            with pytest.raises(AssertionError):
                _tiny(n_heads=13, n_kv_heads=kv, d_head=64)
        _tiny(n_heads=13, n_kv_heads=1, d_head=64)    # MQA: legal
        _tiny(n_heads=13, n_kv_heads=13, d_head=64)   # full MHA: legal
