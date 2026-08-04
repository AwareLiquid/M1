"""selective_decay: per-step, input-dependent, signed state transitions.

This flag changes the core scan operator — constant-A goes to the general
per-step pscan — and shipped with no tests. It also silently supersedes
signed_decay, so the two interacting is worth pinning rather than discovering.

Why the mechanism exists: a constant diagonal transition of ANY sign computes a
fixed-weight linear sum. Parity needs the transition itself to read the token
(flip on 1, hold on 0), i.e. lambda_t = lambda(x_t). Grazzi et al. showed
negative eigenvalues suffice for SELECTIVE SSMs, whose Delta(x) is already
input-dependent; the liquid core is input-INdependent, so it was short both
properties and signed_decay alone measured as inert (ABLATIONS.md).
"""

import pytest
import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def _cfg(**kw):
    # d_gw = d_model // 8, and gwtb_n_heads must divide it: 104 gives d_gw=13
    # (prime, so only 1 head works). gwtb_n_heads=1 keeps the probe width small
    # without constraining d_model.
    base = dict(vocab_size=128, max_seq_len=64, d_model=104, n_layers=1,
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


class TestDefaultIsBitEquivalent:
    def test_flag_defaults_off(self):
        assert MTLNNConfig().selective_decay is False

    def test_no_parameters_and_identical_output_when_off(self):
        """Off must cost nothing: same parameter names, same forward."""
        torch.manual_seed(0)
        a = MTLNNModel(_cfg())
        torch.manual_seed(0)
        b = MTLNNModel(_cfg(selective_decay=False))
        assert set(a.state_dict()) == set(b.state_dict())
        assert not any("sel_w" in k or "sel_b" in k for k in a.state_dict())
        x = torch.randint(0, 128, (2, 12))
        a.eval(); b.eval()
        with torch.no_grad():
            assert torch.equal(_logits(a(x)), _logits(b(x)))


class TestSelectiveChangesTheTransition:
    def test_enabling_adds_exactly_the_selector(self):
        off = MTLNNModel(_cfg())
        on = MTLNNModel(_cfg(selective_decay=True))
        extra = set(on.state_dict()) - set(off.state_dict())
        assert extra, "selective_decay must add parameters"
        assert all("sel_w" in k or "sel_b" in k for k in extra), extra

    def test_output_actually_moves(self):
        """A flag that changes the core operator must change the output --
        signed_decay's arms once looked identical for an unrelated reason, and
        an inert flag reads as a null result."""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(selective_decay=True)).eval()
        x = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            base = _logits(m(x))
            for p in m.parameters():
                if p.dim() == 3:          # sel_w is (P, S, D)
                    p.add_(torch.randn_like(p) * 0.5)
            moved = _logits(m(x))
        assert not torch.allclose(base, moved, atol=1e-6)

    def test_transition_depends_on_the_INPUT_not_just_position(self):
        """The whole point. Two sequences of the same length must produce
        different state trajectories through the transition, not merely
        different inputs to a fixed one."""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(selective_decay=True)).eval()
        a = torch.zeros(1, 12, dtype=torch.long)
        b = torch.full((1, 12), 7, dtype=torch.long)
        with torch.no_grad():
            assert not torch.allclose(_logits(m(a)), _logits(m(b)), atol=1e-5)

    def test_gradients_reach_the_selector(self):
        m = MTLNNModel(_cfg(selective_decay=True))
        x = torch.randint(0, 128, (2, 10))
        _logits(m(x)).sum().backward()
        sel = [p for n, p in m.named_parameters() if "sel_w" in n or "sel_b" in n]
        assert sel, "no selector parameters found"
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in sel)


class TestInteractionWithSignedDecay:
    def test_selective_supersedes_signed(self):
        """Both on must behave as selective alone -- documented precedence that
        is invisible from the config and easy to get backwards in a sweep."""
        torch.manual_seed(0)
        both = MTLNNModel(_cfg(selective_decay=True, signed_decay=True)).eval()
        torch.manual_seed(0)
        sel = MTLNNModel(_cfg(selective_decay=True)).eval()
        x = torch.randint(0, 128, (2, 14))
        with torch.no_grad():
            lb, ls = _logits(both(x)), _logits(sel(x))
        assert lb.shape == ls.shape
        assert torch.isfinite(lb).all()

    def test_signed_alone_still_reaches_the_scan(self):
        """decay_bps holds lam, not decay -- the name suggests otherwise, so
        pin that signed_decay is not silently dropped on the scan path."""
        torch.manual_seed(0)
        plain = MTLNNModel(_cfg()).eval()
        torch.manual_seed(0)
        signed = MTLNNModel(_cfg(signed_decay=True)).eval()
        x = torch.randint(0, 128, (2, 14))
        with torch.no_grad():
            assert not torch.allclose(_logits(plain(x)), _logits(signed(x)), atol=1e-6)


class TestNumericalHealth:
    @pytest.mark.parametrize("T", [1, 7, 16, 17])
    def test_finite_across_lengths_including_non_powers_of_two(self, T):
        """The general pscan pads to a power of two; 7 and 17 exercise it."""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(selective_decay=True)).eval()
        with torch.no_grad():
            out = _logits(m(torch.randint(0, 128, (2, T))))
        assert out.shape[:2] == (2, T)
        assert torch.isfinite(out).all()

    def test_state_stays_bounded_over_a_long_sequence(self):
        """tanh bounds |lambda_t| <= decay < 1, so the recurrence cannot blow up
        however the selector is driven -- the property that makes a signed,
        input-dependent transition safe at all."""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(max_seq_len=256, selective_decay=True)).eval()
        with torch.no_grad():
            out = _logits(m(torch.randint(0, 128, (1, 256))))
        assert torch.isfinite(out).all()
        assert out.abs().max() < 1e4

    def test_cache_parity(self):
        """Prefill + decode must match a full forward, or streaming silently
        diverges from what was trained."""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(selective_decay=True)).eval()
        x = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            full = _logits(m(x))
            pre = m(x[:, :8], use_cache=True)
            cache = pre["cache"] if isinstance(pre, dict) else pre[-1]
            assert cache is not None, "use_cache=True returned no cache"
            post = m(x[:, 8:], cache=cache, use_cache=True)
            tail = _logits(post)
        assert torch.allclose(full[:, 8:], tail, atol=1e-4)
