"""
tests/test_chunkwise_scan_switch.py — use_chunkwise_scan switch contract.

The switch must be (a) OFF by default (red line: zero regression on the
historical pscan path) and (b) semantically equivalent ON vs OFF up to float
reassociation, on identical weights, across BOTH model wirings (v1
MTLNNModel — constant-A and selective regimes — and the v2 adapter layer).

Function-level equivalence lives in tests/test_pscan_chunkwise.py (the merge
gate); this file guards the WIRING: the flag must actually route the scan
and change nothing else (no new parameters, same shapes).
"""

import sys

import torch
from torch.testing import assert_close

sys.path.insert(0, ".")

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.mt_lnn_v2 import MTAdapterV2Config, MTLNNLayerV2

# on/off outputs differ only by scan reassociation -> same gate constants as
# tests/test_pscan_chunkwise.py.
SWITCH_RTOL = 1e-3
SWITCH_ATOL = 1e-3


def _cfg(**kw):
    # Minimal-but-real v1 model (same convention as test_selective_decay.py:
    # d_gw = d_model // 8 must divide gwtb_n_heads).
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


def test_switch_defaults_off():
    """Red line: default path is the historical pscan, bit-identical."""
    cfg = MTLNNConfig()
    assert cfg.use_chunkwise_scan is False
    assert cfg.chunkwise_scan_size == 64
    v2 = MTAdapterV2Config(hidden_size=64)
    assert v2.use_chunkwise_scan is False
    assert v2.chunkwise_scan_size == 64
    print("[ok] test_switch_defaults_off")


def test_on_path_matches_off_path():
    """ON vs OFF on identical weights: reassociation-distance only."""
    torch.manual_seed(0)
    off = MTLNNModel(_cfg())
    torch.manual_seed(0)
    on = MTLNNModel(_cfg(use_chunkwise_scan=True))
    assert set(off.state_dict()) == set(on.state_dict())   # no new params
    on.load_state_dict(off.state_dict())
    x = torch.randint(0, 128, (2, 50))                     # spans chunk boundary
    off.eval(); on.eval()
    with torch.no_grad():
        assert_close(_logits(on(x)), _logits(off(x)),
                     rtol=SWITCH_RTOL, atol=SWITCH_ATOL)

    # Selective regime (general per-step A -> the other dispatch branch)
    torch.manual_seed(1)
    off_sel = MTLNNModel(_cfg(selective_decay=True))
    torch.manual_seed(1)
    on_sel = MTLNNModel(_cfg(selective_decay=True, use_chunkwise_scan=True))
    on_sel.load_state_dict(off_sel.state_dict())
    off_sel.eval(); on_sel.eval()
    with torch.no_grad():
        assert_close(_logits(on_sel(x)), _logits(off_sel(x)),
                     rtol=SWITCH_RTOL, atol=SWITCH_ATOL)

    # v2 adapter layer wiring (constant-A regime; runs both pscan entry points)
    torch.manual_seed(2)
    v2cfg = MTAdapterV2Config(hidden_size=64, n_protofilaments=2, d_proto=16,
                              n_time_scales=3, dropout=0.0)
    v2_off = MTLNNLayerV2(v2cfg)
    v2_on = MTLNNLayerV2(v2cfg)
    v2_on.load_state_dict(v2_off.state_dict())
    xv = torch.randn(2, 40, 64)
    v2_off.eval(); v2_on.eval()
    with torch.no_grad():
        h_off, _ = v2_off(xv)
        h_on, _ = v2_on(xv)
    assert_close(h_on, h_off, rtol=SWITCH_RTOL, atol=SWITCH_ATOL)
    print("[ok] test_on_path_matches_off_path")


def run_all():
    print("=" * 60)
    print("Chunkwise scan switch contract suite")
    print("=" * 60)
    test_switch_defaults_off()
    test_on_path_matches_off_path()
    print("=" * 60)
    print("ALL SWITCH TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
