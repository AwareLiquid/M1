"""Unit tests for LiquidADRegressor — the hybrid structural+learned decay probe.

These pin what the pre-registered R2'/R1' verdict will rest on:
  * BOTH decay paths are wired and the gate really routes between them:
    g forced to 1 reproduces LiquidDTRegressor's bank exactly (single-
    variable attribution), g forced to 0 leaves the MLP path live and
    mlp perturbations inert at g=1
  * gate, mlp_lam and log_tau all receive non-zero gradients (no
    decorative parameters)
  * λ stays bounded in [0, 1] — both paths are exponentials of
    non-positive reals, so the scan cannot blow up by construction
  * the scanned bank equals a naive sequential loop (pscan parity)
  * the parameter budget stays in the gru band; .inp/.head patchable by
    the driver scripts; CPU fwd/bwd fast

CPU-only; the whole file is sized to finish well under a minute.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.battery_soh_edge import LiquidADRegressor, LiquidDTRegressor, build


def _fresh(d_model=78, n_layers=1, n_feat=4, seed=0, **kw):
    torch.manual_seed(seed)
    return LiquidADRegressor(d_model, n_layers, n_feat, **kw)


def _x(B=2, T=8, F=4, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, T, F)
    x[..., -1] = x[..., -1].abs() + 0.05        # dt: positive protocol units
    return x


class TestScanMath:
    def test_bank_matches_sequential_loop(self):
        """The blended-λ scan usage must equal a naive sequential recurrence."""
        m = _fresh(seed=1)
        x = _x(2, 9, seed=1)
        with torch.no_grad():
            seq = m.inp(x)
            dt = x[..., -1]
            e = seq.reshape(2, 9, m.P, m.Dp)
            A = torch.sigmoid(torch.einsum("btpd,psd->btpsd", e, m.w_in_0)
                              + m.b_in_0[None, None, :, :, None])
            lam = m._lam(0, seq, dt)                     # (B,T,P,S)
            h = torch.zeros(2, m.P, m.S, m.Dp)
            hs = []
            for t in range(9):
                h = lam[:, t, :, :, None] * h \
                    + (1.0 - lam[:, t, :, :, None]) * A[:, t]
                hs.append(h.clone())
            ref = torch.stack(hs, dim=1)                 # (B,T,P,S,Dp)
            w = torch.softmax(m.blend_0, dim=-1)
            expected = torch.einsum("btpsd,ps->btpd", ref, w).reshape(2, 9, -1)
            out = m._bank(0, seq, dt)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_lambda_bounded_01(self):
        """λ ∈ [0,1] for any input, strictly inside on ordinary inputs —
        both paths are exponentials of non-positive reals."""
        m = _fresh(seed=2)
        x = _x(3, 12, seed=2)
        with torch.no_grad():
            seq = m.inp(x)
            lam = m._lam(0, seq, x[..., -1])
            lam_huge = m._lam(0, seq, torch.full((3, 12), 1e9))
            lam_zero = m._lam(0, seq, torch.zeros(3, 12))
        assert (lam > 0).all() and (lam < 1).all()
        assert (lam_huge >= 0).all() and (lam_huge <= 1).all()
        assert (lam_zero >= 0).all() and (lam_zero <= 1).all()


class TestTwoPathWiring:
    def test_gate_one_reproduces_dt_bank_exactly(self):
        """g=1 ⇒ the AD bank IS the LiquidDTRegressor bank with the same
        path-1 parameters — the everything-else-identical attribution."""
        ad, dtm = _fresh(seed=3), LiquidDTRegressor(78, 1, 4)
        with torch.no_grad():
            for n_ in ("w_in_0", "b_in_0", "log_tau_0", "blend_0"):
                getattr(dtm, n_).copy_(getattr(ad, n_))
            ad.gate_0.fill_(20.0)                # sigmoid ≈ 1
            x = _x(2, 7, seed=3)
            seq = ad.inp(x)                      # SAME embedding into both
            a = ad._bank(0, seq, x[..., -1])
            b = dtm._bank(0, seq, x[..., -1])
        assert torch.allclose(a, b, atol=1e-6)

    def test_gate_forcing_changes_output(self):
        """Clamping g to the two extremes must change the prediction — both
        paths are wired and produce different decays."""
        m = _fresh(n_layers=2, seed=4)
        x = _x(3, 10, seed=4)
        with torch.no_grad():
            m.gate_0.fill_(20.0); m.gate_1.fill_(20.0)
            y_struct = m(x)
            m.gate_0.fill_(-20.0); m.gate_1.fill_(-20.0)
            y_mlp = m(x)
        assert not torch.allclose(y_struct, y_mlp, atol=1e-6)

    def test_mlp_isolated_by_gate(self):
        """g=1: perturbing mlp_lam leaves the output bit-identical (the gate
        really isolates); g=0: the same perturbation moves it (path 2 live)."""
        x = _x(2, 8, seed=5)
        with torch.no_grad():
            m = _fresh(seed=5)
            m.gate_0.fill_(20.0)
            y0 = m(x)
            m.mlp_lam_0[2].weight.add_(0.5)
            y1 = m(x)
            assert torch.allclose(y0, y1)
            m.gate_0.fill_(-20.0)
            y2 = m(x)
            m.mlp_lam_0[2].weight.add_(0.5)
            y3 = m(x)
        assert not torch.allclose(y2, y3, atol=1e-7)

    def test_mlp_path_reads_dt(self):
        """At g=0 (pure learned path) the output still responds to Δt alone —
        the MLP consumes the gap, not just the sensors."""
        m = _fresh(seed=6)
        with torch.no_grad():
            m.gate_0.fill_(-20.0)
            x = _x(2, 8, seed=6)
            y1 = m(x)
            x2 = x.clone()
            x2[..., -1] = x2[..., -1] * 8.0
            y2 = m(x2)
        assert not torch.allclose(y1, y2, atol=1e-6)


class TestGradientsAndBudget:
    def test_gate_mlp_tau_receive_gradient(self):
        m = _fresh(n_layers=2, seed=7)
        m(_x(2, 8, seed=7)).pow(2).mean().backward()
        for n_ in ("gate_0", "gate_1", "log_tau_0", "log_tau_1"):
            g = getattr(m, n_).grad
            assert g is not None and torch.isfinite(g).all()
            assert float(g.abs().max()) > 0, n_
        for li in (0, 1):
            for p in getattr(m, f"mlp_lam_{li}").parameters():
                assert p.grad is not None
                assert float(p.grad.abs().max()) > 0
            for p in getattr(m, f"lam_proj_{li}").parameters():
                assert p.grad is not None
                assert float(p.grad.abs().max()) > 0

    def test_param_band_matches_gru(self):
        m = build("mt_lnn_ad", 78, 2, 128)
        m.inp = torch.nn.Linear(4, 78)
        n = sum(p.numel() for p in m.parameters())
        ref_gru = sum(p.numel() for p in build("gru", 78, 2).parameters())
        assert 0.5 * ref_gru < n < 1.5 * ref_gru, n

    def test_forward_backward_cpu_fast(self):
        m = _fresh(n_layers=2, seed=8)
        x = _x(4, 16, seed=8)
        t0 = time.time()
        for _ in range(3):
            y = m(x)
            loss = y.pow(2).mean()
            loss.backward()
        assert torch.isfinite(y).all()
        assert time.time() - t0 < 30.0

    def test_head_patchable_like_other_archs(self):
        """Driver scripts (air/synth) patch .inp/.head — must keep working."""
        m = build("mt_lnn_ad", 78, 2, 128)
        m.inp = torch.nn.Linear(12, 78)
        m.head = torch.nn.Linear(78, 12)
        y = m(_x(2, 9, F=12, seed=9))
        assert y.shape == (2, 12)

    def test_freeze_tau_flag(self):
        m = _fresh(seed=10, freeze_tau=True)
        assert m.log_tau_0.requires_grad is False
        m2 = _fresh(seed=10, freeze_tau=False)
        assert m2.log_tau_0.requires_grad is True
