"""Unit tests for LiquidDTRegressor — the Δt-wired liquid mechanism probe.

These pin the mathematics that the mechanism claim rests on:
  * λ_t = exp(-Δt_t/τ) is EXACTLY 1 at Δt=0 (no decay over a zero gap —
    relu-free construction, bitwise, for any learned τ)
  * λ_t → 0 as Δt → ∞, so the state converges to the input coupling A_t
  * the scanned recurrence equals a plain sequential loop (pscan parity)
  * time-warp invariance under CONSTANT input: two steps of (dt1, dt2)
    equal one step of (dt1+dt2) — the closed-form ODE property that
    discrete-step decay (config.dt constant) does NOT have; this is the
    single property the variant exists to test
  * the Δt channel actually reaches the recurrence (output changes when
    only Δt changes), and the parameter budget stays in the mt_lnn/gru band

CPU-only; the whole file is sized to finish well under a minute.
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.battery_soh_edge import LiquidDTRegressor, build


def _fresh(d_model=78, n_layers=1, n_feat=4, seed=0):
    torch.manual_seed(seed)
    return LiquidDTRegressor(d_model, n_layers, n_feat)


class TestScanMath:
    def test_pscan_matches_sequential_loop(self):
        """The bank's scan usage must equal a naive sequential recurrence."""
        from mt_lnn.parallel_scan import pscan
        g = torch.Generator().manual_seed(1)
        B, P, S, T, D = 2, 3, 4, 9, 5
        lam = torch.rand(B, P, S, T, generator=g) * 0.999
        A = torch.randn(B, P, S, T, D, generator=g)
        H = pscan(lam, (1.0 - lam)[..., None] * A)
        h = torch.zeros(B, P, S, D)
        hs = []
        for t in range(T):
            h = lam[..., t, None] * h + (1.0 - lam[..., t, None]) * A[..., t, :]
            hs.append(h.clone())
        H_ref = torch.stack(hs, dim=-2)
        assert torch.allclose(H, H_ref, atol=1e-5)

    def test_zero_dt_bank_output_is_exactly_zero(self):
        """Δt=0 ⇒ λ=1 ⇒ X=0 ⇒ state stays at h_init=0 ⇒ bank adds nothing."""
        m = _fresh(n_layers=1, seed=2)
        x = torch.randn(2, 7, 4)
        with torch.no_grad():
            seq = m.inp(x)
            out = m._bank(0, seq, torch.zeros(2, 7))
        assert float(out.abs().max()) == 0.0

    def test_large_dt_converges_to_input_coupling(self):
        """Δt→∞ ⇒ λ→0 ⇒ every scale's state ≈ its own A_t, so the blend
        equals the weighted sum of the per-scale couplings."""
        m = _fresh(n_layers=1, seed=3)
        x = torch.randn(2, 6, 4)
        dt = torch.full((2, 6), 1e9)
        with torch.no_grad():
            seq = m.inp(x)
            e = seq.reshape(2, 6, m.P, m.Dp)
            A = torch.sigmoid(torch.einsum("btpd,psd->btpsd", e, m.w_in_0)
                              + m.b_in_0[None, None, :, :, None])
            w = torch.softmax(m.blend_0, dim=-1)
            expected = torch.einsum("btpsd,ps->btpd", A, w).reshape(2, 6, -1)
            out = m._bank(0, seq, dt)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_time_warp_invariance_constant_input(self):
        """Two steps of (dt1, dt2) == one step of (dt1+dt2) when input is
        constant across the steps — the continuous-time property."""
        tau, dt1, dt2, h0, a = 0.7, 0.3, 0.9, 0.4, 0.6
        l1, l2 = math.exp(-dt1 / tau), math.exp(-dt2 / tau)
        h_two = l2 * (l1 * h0 + (1 - l1) * a) + (1 - l2) * a
        h_one = math.exp(-(dt1 + dt2) / tau) * h0 + \
            (1 - math.exp(-(dt1 + dt2) / tau)) * a
        assert abs(h_two - h_one) < 1e-12


class TestWiring:
    def test_dt_channel_changes_output(self):
        """The recurrence must read Δt: same values, different gaps ⇒
        different predictions (the production mt_lnn only sees Δt as a
        feature; here it also drives the decay)."""
        m = _fresh(n_layers=2, seed=4)
        x = torch.randn(3, 10, 4)
        x[..., -1] = x[..., -1].abs() + 0.05
        with torch.no_grad():
            y_real = m(x)
            x2 = x.clone()
            x2[..., -1] = x2[..., -1] * 8.0          # 8x longer gaps
            y_wide = m(x2)
        assert not torch.allclose(y_real, y_wide, atol=1e-6)

    def test_tau_receives_gradient(self):
        m = _fresh(n_layers=1, seed=5)
        x = torch.randn(2, 8, 4)
        x[..., -1] = x[..., -1].abs() + 0.05
        m(x).pow(2).mean().backward()
        assert m.log_tau_0.grad is not None
        assert torch.isfinite(m.log_tau_0.grad).all()
        assert float(m.log_tau_0.grad.abs().max()) > 0


class TestBudgetAndFactory:
    def test_param_band_matches_mt_lnn_gru(self):
        m = build("mt_lnn_dt", 78, 2, 128)
        m.inp = torch.nn.Linear(4, 78)
        n = sum(p.numel() for p in m.parameters())
        ref_gru = sum(p.numel() for p in build("gru", 78, 2).parameters())
        assert 0.5 * ref_gru < n < 1.5 * ref_gru, n

    def test_forward_backward_cpu_fast(self):
        m = _fresh(n_layers=2, seed=6)
        x = torch.randn(4, 16, 4)
        x[..., -1] = x[..., -1].abs() + 0.05
        t0 = time.time()
        for _ in range(3):
            y = m(x)
            loss = y.pow(2).mean()
            loss.backward()
        assert torch.isfinite(y).all()
        assert time.time() - t0 < 30.0

    def test_head_patchable_like_other_archs(self):
        """Driver scripts (air/synth) patch .inp/.head — must keep working."""
        m = build("mt_lnn_dt", 78, 2, 128)
        m.inp = torch.nn.Linear(12, 78)
        m.head = torch.nn.Linear(78, 12)
        x = torch.randn(2, 9, 12)
        x[..., -1] = x[..., -1].abs() + 0.05
        y = m(x)
        assert y.shape == (2, 12)
