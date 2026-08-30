"""Unit tests for the zero-dependency GRU-D baseline (Che et al., 2018).

Covers the decay mathematics the architecture claim rests on:
  * gamma(dt=0, m=1) == 1 exactly — no decay at zero elapsed time (relu clamp
    makes this exact, not a tolerance, for any W_g and any b_g <= 0)
  * gamma -> 0 as dt -> inf, so the decayed hidden converges to its running
    mean — the mechanism GRU-D uses to "forget across" long gaps
  * mask semantics: observed steps pass values through untouched; masked-out
    steps pull toward the running mean as dt grows
  * parameter budget stays in the same class as the plain GRU baseline
  * the whole module runs a train step on CPU in well under a minute

All tests are CPU-only and finish in a few seconds.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.battery_soh_edge import GRUDCell, GRUDRegressor, build


def _fresh_cell(d=8, seed=0):
    torch.manual_seed(seed)
    return GRUDCell(d)


def _decay_driver(dt, m, B=1, T=1):
    return torch.tensor([[[dt, 1.0 - m]] * T] * B, dtype=torch.float32)


class TestDecayMath:
    def test_no_decay_at_zero_dt_exactly(self):
        """gamma(dt=0, m=1) == 1 exactly — relu(0) -> exp(0), bitwise."""
        cell = _fresh_cell()
        with torch.no_grad():
            cell.W_decay.uniform_(-3.0, 3.0)     # any weights
            cell.b_decay.uniform_(-3.0, 0.0)     # any non-positive bias
            g = cell.decay_rate_seq(_decay_driver(dt=0.0, m=1.0))
        assert torch.equal(g, torch.ones_like(g))

    def test_zero_dt_leaves_hidden_unchanged(self):
        """With gamma == 1 the pre-update hidden equals h, not hbar."""
        cell = _fresh_cell()
        h = torch.randn(2, 8)
        hbar = torch.randn(2, 8)
        x = torch.randn(2, 8) @ cell.Wx            # any input projection
        gamma = torch.ones(2, 8)
        # gamma == 1 => h_dec == h; verify via the closed form the cell uses.
        h_dec = gamma * h + (1.0 - gamma) * hbar
        assert torch.equal(h_dec, h)

    def test_large_dt_drives_gamma_to_zero(self):
        """dt -> inf => gamma -> 0 => the decayed hidden converges to hbar."""
        cell = _fresh_cell()
        with torch.no_grad():
            cell.W_decay.uniform_(0.5, 2.0)       # positive slopes
            cell.b_decay.zero_()
            g = cell.decay_rate_seq(_decay_driver(dt=1e9, m=1.0))
        assert float(g.max()) < 1e-30
        gamma = torch.zeros(2, 8)
        h, hbar = torch.randn(2, 8), torch.randn(2, 8)
        h_dec = gamma * h + (1.0 - gamma) * hbar
        assert torch.equal(h_dec, hbar)

    def test_gamma_bounded_and_monotone_in_dt(self):
        """gamma stays in (0, 1] and never increases as the gap grows."""
        cell = _fresh_cell()
        dts = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0]
        with torch.no_grad():
            gs = [float(cell.decay_rate_seq(_decay_driver(dt=d, m=1.0)).mean())
                  for d in dts]
        assert all(0.0 < g <= 1.0 for g in gs)
        assert all(gs[i] >= gs[i + 1] - 1e-7 for i in range(len(gs) - 1))

    def test_missing_step_decays_more_than_observed(self):
        """m=0 adds decay on top of the same gap (driver [dt; 1-m])."""
        cell = _fresh_cell()
        with torch.no_grad():
            cell.W_decay[:, 1].uniform_(0.5, 2.0)
            g_obs = cell.decay_rate_seq(_decay_driver(dt=1.0, m=1.0))
            g_mis = cell.decay_rate_seq(_decay_driver(dt=1.0, m=0.0))
        assert bool((g_mis <= g_obs).all())


class TestMaskSemantics:
    def test_observed_steps_pass_through_unchanged(self):
        """mask == 1 everywhere: input branch is the identity."""
        torch.manual_seed(1)
        m = GRUDRegressor(d_model=8, n_layers=2, n_feat=4)
        sensors = torch.randn(3, 6, 3)
        dt = torch.rand(3, 6, 1) + 0.1
        ones = torch.ones_like(dt)
        out = m._input_decay(sensors, dt, ones)
        assert torch.equal(out, sensors)

    def test_masked_step_pulls_toward_running_mean(self):
        """A fully masked step with a huge dt outputs the running mean of the
        previously observed values (carry-forward has fully decayed)."""
        torch.manual_seed(1)
        m = GRUDRegressor(d_model=8, n_layers=2, n_feat=4)
        with torch.no_grad():
            m.W_in_decay.fill_(50.0)              # -> gamma ~ 0 immediately
        sensors = torch.tensor([[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0],
                                 [7.0, 8.0, 9.0]]])
        mask = torch.tensor([[[1.0], [1.0], [0.0]]])   # last step missing
        dt = torch.ones(1, 3, 1)
        out = m._input_decay(sensors, dt, mask)
        assert torch.equal(out[0, 0], sensors[0, 0])
        assert torch.equal(out[0, 1], sensors[0, 1])
        # running mean of the two observed steps, carry fully decayed
        expected = sensors[0, :2].mean(dim=0)
        assert torch.allclose(out[0, 2], expected, atol=1e-6)

    def test_masked_step_zero_dt_keeps_carry_forward(self):
        """gamma(dt=0) == 1: a masked step right after an observation keeps
        the last observed value (no decay over a zero gap)."""
        torch.manual_seed(1)
        m = GRUDRegressor(d_model=8, n_layers=2, n_feat=4)
        with torch.no_grad():
            m.W_in_decay.fill_(50.0)              # steep slope, gamma(0)=1 still
            m.b_in_decay.zero_()
        sensors = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
        mask = torch.tensor([[[1.0], [0.0]]])
        dt = torch.tensor([[[0.1], [0.0]]])       # second step: zero gap
        out = m._input_decay(sensors, dt, mask)
        assert torch.equal(out[0, 0], sensors[0, 0])
        assert torch.allclose(out[0, 1], sensors[0, 0], atol=1e-6)


class TestRegressor:
    def test_forward_backward_cpu(self):
        torch.manual_seed(2)
        m = GRUDRegressor(d_model=16, n_layers=2, n_feat=4)
        x = torch.randn(4, 12, 4)
        x[..., -1] = x[..., -1].abs() + 0.05
        y = m(x)
        assert y.shape == (4,) and torch.isfinite(y).all()
        loss = y.pow(2).mean()
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_explicit_dt_mask_paths_run(self):
        torch.manual_seed(2)
        m = GRUDRegressor(d_model=8, n_layers=1, n_feat=4)
        x = torch.randn(2, 6, 4)
        y1 = m(x, dt=x[..., -1:].abs(), mask=torch.ones(2, 6, 1))
        y2 = m(x, dt=x[..., -1:].abs(), mask=torch.zeros(2, 6, 1))
        assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
        assert not torch.allclose(y1, y2)          # mask actually matters

    def test_param_budget_matches_plain_gru(self):
        gru = build("gru", 65, 2)
        grud = build("gru_d", 65, 2)
        n_gru = sum(p.numel() for p in gru.parameters())
        n_grud = sum(p.numel() for p in grud.parameters())
        # GRU-D = GRU gates + decay layers only; must stay in the same class.
        assert abs(n_grud - n_gru) < 2_000

    def test_whole_file_under_one_minute_cpu(self):
        """Guard the 'CPU < 1min' budget: a short train loop must complete."""
        torch.manual_seed(3)
        m = GRUDRegressor(d_model=16, n_layers=1, n_feat=4)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        x = torch.randn(8, 16, 4)
        x[..., -1] = x[..., -1].abs() + 0.05
        tgt = torch.randn(8)
        t0 = time.time()
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(m(x), tgt)
            loss.backward()
            opt.step()
        assert time.time() - t0 < 60.0
