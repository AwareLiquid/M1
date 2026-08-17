"""
tests/test_failsafe.py -- input-dropout blind rollout + output circuit breaker.

Each behaviour is pinned against deterministic / analytic streams:

* BlindRolloutGuard passes live frames through, coasts on the world model's
  imagination through a brief dropout, and goes *dark* once the imagination's
  confidence drops below the floor or the blind budget is spent -- rather than
  hallucinating forever. Verified both with a tiny fake imagination (so the
  gating is analytic) and with the real LatentImagination/PredictiveStateHead.
* CircuitBreaker always hard-clamps (NaN/Inf scrub, bounds, slew) and trips to a
  debounced fallback when the raw command repeatedly violates the red-lines,
  closing again with bumpless transfer.
"""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.failsafe import (  # noqa: E402
    BlindRolloutGuard,
    GuardOutput,
    CircuitBreaker,
    BreakerResult,
    TopologyBreaker,
    TopologyResult,
)
from mt_lnn.imagination import LatentImagination, ImaginedTrajectory  # noqa: E402
from mt_lnn.world_model import PredictiveStateHead  # noqa: E402


# --------------------------------------------------------------------------- #
# fakes: an analytic stand-in for LatentImagination so guard gating is exact   #
# --------------------------------------------------------------------------- #


class _FakeHead:
    """Minimal head: online_proj is identity-ish so present latent is testable."""

    def __init__(self, P=4):
        self.proj_dim = P

        def _proj(h):
            return h                                     # identity projection

        self.online_proj = _proj


class _FakeImagination:
    """Returns a fixed trajectory with a prescribed confidence schedule."""

    def __init__(self, *, horizon, conf_schedule, P=4):
        self.horizon = horizon
        self.head = _FakeHead(P)
        self._conf = conf_schedule
        self.P = P
        self.calls = 0

    def imagine(self, hidden, horizon=None):
        H = horizon or self.horizon
        B = hidden.shape[0]
        self.calls += 1
        latents = torch.nn.functional.normalize(
            torch.arange(1, B * H * self.P + 1, dtype=torch.float32).reshape(B, H, self.P),
            dim=-1,
        )
        conf = torch.tensor(self._conf[:H], dtype=torch.float32).reshape(1, H).expand(B, H).contiguous()
        traj = ImaginedTrajectory(latents=latents, confidence=conf,
                                  novelty=torch.zeros(B, H), horizon=H)
        return traj


# --------------------------------------------------------------------------- #
# BlindRolloutGuard -- live pass-through                                       #
# --------------------------------------------------------------------------- #


def test_live_input_passes_through_normalised():
    g = BlindRolloutGuard(_FakeImagination(horizon=4, conf_schedule=[1, 1, 1, 1]))
    h = torch.tensor([[3.0, 0.0, 0.0, 4.0]])             # norm 5
    out = g.step(h)
    assert out.source == "live"
    assert out.confidence == 1.0
    assert out.blind_steps == 0
    assert pytest.approx(out.latent.norm().item(), abs=1e-5) == 1.0
    # unit vector in the 3-4 direction
    assert pytest.approx(out.latent[0, 0].item(), abs=1e-5) == 0.6


def test_live_resets_blind_counter():
    g = BlindRolloutGuard(_FakeImagination(horizon=4, conf_schedule=[0.9, 0.9, 0.9, 0.9]))
    g.step(torch.ones(1, 4))
    g.step(None)
    g.step(None)
    assert g.blind_steps == 2
    g.step(torch.ones(1, 4))
    assert g.blind_steps == 0


# --------------------------------------------------------------------------- #
# BlindRolloutGuard -- coasting on imagination                                 #
# --------------------------------------------------------------------------- #


def test_dropout_coasts_on_imagination_while_confident():
    g = BlindRolloutGuard(_FakeImagination(horizon=4, conf_schedule=[0.9, 0.8, 0.7, 0.6]),
                          confidence_floor=0.3)
    g.step(torch.ones(1, 4))                             # prime a good frame
    o1 = g.step(None)
    o2 = g.step(None)
    assert o1.source == "imagined" and o1.blind_steps == 1
    assert pytest.approx(o1.confidence, abs=1e-6) == 0.9
    assert o2.source == "imagined" and o2.blind_steps == 2
    assert pytest.approx(o2.confidence, abs=1e-6) == 0.8
    assert pytest.approx(o1.latent.norm().item(), abs=1e-5) == 1.0


def test_imagine_called_once_per_dropout_window():
    im = _FakeImagination(horizon=4, conf_schedule=[0.9, 0.9, 0.9, 0.9])
    g = BlindRolloutGuard(im, confidence_floor=0.3)
    g.step(torch.ones(1, 4))
    g.step(None)
    g.step(None)
    g.step(None)
    assert im.calls == 1                                 # imagined once, indexed 3x


def test_confidence_floor_goes_dark():
    g = BlindRolloutGuard(_FakeImagination(horizon=4, conf_schedule=[0.9, 0.5, 0.2, 0.1]),
                          confidence_floor=0.3)
    g.step(torch.ones(1, 4))
    assert g.step(None).source == "imagined"             # 0.9
    assert g.step(None).source == "imagined"             # 0.5
    dark = g.step(None)                                  # 0.2 < floor
    assert dark.source == "dark"
    assert dark.latent is None
    assert dark.confidence == pytest.approx(0.2, abs=1e-6)


def test_blind_budget_cap_goes_dark():
    g = BlindRolloutGuard(_FakeImagination(horizon=8, conf_schedule=[0.9] * 8),
                          confidence_floor=0.0, max_blind_steps=2)
    g.step(torch.ones(1, 4))
    assert g.step(None).source == "imagined"             # blind 1
    assert g.step(None).source == "imagined"             # blind 2
    assert g.step(None).source == "dark"                 # blind 3 > cap


def test_default_cap_is_horizon():
    g = BlindRolloutGuard(_FakeImagination(horizon=2, conf_schedule=[0.9, 0.9]),
                          confidence_floor=0.0)
    assert g.max_blind_steps == 2
    g.step(torch.ones(1, 4))
    assert g.step(None).source == "imagined"
    assert g.step(None).source == "imagined"
    assert g.step(None).source == "dark"                 # past horizon


def test_dark_before_any_good_frame():
    g = BlindRolloutGuard(_FakeImagination(horizon=4, conf_schedule=[0.9] * 4))
    out = g.step(None)                                   # never saw input
    assert out.source == "dark"
    assert out.latent is None


def test_recovery_after_dark_re_coasts():
    g = BlindRolloutGuard(_FakeImagination(horizon=2, conf_schedule=[0.9, 0.9]),
                          confidence_floor=0.0)
    g.step(torch.ones(1, 4))
    g.step(None); g.step(None); g.step(None)             # exhaust + dark
    g.step(torch.ones(1, 4) * 2)                         # input returns
    out = g.step(None)
    assert out.source == "imagined" and out.blind_steps == 1


def test_guard_zero_parameters_and_reset():
    g = BlindRolloutGuard(_FakeImagination(horizon=4, conf_schedule=[0.9] * 4))
    assert g.n_parameters == 0
    g.step(torch.ones(1, 4)); g.step(None)
    g.reset()
    assert g.blind_steps == 0
    assert g._last_hidden is None


def test_guard_rejects_non_imagination():
    with pytest.raises(TypeError):
        BlindRolloutGuard(object())


def test_guard_rejects_bad_floor():
    with pytest.raises(ValueError):
        BlindRolloutGuard(_FakeImagination(horizon=4, conf_schedule=[0.9] * 4),
                          confidence_floor=1.0)


# --------------------------------------------------------------------------- #
# BlindRolloutGuard -- real LatentImagination / PredictiveStateHead bridge     #
# --------------------------------------------------------------------------- #


def test_guard_with_real_imagination_bridge():
    torch.manual_seed(0)
    head = PredictiveStateHead(d_model=16)
    head.eval()
    im = LatentImagination(head, horizon=4)
    g = BlindRolloutGuard(im, confidence_floor=0.0)      # floor off: exercise indexing
    hidden = torch.randn(2, 16)
    live = g.step(hidden)
    assert live.source == "live"
    assert pytest.approx(live.latent.norm(dim=-1).mean().item(), abs=1e-4) == 1.0
    blind = g.step(None)
    assert blind.source == "imagined"
    assert blind.latent.shape == (2, head.proj_dim)
    assert pytest.approx(blind.latent.norm(dim=-1).mean().item(), abs=1e-4) == 1.0
    assert 0.0 <= blind.confidence <= 1.0


# --------------------------------------------------------------------------- #
# CircuitBreaker -- unconditional hard clamp                                   #
# --------------------------------------------------------------------------- #


def test_clean_command_passes():
    b = CircuitBreaker(lo=-10, hi=10, max_rate=5)
    r = b.step(3.0)
    assert r.value == 3.0
    assert not r.tripped and not r.overridden and r.reason == ""


def test_bounds_clamp_every_tick_even_before_trip():
    b = CircuitBreaker(lo=-1, hi=1, max_rate=None, trip_after=5)
    r = b.step(100.0)
    assert r.value == 1.0                                # clamped immediately
    assert r.reason == "bounds"
    assert not r.tripped                                 # one tick < trip_after


def test_nonfinite_is_scrubbed_to_last_safe():
    b = CircuitBreaker(lo=-10, hi=10, max_rate=None, initial=2.0)
    r = b.step(float("nan"))
    assert r.value == 2.0                                # held last safe
    assert r.reason == "nonfinite"
    r2 = b.step(float("inf"))
    assert r2.value == 2.0
    assert r2.reason == "nonfinite"


def test_rate_limit_caps_slew():
    b = CircuitBreaker(lo=-100, hi=100, max_rate=2.0, initial=0.0, trip_after=10)
    r = b.step(50.0)
    assert r.value == 2.0                                # only +max_rate per tick
    assert r.reason == "rate"
    r2 = b.step(50.0)
    assert r2.value == 4.0


# --------------------------------------------------------------------------- #
# CircuitBreaker -- debounced trip + fallback                                  #
# --------------------------------------------------------------------------- #


def test_trips_after_debounce_and_holds():
    b = CircuitBreaker(lo=-1, hi=1, max_rate=None, trip_after=3, initial=0.5)
    b.step(100.0)                                        # bad 1 -> clamped to 1.0 (emitted)
    b.step(100.0)                                        # bad 2 -> last_safe now 1.0
    assert not b.tripped
    r = b.step(100.0)                                    # bad 3 -> trip, hold last emitted
    assert b.tripped and r.tripped and r.overridden
    assert r.value == 1.0                                # holds the last *emitted* safe value


def test_fallback_callable_used_when_tripped():
    calls = []

    def fb(last):
        calls.append(last)
        return 0.0                                       # PID-style "return to setpoint 0"

    b = CircuitBreaker(lo=-10, hi=10, max_rate=None, fallback=fb, trip_after=1, initial=5.0)
    r = b.step(float("nan"))                             # trip immediately
    assert r.tripped and r.overridden
    assert r.value == 0.0
    assert calls == [5.0]


def test_closes_again_with_bumpless_transfer():
    b = CircuitBreaker(lo=-100, hi=100, max_rate=3.0, trip_after=1, reset_after=2, initial=0.0)
    b.step(float("nan"))                                 # trip, hold 0.0
    assert b.tripped
    b.step(1.0)                                          # clean 1 (within rate)
    r = b.step(2.0)                                      # clean 2 -> close
    assert not b.tripped
    # bumpless: even a huge command after closing is slew-limited
    r3 = b.step(1000.0)
    assert abs(r3.value - r.value) <= 3.0 + 1e-9


def test_reason_reflects_raw_even_while_tripped():
    b = CircuitBreaker(lo=-1, hi=1, max_rate=None, trip_after=1, initial=0.0)
    b.step(5.0)                                          # trip on bounds
    r = b.step(7.0)
    assert r.tripped
    assert r.reason == "bounds"                          # raw still classified


def test_breaker_zero_parameters_and_reset():
    b = CircuitBreaker(lo=-1, hi=1, max_rate=None, trip_after=1, initial=0.3)
    assert b.n_parameters == 0
    b.step(100.0)
    assert b.tripped
    b.reset()
    assert not b.tripped
    assert b.last_safe == 0.3


def test_breaker_accepts_tensor_scalar():
    b = CircuitBreaker(lo=-10, hi=10, max_rate=None)
    r = b.step(torch.tensor(3.0))
    assert r.value == 3.0


@pytest.mark.parametrize("kwargs", [
    dict(lo=1, hi=1),                                    # lo !< hi
    dict(lo=-1, hi=1, max_rate=0.0),                     # bad rate
    dict(lo=-1, hi=1, trip_after=0),                     # bad trip
    dict(lo=-1, hi=1, reset_after=0),                    # bad reset
    dict(lo=-1, hi=1, initial=5.0),                      # initial out of bounds
])
def test_breaker_validates_config(kwargs):
    with pytest.raises(ValueError):
        CircuitBreaker(**kwargs)


def test_breaker_is_deterministic():
    stream = [3.0, float("nan"), 100.0, -100.0, 2.0, 2.0, 2.0]

    def run():
        b = CircuitBreaker(lo=-5, hi=5, max_rate=2.0, trip_after=2, reset_after=2)
        return [b.step(x).value for x in stream]

    assert run() == run()


# --------------------------------------------------------------------------- #
# TopologyBreaker -- representation-shape tripwire                            #
# --------------------------------------------------------------------------- #

_f64 = torch.float64


def _clusters(spread, jitter, seed):
    """Three blobs of 5 points; ``spread`` apart, ``jitter`` radius."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.0, 0.0], [spread, 0.0], [spread / 2, spread]], dtype=_f64)
    return torch.cat([c + jitter * torch.randn(5, 2, generator=g, dtype=_f64)
                      for c in centres])


def test_topology_breaker_passes_benign_jitter():
    ref = _clusters(5.0, 0.15, 0)
    jit = _clusters(5.0, 0.15, 1)
    br = TopologyBreaker(srtd_threshold=0.3, reference=ref)
    for _ in range(5):
        r = br.step(jit)
        assert isinstance(r, TopologyResult)
        assert r.reason == ""                    # same topology
        assert not r.tripped
        assert r.srtd < 0.3


def test_topology_breaker_trips_on_collapse_after_debounce():
    ref = _clusters(5.0, 0.15, 0)
    collapsed = _clusters(0.4, 0.15, 0)          # clusters fused
    br = TopologyBreaker(srtd_threshold=0.3, trip_after=2, reference=ref)
    r1 = br.step(collapsed)
    assert r1.reason == "srtd" and not r1.tripped  # 1 bad tick: not yet
    r2 = br.step(collapsed)
    assert r2.tripped                              # 2nd consecutive bad -> trip
    assert r2.srtd > 0.3


def test_topology_breaker_closes_after_clean_run():
    ref = _clusters(5.0, 0.15, 0)
    jit = _clusters(5.0, 0.15, 1)
    collapsed = _clusters(0.4, 0.15, 0)
    br = TopologyBreaker(srtd_threshold=0.3, trip_after=2, reset_after=3, reference=ref)
    br.step(collapsed); br.step(collapsed)
    assert br.tripped
    states = [br.step(jit).tripped for _ in range(3)]
    assert states == [True, True, False]           # closes on the 3rd clean tick


def test_topology_breaker_betti_check_flags_component_count_change():
    ref = _clusters(5.0, 0.15, 0)
    collapsed = _clusters(0.4, 0.15, 0)
    # SRTD threshold loose enough to NOT fire; the Betti-0 change (3 -> 1) catches it
    br = TopologyBreaker(srtd_threshold=10.0, eps=2.5, trip_after=1, reference=ref)
    clean = br.step(ref)
    assert clean.reason == "" and clean.betti0 == 3
    bad = br.step(collapsed)
    assert bad.reason == "betti" and bad.betti0 == 1 and bad.tripped


def test_topology_breaker_auto_latches_first_cloud():
    ref = _clusters(5.0, 0.15, 0)
    br = TopologyBreaker(srtd_threshold=0.3)
    assert not br.has_reference
    first = br.step(ref)
    assert first.reason == "noref" and not first.tripped
    assert br.has_reference
    assert not br.step(_clusters(5.0, 0.15, 1)).tripped   # benign jitter is clean


def test_topology_breaker_zero_parameters_and_reset():
    ref = _clusters(5.0, 0.15, 0)
    br = TopologyBreaker(srtd_threshold=0.3, reference=ref)
    assert br.n_parameters == 0
    br.step(_clusters(0.4, 0.15, 0)); br.step(_clusters(0.4, 0.15, 0))
    br.reset()
    assert not br.tripped and not br.has_reference


@pytest.mark.parametrize("kwargs", [
    {"srtd_threshold": 0.0},
    {"srtd_threshold": -1.0},
    {"srtd_threshold": 1.0, "eps": 0.0},
    {"srtd_threshold": 1.0, "trip_after": 0},
    {"srtd_threshold": 1.0, "reset_after": 0},
])
def test_topology_breaker_validates_config(kwargs):
    with pytest.raises(ValueError):
        TopologyBreaker(**kwargs)


def test_topology_breaker_is_deterministic():
    ref = _clusters(5.0, 0.15, 0)
    seq = [_clusters(5.0, 0.15, 1), _clusters(0.4, 0.15, 0),
           _clusters(0.4, 0.15, 0), _clusters(5.0, 0.15, 2)]

    def run():
        br = TopologyBreaker(srtd_threshold=0.3, eps=2.5, trip_after=2, reference=ref)
        return [(round(br.step(c).srtd, 6), br.step(c).tripped) for c in seq]

    assert run() == run()
