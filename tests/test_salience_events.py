"""
tests/test_salience_events.py -- global-workspace ignition / state-change events.

Each behaviour is pinned against a deterministic synthetic salience stream: a
calm baseline ignites on a sharp departure and quiesces when it settles; a slow
drift the detector adapts to does NOT fire; the refractory period prevents event
storms; warmup is silent; and the world-model surprise channel is read in a
fully decoupled, duck-typed way.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.salience_events import (  # noqa: E402
    StateChangeEvent,
    SalienceEventDetector,
    world_model_surprise,
)


def _calm(n, level=0.1):
    return [level] * n


# --- ignition / quiescence ------------------------------------------------

def test_sharp_departure_ignites_then_quiesces():
    # calm baseline, a sustained high plateau, then back to calm
    stream = _calm(12) + [1.0] * 8 + _calm(12)
    det = SalienceEventDetector(ignite_z=3.0, release_z=1.0, refractory=2, warmup=8)
    events = det.observe(stream)
    kinds = [e.kind for e in events]
    assert "ignition" in kinds and "quiescence" in kinds
    ig = next(e for e in events if e.kind == "ignition")
    qu = next(e for e in events if e.kind == "quiescence")
    assert ig.step == 13                                # first high sample after warmup-seeded calm
    assert ig.step < qu.step                            # ignite before quiesce
    assert ig.salience >= 3.0                           # crossed the ignition z


def test_single_spike_ignites_exactly_once():
    stream = _calm(12) + [2.0] + _calm(12)
    det = SalienceEventDetector(ignite_z=3.0, release_z=1.0, refractory=3, warmup=8)
    events = det.observe(stream)
    assert sum(e.kind == "ignition" for e in events) == 1


# --- adaptation: slow drift must not fire ---------------------------------

def test_slow_drift_does_not_ignite():
    # a gentle ramp the EMA baseline tracks -> no surprise -> no events
    stream = [0.1 + 0.01 * i for i in range(80)]
    det = SalienceEventDetector(ignite_z=3.0, release_z=1.0, warmup=8, ema_decay=0.9)
    events = det.observe(stream)
    assert events == []


# --- refractory: no event storm -------------------------------------------

def test_refractory_suppresses_event_storm():
    # rapidly alternating calm/high would chatter without a refractory period
    stream = _calm(10)
    for _ in range(20):
        stream += [1.0, 0.1]
    no_refr = SalienceEventDetector(ignite_z=3.0, release_z=1.0, refractory=0, warmup=8)
    with_refr = SalienceEventDetector(ignite_z=3.0, release_z=1.0, refractory=4, warmup=8)
    assert len(with_refr.observe(stream)) < len(no_refr.observe(stream))


# --- hysteresis -----------------------------------------------------------

def test_hysteresis_holds_ignited_between_thresholds():
    det = SalienceEventDetector(ignite_z=3.0, release_z=1.0, refractory=0, warmup=8)
    # seed a tiny-variance baseline so z-scores are large and predictable
    det.observe(_calm(10))
    # craft: ignite high, then sit at a mid value whose z is between release/ignite
    # build a fresh stream so steps are clean
    stream = _calm(10) + [1.0, 1.0, 1.0]               # ignite & stay ignited
    events = det.observe(stream)
    # exactly one ignition, no quiescence while still elevated
    assert [e.kind for e in events] == ["ignition"]
    assert det.state == "ignited"


# --- warmup ---------------------------------------------------------------

def test_warmup_is_silent():
    # even with wild values during warmup, no events are emitted
    stream = [0.0, 5.0, 0.0, 9.0, 0.0, 7.0]
    det = SalienceEventDetector(warmup=len(stream))
    assert det.observe(stream) == []


# --- bookkeeping ----------------------------------------------------------

def test_zero_parameters_and_not_nn_module():
    det = SalienceEventDetector()
    assert det.n_parameters == 0
    assert not hasattr(det, "parameters")               # not an nn.Module


def test_reset_clears_state():
    det = SalienceEventDetector(warmup=4)
    det.observe(_calm(8) + [3.0])
    det.reset()
    assert det.state == "quiescent" and det.n_events == 0 and det.baseline == 0.0


def test_observe_is_deterministic():
    stream = _calm(12) + [1.5] * 5 + _calm(12)
    a = SalienceEventDetector().observe(stream)
    b = SalienceEventDetector().observe(stream)
    assert [(e.step, e.kind) for e in a] == [(e.step, e.kind) for e in b]


def test_event_payload_fields():
    stream = _calm(12) + [4.0] + _calm(4)
    ev = SalienceEventDetector(warmup=8).observe(stream)[0]
    assert isinstance(ev, StateChangeEvent)
    assert ev.kind == "ignition"
    assert ev.signal == pytest.approx(4.0)
    assert ev.z == ev.salience and ev.z > 0
    assert ev.baseline < ev.signal                      # fired above the baseline


# --- decoupled world-model surprise channel -------------------------------

class _FakeHead:
    def __init__(self, err):
        self.last_pred_error = err


def test_world_model_surprise_reads_duck_typed_head():
    assert world_model_surprise(_FakeHead(0.25)) == pytest.approx(0.25)

    import torch
    assert world_model_surprise(_FakeHead(torch.tensor(0.4))) == pytest.approx(0.4)


def test_world_model_surprise_requires_the_attribute():
    with pytest.raises(TypeError):
        world_model_surprise(object())


def test_reads_surprise_from_real_predictive_head():
    # the decoupled bridge works on the real head without importing model.py
    from mt_lnn.world_model import PredictiveStateHead
    import torch
    head = PredictiveStateHead(d_model=16)
    with torch.no_grad():
        head(torch.randn(2, 5, 16), compute_loss=True)
    s = world_model_surprise(head)
    assert isinstance(s, float) and 0.0 <= s <= 1.0


def test_detector_consumes_world_model_surprise_stream():
    # end-to-end: a surprise stream from heads drives the detector, decoupled
    det = SalienceEventDetector(warmup=6)
    heads = [_FakeHead(0.05) for _ in range(10)] + [_FakeHead(0.9)] + \
            [_FakeHead(0.05) for _ in range(6)]
    events = [det.update(world_model_surprise(h)) for h in heads]
    assert any(e is not None and e.kind == "ignition" for e in events)


# --- validation -----------------------------------------------------------

def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        SalienceEventDetector(ignite_z=1.0, release_z=2.0)   # release !< ignite
    with pytest.raises(ValueError):
        SalienceEventDetector(refractory=-1)
    with pytest.raises(ValueError):
        SalienceEventDetector(ema_decay=1.0)
    with pytest.raises(ValueError):
        SalienceEventDetector(warmup=0)


def test_module_does_not_import_model():
    import mt_lnn.salience_events as se
    src = open(se.__file__, encoding="utf-8").read()
    assert "import model" not in src and "from .model" not in src


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception:
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
