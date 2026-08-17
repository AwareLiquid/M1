"""
tests/test_replay.py -- experience-replay buffers for streaming / continual
learning.

Each behaviour is pinned against the analytic guarantee, not just "it stores
stuff":

* ReservoirBuffer infers its field schema from the first add, keeps all items
  while under capacity, never exceeds capacity, and -- the load-bearing law --
  retains a UNIFORM sample of the whole stream: after n >> C items, every item's
  empirical retention frequency matches the theoretical C / n (Vitter Algorithm
  R). Rows stay aligned across parallel fields; draws are seed-deterministic.
* ClassBalancedReservoir routes by label, gives every class an equal slot budget
  regardless of arrival skew, and samples evenly across non-empty classes.
"""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.replay import (  # noqa: E402
    ClassBalancedReservoir,
    ReplayStats,
    ReservoirBuffer,
)


# --------------------------------------------------------------------------- #
# ReservoirBuffer -- schema + occupancy bookkeeping                           #
# --------------------------------------------------------------------------- #
def test_schema_inferred_and_fields_stay_aligned():
    buf = ReservoirBuffer(capacity=8, seed=0)
    for i in range(5):
        buf.add(x=torch.tensor([float(i), float(i)]), y=torch.tensor(i))
    assert len(buf) == 5
    snap = buf.snapshot()
    assert set(snap) == {"x", "y"}
    assert snap["x"].shape == (5, 2)
    assert snap["y"].shape == (5,)
    # row r must carry x == [r, r] AND y == r together (alignment)
    for r in range(5):
        xr = snap["x"][r]
        assert torch.equal(snap["y"][r], xr[0].long())
        assert torch.equal(xr, torch.tensor([float(snap["y"][r]), float(snap["y"][r])]))


def test_fill_phase_keeps_everything_then_caps():
    buf = ReservoirBuffer(capacity=4, seed=1)
    for i in range(4):                       # exactly fills
        buf.add(v=torch.tensor(float(i)))
    assert len(buf) == 4
    assert sorted(int(v) for v in buf.snapshot()["v"]) == [0, 1, 2, 3]
    for i in range(4, 100):                  # never exceeds capacity
        buf.add(v=torch.tensor(float(i)))
        assert len(buf) == 4
    assert buf.seen == 100


def test_stats_snapshot_is_a_frozen_dataclass():
    buf = ReservoirBuffer(capacity=3, seed=0)
    buf.add(a=torch.zeros(2), b=torch.tensor(1.0))
    s = buf.stats()
    assert isinstance(s, ReplayStats)
    assert (s.size, s.capacity, s.seen) == (1, 3, 1)
    assert s.fields == ("a", "b")
    with pytest.raises(Exception):
        s.size = 2                           # frozen


def test_zero_parameters_and_resettable():
    buf = ReservoirBuffer(capacity=4, seed=7)
    assert buf.n_parameters == 0
    for i in range(20):
        buf.add(v=torch.tensor(float(i)))
    buf.reset()
    assert len(buf) == 0 and buf.seen == 0
    assert buf.snapshot() == {}


# --------------------------------------------------------------------------- #
# ReservoirBuffer -- the uniformity invariant (Vitter Algorithm R)            #
# --------------------------------------------------------------------------- #
def test_reservoir_retention_is_uniform_over_the_stream():
    """After many independent runs, item i is retained with frequency ~ C / n,
    the SAME for every i -- the defining property of reservoir sampling."""
    C, n, runs = 10, 200, 4000
    counts = torch.zeros(n)
    for r in range(runs):
        buf = ReservoirBuffer(capacity=C, seed=r)
        for i in range(n):
            buf.add(idx=torch.tensor(i))     # store each item's stream index
        for v in buf.snapshot()["idx"]:
            counts[int(v)] += 1
    freq = counts / runs
    expected = C / n                          # = 0.05
    # every item -- the first and the last alike -- is kept ~5% of the time
    assert abs(float(freq.mean()) - expected) < 0.005
    assert float(freq.max() - freq.min()) < 0.05      # no position is favoured
    # specifically: the very first item is not starved vs the very last
    assert abs(float(freq[0] - freq[-1])) < 0.04


def test_draws_are_seed_deterministic():
    def run(seed):
        buf = ReservoirBuffer(capacity=5, seed=seed)
        for i in range(50):
            buf.add(v=torch.tensor(float(i)))
        return buf.sample(3)["v"]
    a, b, c = run(123), run(123), run(124)
    assert torch.equal(a, b)                  # same seed -> identical
    assert not torch.equal(a, c)              # different seed -> (almost surely) differs


# --------------------------------------------------------------------------- #
# ReservoirBuffer -- sampling                                                 #
# --------------------------------------------------------------------------- #
def test_sample_without_replacement_is_capped_and_aligned():
    buf = ReservoirBuffer(capacity=8, seed=2)
    for i in range(8):
        buf.add(x=torch.tensor([float(i)]), y=torch.tensor(i))
    out = buf.sample(100)                      # ask for more than we hold
    assert out["x"].shape == (8, 1) and out["y"].shape == (8,)
    # alignment survives the shuffle: x[k] == [y[k]]
    for k in range(8):
        assert float(out["x"][k][0]) == float(out["y"][k])
    # without replacement => the 8 drawn ys are a permutation of 0..7
    assert sorted(int(v) for v in out["y"]) == list(range(8))


def test_sample_with_replacement_can_exceed_size():
    buf = ReservoirBuffer(capacity=3, seed=0)
    for i in range(3):
        buf.add(v=torch.tensor(float(i)))
    out = buf.sample(10, replacement=True)
    assert out["v"].shape == (10,)
    assert set(int(v) for v in out["v"]).issubset({0, 1, 2})


def test_add_batch_matches_repeated_add():
    xb = torch.arange(6.0).reshape(6, 1)
    one = ReservoirBuffer(capacity=4, seed=9)
    for r in range(6):
        one.add(x=xb[r])
    many = ReservoirBuffer(capacity=4, seed=9)
    many.add_batch(x=xb)
    assert torch.equal(
        torch.sort(one.snapshot()["x"].flatten())[0],
        torch.sort(many.snapshot()["x"].flatten())[0],
    )
    assert one.seen == many.seen == 6


# --------------------------------------------------------------------------- #
# ReservoirBuffer -- error handling                                           #
# --------------------------------------------------------------------------- #
def test_errors_are_explicit():
    with pytest.raises(ValueError):
        ReservoirBuffer(capacity=0)
    buf = ReservoirBuffer(capacity=4, seed=0)
    with pytest.raises(IndexError):           # sampling an empty buffer
        buf.sample(1)
    buf.add(x=torch.zeros(2), y=torch.tensor(0))
    with pytest.raises(KeyError):             # schema drift
        buf.add(x=torch.zeros(2))             # missing 'y'
    with pytest.raises(ValueError):
        buf.sample(0)


# --------------------------------------------------------------------------- #
# ClassBalancedReservoir                                                      #
# --------------------------------------------------------------------------- #
def test_class_balanced_splits_budget_evenly():
    cb = ClassBalancedReservoir(capacity=12, n_classes=3, seed=0)
    assert cb.capacity == 12                  # 4 slots * 3 classes
    assert cb.n_parameters == 0
    # a wildly skewed stream: class 0 floods, class 2 trickles
    for i in range(100):
        cb.add(0, v=torch.tensor(float(i)))
    for i in range(100, 110):
        cb.add(1, v=torch.tensor(float(i)))
    for i in range(110, 113):
        cb.add(2, v=torch.tensor(float(i)))
    sizes = cb.per_class_sizes()
    assert sizes[0] == 4                       # capped at its even budget
    assert sizes[1] == 4
    assert sizes[2] == 3                       # only 3 ever seen -> all kept
    assert len(cb) == 11


def test_class_balanced_remainder_budget_is_dropped_evenly():
    # capacity 10 over 3 classes -> 3 slots each, 1 slot unused (fairness)
    cb = ClassBalancedReservoir(capacity=10, n_classes=3, seed=0)
    assert cb.capacity == 9
    for c in range(3):
        for i in range(20):
            cb.add(c, v=torch.tensor(float(i)))
    assert cb.per_class_sizes() == {0: 3, 1: 3, 2: 3}


def test_class_balanced_sample_spreads_and_labels():
    cb = ClassBalancedReservoir(capacity=12, n_classes=3, seed=1)
    for c in range(3):
        for i in range(20):
            cb.add(c, v=torch.tensor(float(c)))   # class c stores only value c
    out = cb.sample(9)
    assert out["v"].shape == (9,) and out["label"].shape == (9,)
    # value equals label (each class stored its own id) -> routing is correct
    assert torch.equal(out["v"].long(), out["label"])
    # 9 across 3 even classes -> exactly 3 each
    counts = torch.bincount(out["label"], minlength=3)
    assert counts.tolist() == [3, 3, 3]


def test_class_balanced_errors():
    with pytest.raises(ValueError):
        ClassBalancedReservoir(capacity=2, n_classes=3)   # < 1 slot/class
    cb = ClassBalancedReservoir(capacity=9, n_classes=3, seed=0)
    with pytest.raises(ValueError):
        cb.add(5, v=torch.tensor(0.0))                    # label out of range
    with pytest.raises(IndexError):
        cb.sample(1)                                       # empty


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
