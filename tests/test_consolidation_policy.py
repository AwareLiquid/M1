"""consolidation_policy — purity + rules + the sleep-cycle hook.

The hook's off-position is asserted explicitly: ``nrem_replay`` with
``write_policy=None`` must behave exactly as before the hook existed, and
``write_policy=policy_write`` with no salience gate and explicit keys must
produce the SAME writes as the plain path (same keys, same count).
"""

from __future__ import annotations

import torch

from mt_lnn.memory_broker.consolidation_policy import policy_forget, policy_write
from mt_lnn.sleep_consolidation import SleepWakeConsolidator


# ---------------------------------------------------------------- policy_write

class TestPolicyWrite:
    def test_explicit_key_binds_value(self):
        pairs = policy_write({"key": "color", "content": "blue"})
        assert pairs == [("color", "blue")]

    def test_multi_keys_one_binding_each(self):
        pairs = policy_write({"keys": ["a", "b"], "value": "payload"})
        assert pairs == [("a", "payload"), ("b", "payload")]

    def test_content_addressable_fallback(self):
        assert policy_write({"content": "self keying"}) == [("self keying", "self keying")]

    def test_empty_record_yields_nothing(self):
        assert policy_write({}) == []
        assert policy_write({"key": "k"}) == []          # key but no payload
        assert policy_write({"salience": 0.9}) == []     # salience but no payload

    def test_salience_gate_drops_low_only(self):
        rec = {"key": "k", "content": "v", "salience": 0.2}
        assert policy_write(rec, min_salience=0.5) == []
        rec_hi = {"key": "k", "content": "v", "salience": 0.9}
        assert policy_write(rec_hi, min_salience=0.5) == [("k", "v")]

    def test_missing_or_nonnumeric_salience_is_ungated(self):
        assert policy_write({"key": "k", "content": "v"}, min_salience=0.5) == [("k", "v")]
        assert policy_write({"key": "k", "content": "v", "salience": "high"},
                            min_salience=0.5) == [("k", "v")]

    def test_bare_scalar_keys_field_is_one_key(self):
        # a malformed keys field degrades to one binding, never a crash
        assert policy_write({"keys": 7, "content": "v"}) == [(7, "v")]

    def test_typed_key_tensor_passes_through(self):
        k = torch.randn(8)
        pairs = policy_write({"key": k, "content": "v"})
        assert pairs[0][0] is k and pairs[0][1] == "v"

    def test_pure_never_mutates_input(self):
        rec = {"key": "k", "content": "v", "salience": 0.1}
        snapshot = dict(rec)
        policy_write(rec, min_salience=0.5)
        assert rec == snapshot


# ---------------------------------------------------------------- policy_forget

class TestPolicyForget:
    def test_update_returns_old_key(self):
        ev = {"type": "update",
              "old": {"key": "color", "content": "blue"},
              "new": {"key": "color", "content": "green"}}
        assert policy_forget(ev) == ["color"]

    def test_contradiction_returns_old_key(self):
        ev = {"type": "contradiction",
              "old": {"key": "cat", "value": "is friendly"},
              "new": {"key": "cat", "value": "bit me twice"}}
        assert policy_forget(ev) == ["cat"]

    def test_duplicate_and_identical_content_yield_nothing(self):
        assert policy_forget({"type": "duplicate",
                              "old": {"key": "k", "content": "x"},
                              "new": {"key": "k", "content": "x"}}) == []
        assert policy_forget({"old": {"key": "k", "content": "x"},
                              "new": {"key": "k2", "content": "x"}}) == []

    def test_no_old_key_yields_nothing(self):
        assert policy_forget({"new": {"key": "k", "content": "x"}}) == []
        assert policy_forget({}) == []
        assert policy_forget({"old": "not-a-mapping", "new": None}) == []

    def test_unknown_type_treated_as_update(self):
        ev = {"type": "mystery",
              "old": {"key": "k", "value": 1},
              "new": {"key": "k", "value": 2}}
        assert policy_forget(ev) == ["k"]

    def test_pure_never_mutates_input(self):
        ev = {"type": "update",
              "old": {"key": "k", "content": "a"},
              "new": {"key": "k", "content": "b"}}
        import copy
        snapshot = copy.deepcopy(ev)
        policy_forget(ev)
        assert ev == snapshot

    def test_deterministic_across_calls(self):
        ev = {"old": {"key": "k", "content": "a"}, "new": {"key": "k", "content": "b"}}
        assert policy_forget(ev) == policy_forget(ev) == ["k"]


# ------------------------------------------------------- sleep-cycle hook

class _SpyStore:
    def __init__(self):
        self.writes = []

    def write(self, key, content, meta=None):
        self.writes.append((key, content, meta))


class _Buf:
    def __init__(self, batch):
        self._batch = batch

    def __len__(self):
        return self._batch["keys"].shape[0]

    def sample(self, k, replacement=False):
        return {f: v[:k] for f, v in self._batch.items()}


class TestNremReplayHook:
    def _buffer(self):
        return _Buf({
            "keys": torch.arange(4).reshape(4, 1).float(),
            "texts": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
            "sal": torch.tensor([0.1, 0.9, 0.5, 0.3]),
        })

    def test_off_position_is_byte_equivalent(self):
        # write_policy=None must reproduce the plain path exactly
        cons = SleepWakeConsolidator(seed=0)
        spy = _SpyStore()
        out = cons.nrem_replay(self._buffer(), spy, key_field="keys",
                               content_field="texts", salience_field="sal")
        assert (out.replayed, out.consolidated) == (4, 4)
        # salience ranking: 0.9, 0.5, 0.3, 0.1 -> rows 20, 30, 40, 10
        assert [c.item() for _, c, _ in spy.writes] == [20.0, 30.0, 40.0, 10.0]

    def test_policy_with_explicit_keys_matches_plain_path(self):
        from mt_lnn.memory_broker.consolidation_policy import policy_write
        cons_plain, cons_hk = SleepWakeConsolidator(seed=0), SleepWakeConsolidator(seed=0)
        plain, hooked = _SpyStore(), _SpyStore()
        cons_plain.nrem_replay(self._buffer(), plain, key_field="keys",
                               content_field="texts")
        cons_hk.nrem_replay(self._buffer(), hooked, key_field="keys",
                            content_field="texts", write_policy=policy_write)
        assert plain.writes == hooked.writes   # same keys, same order, same meta

    def test_policy_salience_gate_drops_bindings(self):
        from mt_lnn.memory_broker.consolidation_policy import policy_write

        def gated(record):
            return policy_write(record, min_salience=0.5)

        cons = SleepWakeConsolidator(seed=0)
        spy = _SpyStore()
        out = cons.nrem_replay(self._buffer(), spy, key_field="keys",
                               content_field="texts", salience_field="sal",
                               write_policy=gated)
        # ranking unchanged (sal 0.9, 0.5, 0.3, 0.1); gate keeps sal>=0.5 only
        assert (out.replayed, out.consolidated) == (4, 2)
        assert [c.item() for _, c, _ in spy.writes] == [20.0, 30.0]

    def test_policy_derived_extra_bindings_are_written(self):
        def fanout(record):
            base = policy_write(record)
            return base + [("alias", record["content"])]

        cons = SleepWakeConsolidator(seed=0)
        spy = _SpyStore()
        out = cons.nrem_replay(self._buffer(), spy, key_field="keys",
                               content_field="texts", n_replay=1,
                               write_policy=fanout)
        assert out.consolidated == 2                    # base + alias binding
        assert spy.writes[1][0] == "alias"
