"""ParametricMemory unit tests (CPU, fast — no downloads, no LM).

Covers the four properties the parametric-memory claim rests on:
  1. write -> recall round-trips (token-id path = the 0.56-anchor space,
     text path through the default hash encoder);
  2. snapshot/save -> load/restore is BIT-EXACT through session_state.py's
     JSON envelope (the cross-session persistence claim's entry ticket);
  3. forget is surgical — the target binding drops to <= chance while every
     other binding survives, and a whole-session forget zeroes everything;
  4. state_bytes is CONSTANT in the number of writes (the O(1) claim), and
     the delta-vs-sum write rules behave as their mechanisms predict
     (delta resolves a re-written key to the LATEST value; sum blends).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.parametric_memory import ParametricMemory, _hash_vec

# Fixed token-id ranges, same shape as the cross-window recall protocol
# (disjoint key/value vocab slices; chance = 1 / |value range|).
KEYS = list(range(5000, 5000 + 16))
VALS = list(range(7000, 7000 + 16))
CHANCE = 1.0 / 1000.0


def make_mem(**kw):
    return ParametricMemory(d_mem=128, vocab_size=9000, seed=0, **kw)


def write_bindings(mem, n=16, rule=None):
    for k, v in zip(KEYS[:n], VALS[:n]):
        mem.write("s", key=k, value=v, update_rule=rule)


def test_token_roundtrip_sum_and_delta():
    for rule in ("sum", "delta"):
        mem = make_mem(update_rule=rule)
        write_bindings(mem, rule=rule)
        hits = 0
        for k, v in zip(KEYS, VALS):
            got = mem.recall("s", query=k, top_k=1, candidates=range(7000, 8000))
            hits += int(got[0][0] == v)
        assert hits == len(KEYS), f"{rule}: {hits}/{len(KEYS)} recalled"


def test_text_roundtrip_and_lexicon_lru_bound():
    mem = make_mem()
    pairs = [("favorite color", "blue"), ("hometown", "Osaka"),
             ("anniversary", "June 3"), ("job title", "nurse")]
    for k, v in pairs:
        mem.write("t", key=k, value=v)
    for k, v in pairs:
        got = mem.recall("t", query=k, top_k=1)
        assert got[0][0] == v, f"{k!r} -> {got}"
    # bounded lexicon: capacity 2 keeps the 2 most recent values decodable
    mem_small = ParametricMemory(d_mem=64, vocab_size=100, seed=1,
                                 lexicon_capacity=2)
    mem_small.write("t", key="a", value="v1")
    mem_small.write("t", key="b", value="v2")
    mem_small.write("t", key="c", value="v3")
    assert len(mem_small.snapshot("t")["lexicon"]) == 2


def test_snapshot_bit_exact_across_fresh_instance(tmp_path):
    mem = make_mem(update_rule="sum")
    write_bindings(mem, n=12)
    mem.write("s", key="note", value="hello")     # mixed text binding
    path = str(tmp_path / "s1.json")
    mem.save("s", path)

    fresh = make_mem(update_rule="sum")
    fresh.load("s", path)
    a, b = mem.snapshot("s"), fresh.snapshot("s")
    # base64 of raw fp32 bytes: string equality IS bit equality
    assert a["F"] == b["F"] and a["z"] == b["z"]
    assert [v for v, _ in b["lexicon"]] == ["hello"]
    # recall behaves identically after the round-trip
    for k, v in zip(KEYS[:12], VALS[:12]):
        assert mem.recall("s", k, 1, range(7000, 8000)) == \
            fresh.recall("s", k, 1, range(7000, 8000))
    assert fresh.recall("s", "note", 1)[0][0] == "hello"


def test_forget_is_surgical():
    mem = make_mem()
    write_bindings(mem, n=12)
    target_key, target_val = KEYS[3], VALS[3]
    assert mem.recall("s", target_key, 1, range(7000, 8000))[0][0] == target_val

    assert mem.forget("s", key=target_key) is True

    # target: the erased value must not come back (score 0 / None, or a
    # different id) — i.e. <= chance on this binding, never a stale answer.
    got = mem.recall("s", target_key, 1, range(7000, 8000))
    assert got[0][0] != target_val, f"stale recall after forget: {got}"

    # everyone else survives
    for k, v in zip(KEYS[:12], VALS[:12]):
        if k == target_key:
            continue
        assert mem.recall("s", k, 1, range(7000, 8000))[0][0] == v, \
            f"forget({target_key}) collateral damage on {k}"


def test_forget_target_drops_to_random_level():
    """Over many trials the erased binding's hit rate must sit at chance."""
    hits = 0
    trials = 40
    for t in range(trials):
        mem = ParametricMemory(d_mem=128, vocab_size=9000, seed=100 + t)
        keys = list(range(5000, 5000 + 8))
        vals = [7000 + ((t * 7 + i * 113) % 1000) for i in range(8)]
        for k, v in zip(keys, vals):
            mem.write("s", key=k, value=v)
        mem.forget("s", key=keys[0])
        got = mem.recall("s", keys[0], 1, range(7000, 8000))[0][0]
        hits += int(got == vals[0])
    assert hits / trials <= max(CHANCE * 5, 0.05), \
        f"erased binding recalled {hits}/{trials} — forget leaks"


def test_forget_whole_session_zeroes():
    mem = make_mem()
    write_bindings(mem)
    assert mem.forget("s") is True
    assert mem.recall("s", KEYS[0], 1, range(7000, 8000)) == [(None, 0.0)]
    assert mem.state_bytes("s") == mem.state_bytes("s")   # still defined
    # forgotten session id is reusable
    mem.write("s", key=KEYS[0], value=VALS[1])
    assert mem.recall("s", KEYS[0], 1, range(7000, 8000))[0][0] == VALS[1]


def test_state_bytes_constant_in_writes():
    mem = make_mem()
    assert mem.state_bytes("s") == 0            # unknown session
    mem.write("s", key=KEYS[0], value=VALS[0])
    one = mem.state_bytes("s")
    for i in range(1, 1000):
        mem.write("s", key=KEYS[i % 16], value=VALS[i % 16])
        if i in (10, 100, 999):
            assert mem.state_bytes("s") == one, f"state grew at write {i}"
    assert one == (128 * 128 + 128) * 4         # (d^2 + d) float32


def test_delta_resolves_conflict_sum_blends():
    """Mechanism preview of the conflict-resolution benchmark: re-writing a
    key under delta must answer the LATEST value; under sum the read carries
    both values (recency-blind blend)."""
    v1, v2 = 7001, 7997
    dmem = make_mem(update_rule="delta")
    dmem.write("c", key=KEYS[0], value=v1)
    dmem.write("c", key=KEYS[0], value=v2)
    got = dmem.recall("c", KEYS[0], 1, range(7000, 8000))
    assert got[0][0] == v2, f"delta should answer the LATEST value, got {got}"

    smem = make_mem(update_rule="sum")
    smem.write("c", key=KEYS[0], value=v1)
    smem.write("c", key=KEYS[0], value=v2)
    both = smem.recall("c", KEYS[0], 1000, range(7000, 8000))
    score = {val: sc for val, sc in both}
    assert score[v1] > 0.3 and score[v2] > 0.3, \
        f"sum read should blend BOTH writes, got {score[v1]:.2f}/{score[v2]:.2f}"
    assert abs(score[v1] - score[v2]) < 0.2, "sum should be ~recency-blind"


def test_update_rule_validation_and_override():
    try:
        ParametricMemory(update_rule="sgd")
        assert False, "bad update_rule accepted"
    except ValueError:
        pass
    mem = make_mem(update_rule="sum")
    try:
        mem.write("s", key=1, value=2, update_rule="adam")
        assert False, "bad per-write rule accepted"
    except ValueError:
        pass
    # per-write override lands: one delta write flips the read rule
    mem.write("s", key=KEYS[0], value=VALS[0], update_rule="delta")
    assert mem.snapshot("s")["read_rule"] == "delta"


def test_hash_vec_deterministic_and_orthogonal():
    a = _hash_vec("alpha", 128)
    b = _hash_vec("beta", 128)
    assert torch.equal(a, _hash_vec("alpha", 128))
    assert abs(float(a @ b)) < 0.35    # distinct texts near-orthogonal
    assert abs(float(a.norm()) - 1.0) < 1e-4


if __name__ == "__main__":
    import tempfile
    for fn in [test_token_roundtrip_sum_and_delta,
               test_text_roundtrip_and_lexicon_lru_bound,
               test_forget_is_surgical,
               test_forget_target_drops_to_random_level,
               test_forget_whole_session_zeroes,
               test_state_bytes_constant_in_writes,
               test_delta_resolves_conflict_sum_blends,
               test_update_rule_validation_and_override,
               test_hash_vec_deterministic_and_orthogonal]:
        fn()
    with tempfile.TemporaryDirectory() as d:
        test_snapshot_bit_exact_across_fresh_instance(Path(d))
    print("all parametric memory tests passed")
