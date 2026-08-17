"""
test_memory.py — Tests for SQLite-backed persistent recurrent state.

Run:  python -m pytest tests/test_memory.py -v
"""

import sys
import warnings
import torch

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel, ModelCacheStruct, SessionMemory
from mt_lnn.memory import _tensors_to_bytes, _bytes_to_tensors
from mt_lnn.streaming import streaming_inference


def small_cfg():
    return MTLNNConfig(
        vocab_size=200, max_seq_len=32, d_model=128,
        n_layers=2, n_heads=4, n_kv_heads=2, d_head=32,
        dropout=0.0, attention_dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Unit: tensor serialisation round-trip
# ---------------------------------------------------------------------------

def test_tensor_roundtrip_all_none():
    tensors = [None, None, None]
    restored = _bytes_to_tensors(_tensors_to_bytes(tensors))
    assert restored == tensors


def test_tensor_roundtrip_mixed():
    t0 = torch.randn(1, 4, 10)
    t1 = None
    t2 = torch.zeros(1, 4, 10)
    restored = _bytes_to_tensors(_tensors_to_bytes([t0, t1, t2]))
    assert restored[1] is None
    assert torch.allclose(restored[0], t0)
    assert torch.allclose(restored[2], t2)


# ---------------------------------------------------------------------------
# Unit: SessionMemory CRUD (in-memory SQLite)
# ---------------------------------------------------------------------------

def test_session_not_found():
    mem = SessionMemory(":memory:")
    assert mem.load("nonexistent") is None
    assert mem.session_info("nonexistent") is None
    mem.close()


def test_save_and_load():
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache: ModelCacheStruct = out["cache"]

    mem = SessionMemory(":memory:")
    mem.save("sess_a", cache, token_count=8)

    # Info round-trip
    info = mem.session_info("sess_a")
    assert info["session_id"] == "sess_a"
    assert info["token_count"] == 8

    # Load and verify shapes
    h_states = mem.load("sess_a")
    assert h_states is not None
    assert len(h_states) == cfg.n_layers
    for h in h_states:
        if h is not None:
            assert h.shape[0] == 1  # batch size

    mem.close()


def test_overwrite():
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out1 = model(ids, use_cache=True)
        out2 = model(ids, use_cache=True)

    mem = SessionMemory(":memory:")
    mem.save("sess_b", out1["cache"], token_count=4)
    mem.save("sess_b", out2["cache"], token_count=8)

    assert mem.session_info("sess_b")["token_count"] == 8
    mem.close()


def test_delete():
    mem = SessionMemory(":memory:")
    # Create a fake cache with None layers
    cache = ModelCacheStruct()
    # Need at least one layer entry (even None h)
    cache.layers.append((None, torch.zeros(1, 2, 4), None))
    mem.save("sess_del", cache)
    assert mem.delete("sess_del") is True
    assert mem.delete("sess_del") is False  # already gone
    mem.close()


def test_list_sessions():
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        cache = model(ids, use_cache=True)["cache"]

    mem = SessionMemory(":memory:")
    mem.save("alpha", cache, token_count=10)
    mem.save("beta", cache, token_count=20)
    sessions = mem.list_sessions()
    ids_found = {s["session_id"] for s in sessions}
    assert {"alpha", "beta"} == ids_found
    mem.close()


# ---------------------------------------------------------------------------
# Integration: save → load → seed model → continue inference
# ---------------------------------------------------------------------------

def test_persistent_state_continues_inference():
    """h_prev loaded from memory should produce different (richer) logits than
    starting from zeros — i.e. the seeded state is actually used."""
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()

    ids_turn1 = torch.randint(0, cfg.vocab_size, (1, 6))
    ids_turn2 = torch.randint(0, cfg.vocab_size, (1, 4))

    with torch.no_grad():
        # Turn 1: run model, save state
        out1 = model(ids_turn1, use_cache=True)

        mem = SessionMemory(":memory:")
        mem.save("convo", out1["cache"], token_count=6)

        # Turn 2a: continue FROM saved state
        h_states = mem.load("convo")
        seed_cache = mem.restore_cache(h_states)
        out2_seeded = model(ids_turn2, cache=seed_cache, use_cache=True,
                            use_lnn_recurrence=True)

        # Turn 2b: start from scratch (no prior state)
        out2_fresh = model(ids_turn2, use_cache=True, use_lnn_recurrence=True)

        mem.close()

    # Logits should differ because h_prev carries different recurrent context
    assert not torch.allclose(out2_seeded["logits"], out2_fresh["logits"], atol=1e-5), \
        "Seeded and fresh logits are identical — h_prev is not being used"


# ---------------------------------------------------------------------------
# Integration: MTLNNModel convenience methods (save_state / load_state)
# ---------------------------------------------------------------------------

def test_model_save_load_state(tmp_path):
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()

    db = str(tmp_path / "test.db")
    ids = torch.randint(0, cfg.vocab_size, (1, 5))

    with torch.no_grad():
        cache = model(ids, use_cache=True)["cache"]

    model.save_state("s1", cache, token_count=5, db_path=db)
    restored = model.load_state("s1", db_path=db)

    assert restored is not None
    assert len(restored.layers) == cfg.n_layers


def test_state_only_streaming_cache_stays_constant():
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()

    cache = None
    sizes = []
    for _ in range(cfg.max_seq_len + 8):
        tok = torch.randint(0, cfg.vocab_size, (1, 1))
        logits, cache = streaming_inference(model, tok, cache, state_only=True)
        assert logits.shape == (1, 1, cfg.vocab_size)
        sizes.append(cache.tensor_bytes())

        for layer_cache in cache.layers:
            assert layer_cache[0] is None
            assert layer_cache[1] is not None
            assert layer_cache[2] is None
        assert cache.gwtb_kv is None
        assert cache.coherence_kv is None

    assert cache.token_count == cfg.max_seq_len + 8
    assert len(set(sizes[1:])) == 1


def test_state_only_session_resume_with_token_count(tmp_path):
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()

    cache = None
    for _ in range(5):
        tok = torch.randint(0, cfg.vocab_size, (1, 1))
        _, cache = streaming_inference(model, tok, cache, state_only=True)

    db = str(tmp_path / "state_only.db")
    model.save_state("stream", cache, token_count=cache.token_count, db_path=db)

    with SessionMemory(db) as mem:
        h_states = mem.load("stream")
        info = mem.session_info("stream")
        restored = mem.restore_cache(h_states)
        restored.token_count = info["token_count"]

    assert restored.token_count == 5
    logits, resumed = streaming_inference(
        model,
        torch.randint(0, cfg.vocab_size, (1, 1)),
        restored,
        state_only=True,
    )
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert resumed.token_count == 6
    for layer_cache in resumed.layers:
        assert layer_cache[0] is None
        assert layer_cache[1] is not None
        assert layer_cache[2] is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
