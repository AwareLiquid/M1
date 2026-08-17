"""
tests/test_sleep_consolidation.py — behavioural contract for the offline
sleep-wake memory consolidation cycle (mt_lnn/sleep_consolidation.py, 2026-06-15).

SleepWakeConsolidator orchestrates three brain-grounded offline maintenance
stages over machinery the codebase already has:
  • NREM replay → consolidation: sample episodes from a ReservoirBuffer and
    write the salient ones, by content, into a PersistentKnowledgeMemory
    (fast hippocampal → slow neocortical transfer);
  • synaptic downscaling (SHY): multiplicatively renormalise weights downward,
    preserving the relative pattern (direction) while shrinking the norm;
  • REM generative recombination: convex blends of replayed latents.

We pin:
  • replay consolidates the right count and the content is recallable afterwards;
  • salience ranking consolidates the most-salient episodes first;
  • downscaling shrinks the norm by exactly the factor and preserves direction
    (cosine ~1), with protect() sparing named params; factor=1.0 is a no-op;
  • REM recombination produces the requested count, on the segment between pairs;
  • the full consolidate() cycle runs each stage only when its inputs are given;
  • determinism by seed; zero model coupling (not an nn.Module).

Run:  python -m pytest tests/test_sleep_consolidation.py -v
"""
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from mt_lnn import (                                                  # noqa: E402
    SleepWakeConsolidator,
    PersistentKnowledgeMemory,
)
from mt_lnn.replay import ReservoirBuffer                            # noqa: E402


def _basis(dim, i):
    v = torch.zeros(dim)
    v[i] = 1.0
    return v


def _fill_buffer(cap=8, key_dim=4, seed=0):
    """A reservoir of (key, content, salience) episodes with distinct keys."""
    buf = ReservoirBuffer(cap, seed=seed)
    for i in range(cap):
        buf.add(
            key=_basis(key_dim, i % key_dim),
            content=torch.tensor(float(i)),
            salience=torch.tensor(float(i)),     # episode i is more salient
        )
    return buf


# --------------------------------------------------------------------------- #
# NREM replay → consolidation                                                  #
# --------------------------------------------------------------------------- #
def test_nrem_replay_consolidates_into_knowledge_store():
    buf = _fill_buffer(cap=4, key_dim=4)
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:")
    sc = SleepWakeConsolidator(seed=1)

    res = sc.nrem_replay(buf, kb, key_field="key", content_field="content")
    assert res.replayed == 4
    assert res.consolidated == 4
    assert len(kb) == 4
    # The content stored under key e_2 is recallable.
    recalled = kb.recall(_basis(4, 2))
    assert recalled is not None


def test_nrem_replay_consolidates_most_salient_first():
    # consolidate only half; the higher-salience episodes must be the ones kept.
    buf = _fill_buffer(cap=6, key_dim=6)
    kb = PersistentKnowledgeMemory(key_dim=6, db_path=":memory:")
    sc = SleepWakeConsolidator(seed=2)

    res = sc.nrem_replay(
        buf, kb,
        key_field="key", content_field="content", salience_field="salience",
        consolidate_fraction=0.5,
    )
    assert res.replayed == 6
    assert res.consolidated == 3
    # Episodes 3,4,5 had the top salience → their contents (3.0,4.0,5.0) are stored.
    stored = set()
    for i in range(6):
        hits = kb.query(_basis(6, i), top_k=6, touch=False)
        for content, score, _ in hits:
            if score > 0.999:                    # exact key match
                stored.add(float(content))
    assert stored == {3.0, 4.0, 5.0}


def test_nrem_replay_empty_buffer_is_noop():
    buf = ReservoirBuffer(4, seed=0)
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:")
    sc = SleepWakeConsolidator()
    res = sc.nrem_replay(buf, kb, key_field="key", content_field="content")
    assert (res.replayed, res.consolidated) == (0, 0)
    assert len(kb) == 0


def test_nrem_replay_missing_field_raises():
    buf = _fill_buffer(cap=2, key_dim=4)
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:")
    sc = SleepWakeConsolidator()
    with pytest.raises(KeyError):
        sc.nrem_replay(buf, kb, key_field="nope", content_field="content")


# --------------------------------------------------------------------------- #
# synaptic downscaling (SHY)                                                   #
# --------------------------------------------------------------------------- #
def test_downscale_shrinks_norm_and_preserves_direction():
    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    before = lin.weight.detach().clone().flatten()

    sc = SleepWakeConsolidator(downscale_factor=0.5)
    res = sc.synaptic_downscale(lin)
    after = lin.weight.detach().clone().flatten()

    assert res.post_norm == pytest.approx(0.5 * res.pre_norm, rel=1e-5)
    # Pure multiplicative rescale → direction preserved (cosine ~ 1).
    cos = torch.nn.functional.cosine_similarity(before, after, dim=0)
    assert cos.item() == pytest.approx(1.0, abs=1e-6)
    # And the values are exactly halved.
    assert torch.allclose(after, 0.5 * before, atol=1e-6)


def test_downscale_factor_one_is_noop():
    torch.manual_seed(1)
    lin = nn.Linear(4, 4)
    before = {k: v.detach().clone() for k, v in lin.named_parameters()}
    sc = SleepWakeConsolidator()
    sc.synaptic_downscale(lin, factor=1.0)
    for k, v in lin.named_parameters():
        assert torch.equal(v.detach(), before[k])


def test_downscale_protect_spares_named_params():
    torch.manual_seed(2)
    lin = nn.Linear(4, 4)
    bias_before = lin.bias.detach().clone()
    weight_before = lin.weight.detach().clone()
    sc = SleepWakeConsolidator(downscale_factor=0.5)
    res = sc.synaptic_downscale(lin, protect=lambda n: n.endswith("bias"))
    # bias untouched, weight halved; only one tensor rescaled.
    assert torch.equal(lin.bias.detach(), bias_before)
    assert torch.allclose(lin.weight.detach(), 0.5 * weight_before, atol=1e-6)
    assert res.n_tensors == 1


def test_downscale_accepts_tensor_list():
    a = torch.ones(3)
    b = torch.full((2,), 2.0)
    sc = SleepWakeConsolidator()
    res = sc.synaptic_downscale([a, b], factor=0.25)
    assert torch.allclose(a, torch.full((3,), 0.25))
    assert torch.allclose(b, torch.full((2,), 0.5))
    assert res.n_tensors == 2


# --------------------------------------------------------------------------- #
# REM generative recombination                                                 #
# --------------------------------------------------------------------------- #
def test_rem_recombine_count_and_on_segment():
    sc = SleepWakeConsolidator(seed=3)
    latents = torch.tensor([[0.0, 0.0], [1.0, 1.0]])     # two corners
    dreamed = sc.rem_recombine(latents, n_samples=20)
    assert dreamed.shape == (20, 2)
    # Every blend of these two points lies on the diagonal x==y, in [0,1].
    assert torch.allclose(dreamed[:, 0], dreamed[:, 1], atol=1e-6)
    assert (dreamed >= -1e-6).all() and (dreamed <= 1.0 + 1e-6).all()


def test_rem_recombine_fixed_alpha_midpoint():
    sc = SleepWakeConsolidator(seed=4)
    latents = torch.tensor([[0.0], [4.0]])
    dreamed = sc.rem_recombine(latents, n_samples=10, alpha=0.5)
    # midpoint of {0,4} pairs ∈ {0,2,4}; with alpha 0.5 each sample is mean of a pair.
    for v in dreamed.flatten().tolist():
        assert v in (0.0, 2.0, 4.0)


def test_rem_recombine_deterministic_by_seed():
    # Same seed → same pair indices & blend weights → identical recombination
    # applied to the same latents.
    torch.manual_seed(0)
    L = torch.randn(5, 3)
    da = SleepWakeConsolidator(seed=7).rem_recombine(L, n_samples=8)
    db = SleepWakeConsolidator(seed=7).rem_recombine(L, n_samples=8)
    assert torch.equal(da, db)


# --------------------------------------------------------------------------- #
# full cycle + housekeeping                                                    #
# --------------------------------------------------------------------------- #
def test_full_consolidate_cycle_runs_selected_stages():
    buf = _fill_buffer(cap=4, key_dim=4)
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:")
    lin = nn.Linear(4, 4)
    sc = SleepWakeConsolidator(downscale_factor=0.5, seed=5)

    report = sc.consolidate(
        buffer=buf, knowledge_memory=kb,
        replay_kwargs=dict(key_field="key", content_field="content"),
        module=lin,
        rem_latents=torch.randn(6, 4), rem_samples=5,
    )
    assert report.replay.consolidated == 4
    assert report.downscale.factor == 0.5
    assert report.downscale.post_norm == pytest.approx(0.5 * report.downscale.pre_norm, rel=1e-5)
    assert report.recombined == 5


def test_consolidate_skips_absent_stages():
    sc = SleepWakeConsolidator()
    report = sc.consolidate()      # nothing provided
    assert report.replay is None
    assert report.downscale is None
    assert report.recombined == 0


def test_consolidator_is_not_nn_module_and_validates():
    sc = SleepWakeConsolidator()
    assert not isinstance(sc, nn.Module)
    with pytest.raises(ValueError):
        SleepWakeConsolidator(downscale_factor=0.0)
    with pytest.raises(ValueError):
        SleepWakeConsolidator(downscale_factor=1.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
