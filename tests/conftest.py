"""
tests/conftest.py -- shared, deterministic test fixtures.

A single autouse fixture pins every RNG *before each test* so the suite is
reproducible no matter the collection order, and a stray test that forgets to
seed can no longer go flaky on a neighbour's leftover RNG state.

Tests that need a *specific* seed (e.g. to pin an exact ignition tick) still
call ``torch.manual_seed(...)`` inside the body; that simply overrides this
baseline, so existing per-test seeding is unaffected.
"""
import random

import pytest
import torch

_BASELINE_SEED = 1234


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Reset Python/torch/numpy RNGs to a fixed baseline before every test."""
    random.seed(_BASELINE_SEED)
    torch.manual_seed(_BASELINE_SEED)
    try:
        import numpy as np

        np.random.seed(_BASELINE_SEED)
    except Exception:
        pass
    yield
