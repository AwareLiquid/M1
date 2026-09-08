from __future__ import annotations

import warnings

import pytest
import torch

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.profiles import (
    M2Experiment,
    m1_stable_config,
    m2_research_config,
)


def test_m1_stable_profile_keeps_only_the_validated_language_trunk() -> None:
    config = m1_stable_config()

    assert config.ffn_swiglu is True
    assert config.qk_norm is True
    assert config.scaled_residual_init is True
    assert config.use_gwtb is False
    assert config.use_global_coherence is False
    assert config.core_iterations == 1
    assert config.stack_iterations == 1
    assert config.workspace_iterations == 1
    assert config.selective_decay is False
    assert config.fast_weight_core is False
    assert config.use_top_down is False
    assert config.use_world_model is False
    assert config.use_hamiltonian_world_model is False
    assert config.use_rhythm is False
    assert config.global_rhythm is False


@pytest.mark.parametrize(
    ("experiment", "field", "expected"),
    [
        (M2Experiment.SELECTIVE_STATE, "selective_decay", True),
        (M2Experiment.FAST_WEIGHT_MEMORY, "fast_weight_core", True),
        (M2Experiment.LATENT_CORE, "core_iterations", 4),
        (M2Experiment.LATENT_STACK, "stack_iterations", 4),
        (M2Experiment.WORKSPACE, "workspace_iterations", 4),
        (
            M2Experiment.COMPETITIVE_WORKSPACE,
            "use_competitive_gwtb",
            True,
        ),
        (M2Experiment.TOP_DOWN, "use_top_down", True),
        (
            M2Experiment.PREDICTIVE_CODING,
            "use_predictive_coding",
            True,
        ),
        (M2Experiment.WORLD_MODEL, "use_world_model", True),
        (
            M2Experiment.HAMILTONIAN_WORLD_MODEL,
            "use_hamiltonian_world_model",
            True,
        ),
        (M2Experiment.GLOBAL_RHYTHM, "global_rhythm", True),
        (M2Experiment.GLOBAL_COHERENCE, "use_global_coherence", True),
        (M2Experiment.HEBBIAN, "use_hebbian", True),
    ],
)
def test_m2_profile_enables_only_requested_experiment(
    experiment: M2Experiment,
    field: str,
    expected: bool | int,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        config = m2_research_config((experiment,))

    assert getattr(config, field) == expected


def test_m2_workspace_enables_its_required_gwtb_parent() -> None:
    config = m2_research_config((M2Experiment.WORKSPACE,))

    assert config.use_gwtb is True
    assert config.workspace_iterations == 4


def test_m2_namespace_owns_the_research_profile_api() -> None:
    from mt_lnn.m2 import M2Experiment as NamespacedExperiment
    from mt_lnn.m2 import m2_research_config as namespaced_config

    assert NamespacedExperiment is M2Experiment
    assert namespaced_config is m2_research_config


def test_m2_research_without_experiments_matches_m1_stable() -> None:
    assert m2_research_config() == m1_stable_config()


def test_global_coherence_can_be_removed_from_the_model_graph() -> None:
    config = MTLNNConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=104,
        n_layers=1,
        n_heads=13,
        n_kv_heads=1,
        d_head=8,
        n_time_scales=2,
        gwtb_n_heads=1,
        use_gwtb=False,
        use_global_coherence=False,
    )
    model = MTLNNModel(config).eval()

    with torch.no_grad():
        result = model(torch.randint(0, config.vocab_size, (1, 4)))

    assert model.coherence is None
    assert result["logits"].shape == (1, 4, config.vocab_size)
