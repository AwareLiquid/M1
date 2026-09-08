from __future__ import annotations

from dataclasses import replace
from enum import StrEnum
from typing import Iterable, assert_never

from .config import MTLNNConfig


class M2Experiment(StrEnum):
    SELECTIVE_STATE = "selective_state"
    FAST_WEIGHT_MEMORY = "fast_weight_memory"
    LATENT_CORE = "latent_core"
    LATENT_STACK = "latent_stack"
    WORKSPACE = "workspace"
    COMPETITIVE_WORKSPACE = "competitive_workspace"
    TOP_DOWN = "top_down"
    PREDICTIVE_CODING = "predictive_coding"
    WORLD_MODEL = "world_model"
    HAMILTONIAN_WORLD_MODEL = "hamiltonian_world_model"
    GLOBAL_RHYTHM = "global_rhythm"
    GLOBAL_COHERENCE = "global_coherence"
    HEBBIAN = "hebbian"


def m1_stable_config() -> MTLNNConfig:
    return MTLNNConfig(
        ffn_swiglu=True,
        qk_norm=True,
        scaled_residual_init=True,
        use_gwtb=False,
        use_global_coherence=False,
    )


def m2_research_config(
    experiments: Iterable[M2Experiment] = (),
) -> MTLNNConfig:
    config = m1_stable_config()
    for experiment in experiments:
        match experiment:
            case M2Experiment.SELECTIVE_STATE:
                config = replace(config, selective_decay=True)
            case M2Experiment.FAST_WEIGHT_MEMORY:
                config = replace(config, fast_weight_core=True)
            case M2Experiment.LATENT_CORE:
                config = replace(config, core_iterations=4)
            case M2Experiment.LATENT_STACK:
                config = replace(config, stack_iterations=4)
            case M2Experiment.WORKSPACE:
                config = replace(
                    config,
                    use_gwtb=True,
                    workspace_iterations=4,
                )
            case M2Experiment.COMPETITIVE_WORKSPACE:
                config = replace(
                    config,
                    use_gwtb=True,
                    use_competitive_gwtb=True,
                )
            case M2Experiment.TOP_DOWN:
                config = replace(config, use_top_down=True)
            case M2Experiment.PREDICTIVE_CODING:
                config = replace(config, use_predictive_coding=True)
            case M2Experiment.WORLD_MODEL:
                config = replace(config, use_world_model=True)
            case M2Experiment.HAMILTONIAN_WORLD_MODEL:
                config = replace(config, use_hamiltonian_world_model=True)
            case M2Experiment.GLOBAL_RHYTHM:
                config = replace(config, use_rhythm=True, global_rhythm=True)
            case M2Experiment.GLOBAL_COHERENCE:
                config = replace(config, use_global_coherence=True)
            case M2Experiment.HEBBIAN:
                config = replace(config, use_hebbian=True)
            case unreachable:
                assert_never(unreachable)
    return config
