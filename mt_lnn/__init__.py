from .config import MTLNNConfig
from .model import MTLNNModel, MTLNNBlock, ModelCacheStruct
from .memory import SessionMemory
from .knowledge_memory import PersistentKnowledgeMemory
from .anesthesia import AnesthesiaController, anesthetize
from .phi_hat import (
    compute_phi_hat,
    compute_phi_hat_from_model,
    phi_hat_anesthesia_sweep,
    anesthesia_test_result,
    knn_entropy_chebyshev,
)
from .phi_spectral import (
    gaussian_total_correlation,
    effective_rank,
    integration_ratio,
    compute_phi_spectral_from_model,
    phi_spectral_anesthesia_sweep,
    anesthesia_test_result_spectral,
    compare_phi_metrics,
)
from .mt_lnn_layer import MTLNNLayer, ProtofilamentLTC, LateralCoupling, MAPGate, MultiScaleResonance
from .mt_attention import MicrotubuleAttention
from .global_coherence import GlobalCoherenceLayer
from .gwtb import GWTBLayer, CompetitiveGWTBLayer, BidProjector
from .embedding import MTLNNEmbedding, RotaryEmbedding
from .multimodal import (
    ModalityProjector,
    VisionPatchEmbed,
    CLIPVisionTower,
    CLIPModalityEncoder,
    fuse,
    build_modality_pad_mask,
)
from .spatial import (
    GridCellEncoding,
    MultiScaleGridCellModules,
    HeadDirectionCells,
    BoundaryDistanceCells,
    PlaceCellCode,
    SpatialCoordEncoder,
    PointCloudEncoder,
    VoxelPatchEmbed,
)
from .parallel_scan import pscan, pscan_sequential, pscan_constant_A
from .llama_adapter import (
    MTAdapterConfig,
    MTResidualAdapter,
    DecoderLayerWithMTAdapter,
    attach_mt_adapters,
)
from .streaming import streaming_inference, prefill_state_only
from .observability import JsonlMetricWriter, cache_summary, setup_logging
from .capsule import (
    CAPSULE_VERSION,
    save_capsule,
    load_capsule,
    add_open_question,
    add_evidence,
)
from .reasoning_trace import ReasoningTrace
from .deliberation import (
    DeliberationRouter,
    Route,
    RouterThresholds,
    RouteDecision,
    token_entropy,
    semantic_entropy,
    lexical_fact_gap,
)
from .thinking import (
    StepTrace,
    ThinkingTrace,
    self_consistency_vote,
    generate_with_thinking,
    render_trace_markdown,
    render_trace_html,
)
from .spatial_reasoning import SpatialThinkingResult, SpatialReasoner
from .rhythm import LAVIEstimator, GlobalRhythmController
from .causality import CausalConsistencyChecker
from .causal_steering import CausalActivationSteerer, SteerResult
from .world_model import PredictiveStateHead
# Hard-constraint physics-informed head (Hamiltonian NN + symplectic integrator).
# RESEARCH / off the served-LM path: a continuous-state (q,p) trajectory
# component, NOT a language module — a physics prior is PPL-neutral on tokens.
# Validated only on physics metrics (benchmarks/physics_rollout_eval.py).
from .hamiltonian_head import HamiltonianHead, MLPFieldHead
from .predictive_coding import (
    HierarchicalPredictiveCoder,
    PredictiveCodingResult,
)
from .imagination import ImaginedTrajectory, LatentImagination
from .active_inference import EFEPlan, ActiveInferencePlanner
from .spatial_ops import (
    pairwise_distance,
    relative_direction,
    bearing,
    in_bounding_box,
    in_ball,
    radius_graph,
    knn_graph,
    reachable_from,
    hop_distance,
    connected_components,
)
from .physics_ops import (
    PhysicsRollout,
    integrate,
    uniform_gravity,
    pairwise_gravity,
    kinetic_energy,
    momentum,
    overlapping_pairs,
    resolve_sphere_collisions,
    reflect_in_box,
    rollout,
)
from .salience_events import (
    StateChangeEvent,
    SalienceEventDetector,
    world_model_surprise,
)
from .failsafe import (
    GuardOutput,
    BlindRolloutGuard,
    BreakerResult,
    CircuitBreaker,
)
from .acoustic_ops import (
    SPEED_OF_SOUND,
    BinauralScene,
    propagation_delay,
    spherical_spreading_gain,
    interaural_time_difference,
    interaural_level_difference,
    doppler_shift,
    superpose_arrivals,
    localize_azimuth,
    binaural_scene,
)
from .ingest_ops import (
    AlignedStream,
    resample_uniform,
    nearest_sample_gap,
    coverage_mask,
    interval_jitter,
    uniform_grid,
    align_stream,
)
from .slow_layer import (
    ThreatAssessment,
    SlowThreatAssessor,
)
from .pipeline import (
    DualSpeedSentry,
    SentryTick,
    PerceptionEvent,
)
from .plasticity import HebbianRegularizer
from .astrocyte import AstrocyteGate, AstrocyteState
from .neuromodulation import NeuromodulationController, NeuromodulatorState
from .sleep_consolidation import (
    SleepWakeConsolidator,
    ReplayConsolidationResult,
    DownscaleResult,
    ConsolidationReport,
)
from .recipes import (
    apply_efficient_recipe,
    EFFORT_LEVELS,
)
from .cloud_client import (
    OracleClient,
    OracleResult,
    MockOracleClient,
    HttpOracleClient,
    build_oracle_client,
)

# Optional scientific-rigour modules (gracefully degrade if dependencies missing)
try:
    from .phi_iit import (
        compute_iit_phi,
        compute_iit_phi_from_model,
        iit_phi_anesthesia_sweep,
        PYPHI_AVAILABLE,
    )
except ImportError:
    PYPHI_AVAILABLE = False

# quantum_coupling.py removed 2026-07-13 (dead research code: a variational
# quantum-circuit drop-in for LateralCoupling that nothing ever swapped in — see
# the architecture audit). PENNYLANE_AVAILABLE kept as a stable False so any
# downstream `mt_lnn.PENNYLANE_AVAILABLE` check still resolves.
PENNYLANE_AVAILABLE = False

__all__ = [
    "MTLNNConfig",
    "MTLNNModel",
    "MTLNNBlock",
    "ModelCacheStruct",
    "SessionMemory",
    "PersistentKnowledgeMemory",
    "AnesthesiaController",
    "anesthetize",
    "compute_phi_hat",
    "compute_phi_hat_from_model",
    "phi_hat_anesthesia_sweep",
    "anesthesia_test_result",
    "knn_entropy_chebyshev",
    # Spectral / Gaussian integration metrics (Φ_G)
    "gaussian_total_correlation",
    "effective_rank",
    "integration_ratio",
    "compute_phi_spectral_from_model",
    "phi_spectral_anesthesia_sweep",
    "anesthesia_test_result_spectral",
    "compare_phi_metrics",
    "MTLNNLayer",
    "ProtofilamentLTC",
    "LateralCoupling",
    "MAPGate",
    "MultiScaleResonance",
    "MicrotubuleAttention",
    "GlobalCoherenceLayer",
    "GWTBLayer",
    "CompetitiveGWTBLayer",
    "BidProjector",
    "MTLNNEmbedding",
    "RotaryEmbedding",
    "ModalityProjector",
    "VisionPatchEmbed",
    "CLIPVisionTower",
    "CLIPModalityEncoder",
    "fuse",
    "build_modality_pad_mask",
    # Spatial computation frontends (grid-cell code, point clouds, voxels)
    "GridCellEncoding",
    # Multi-module entorhinal code: grid modules + head-direction + boundary cells
    "MultiScaleGridCellModules",
    "HeadDirectionCells",
    "BoundaryDistanceCells",
    "PlaceCellCode",
    "SpatialCoordEncoder",
    "PointCloudEncoder",
    "VoxelPatchEmbed",
    "pscan",
    "pscan_sequential",
    "pscan_constant_A",
    "MTAdapterConfig",
    "MTResidualAdapter",
    "DecoderLayerWithMTAdapter",
    "attach_mt_adapters",
    "streaming_inference",
    "prefill_state_only",
    "JsonlMetricWriter",
    "cache_summary",
    "setup_logging",
    "CAPSULE_VERSION",
    "save_capsule",
    "load_capsule",
    "add_open_question",
    "add_evidence",
    "ReasoningTrace",
    "DeliberationRouter",
    "Route",
    "RouterThresholds",
    "RouteDecision",
    "token_entropy",
    "semantic_entropy",
    "lexical_fact_gap",
    # Self-thinking serve path (live router-driven generation + trace)
    "StepTrace",
    "ThinkingTrace",
    "self_consistency_vote",
    "generate_with_thinking",
    "render_trace_markdown",
    "render_trace_html",
    # 空间思考: spatial perception + deliberation over space
    "SpatialThinkingResult",
    "SpatialReasoner",
    "LAVIEstimator",
    "GlobalRhythmController",
    "CausalConsistencyChecker",
    "CausalActivationSteerer",
    "SteerResult",
    "PredictiveStateHead",
    "HamiltonianHead",
    "MLPFieldHead",
    # Hierarchical predictive coding: top-down prediction + per-layer error spectrum
    "HierarchicalPredictiveCoder",
    "PredictiveCodingResult",
    "ImaginedTrajectory",
    "LatentImagination",
    # Active inference: Expected-Free-Energy autonomous goal selection over imagination
    "EFEPlan",
    "ActiveInferencePlanner",
    "pairwise_distance",
    "relative_direction",
    "bearing",
    "in_bounding_box",
    "in_ball",
    "radius_graph",
    "knn_graph",
    "reachable_from",
    "hop_distance",
    "connected_components",
    # Composable Newtonian dynamics operators (compute physics, don't memorise)
    "PhysicsRollout",
    "integrate",
    "uniform_gravity",
    "pairwise_gravity",
    "kinetic_energy",
    "momentum",
    "overlapping_pairs",
    "resolve_sphere_collisions",
    "reflect_in_box",
    "rollout",
    # Global-workspace ignition / state-change events (dual-speed engine trigger)
    "StateChangeEvent",
    "SalienceEventDetector",
    "world_model_surprise",
    # Input-dropout blind rollout + model-external output circuit breaker
    "GuardOutput",
    "BlindRolloutGuard",
    "BreakerResult",
    "CircuitBreaker",
    # Composable acoustic / binaural-hearing operators (ITD/ILD/Doppler/localize)
    "SPEED_OF_SOUND",
    "BinauralScene",
    "propagation_delay",
    "spherical_spreading_gain",
    "interaural_time_difference",
    "interaural_level_difference",
    "doppler_shift",
    "superpose_arrivals",
    "localize_azimuth",
    "binaural_scene",
    # Sensor ingestion: resample a jittered/timestamped stream onto the core's fixed dt
    "AlignedStream",
    "resample_uniform",
    "nearest_sample_gap",
    "coverage_mask",
    "interval_jitter",
    "uniform_grid",
    "align_stream",
    # Slow half of the dual-speed engine: multi-step threat forecast, woken on ignition
    "ThreatAssessment",
    "SlowThreatAssessor",
    # Dual-speed sentry: the loop wiring perception + prediction + salience + safety
    "DualSpeedSentry",
    "SentryTick",
    "PerceptionEvent",
    "HebbianRegularizer",
    # Slow astrocytic (glial) calcium gate over Hebbian consolidation strength
    "AstrocyteGate",
    "AstrocyteState",
    # Multi-neuromodulator global regulator (DA/ACh/5HT/NE orchestration hub)
    "NeuromodulationController",
    "NeuromodulatorState",
    # Offline sleep-wake memory consolidation (NREM replay + SHY + REM)
    "SleepWakeConsolidator",
    "ReplayConsolidationResult",
    "DownscaleResult",
    "ConsolidationReport",
    # Effort-level runtime API (GLM-5.2 style tiered compute intensity)
    "apply_efficient_recipe",
    "EFFORT_LEVELS",
    "OracleClient",
    "OracleResult",
    "MockOracleClient",
    "HttpOracleClient",
    "build_oracle_client",
    # Optional scientific-rigour modules
    "PYPHI_AVAILABLE",
    "PENNYLANE_AVAILABLE",
]

# Add optional exports only if their dependencies are present
if PYPHI_AVAILABLE:
    __all__.extend([
        "compute_iit_phi",
        "compute_iit_phi_from_model",
        "iit_phi_anesthesia_sweep",
    ])
