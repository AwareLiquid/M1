from .config import MTLNNConfig
from .model import MTLNNModel, MTLNNBlock, ModelCacheStruct
from .memory import SessionMemory
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
from .imagination import ImaginedTrajectory, LatentImagination
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
from .plasticity import HebbianRegularizer
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

try:
    from .quantum_coupling import QuantumLateralCoupling, PENNYLANE_AVAILABLE
except ImportError:
    PENNYLANE_AVAILABLE = False

__all__ = [
    "MTLNNConfig",
    "MTLNNModel",
    "MTLNNBlock",
    "ModelCacheStruct",
    "SessionMemory",
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
    "ImaginedTrajectory",
    "LatentImagination",
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
    "HebbianRegularizer",
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
if PENNYLANE_AVAILABLE:
    __all__.append("QuantumLateralCoupling")
