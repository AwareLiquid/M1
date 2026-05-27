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
from .gwtb import GWTBLayer
from .embedding import MTLNNEmbedding, RotaryEmbedding
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
    "MTLNNEmbedding",
    "RotaryEmbedding",
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
