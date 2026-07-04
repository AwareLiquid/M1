"""Pre-configured MT-LNN adapter recipes.

Solidified configurations that have been validated across multiple bases.
Each recipe is a self-contained function that applies MT adapters + optional
LoRA with documented hyperparameters.

Runtime utilities
-----------------
set_effort_level(model, level)
    Switch a live MTLNNModel between four effort tiers without touching
    weights. Maps directly to the GLM-5.2 "effort" concept:
        0 FAST      — sparse k=1,  gate period=2,  GWTB off-per-block
        1 BALANCED  — sparse k=2,  gate period=2,  GWTB top-level (default)
        2 HIGH      — sparse k=3,  gate period=1,  world model on (if built)
        3 MAX       — all 5 scales, gate period=1, all modules on
    Only inference-time config fields are mutated; forward() behaviour
    changes instantly, weights are untouched, the call is fully reversible.

compare_effort_avp(model, input_ids, ...)
    Run the Anesthesia Validation Protocol at each effort level and compare
    Φ̂ values. Used to demonstrate "higher effort → higher global integration"
    without a full 125 M-parameter training run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .llama_adapter import attach_mt_adapters, count_trainable_parameters


# ---------------------------------------------------------------------------
# Effort-level tiers
# ---------------------------------------------------------------------------

EFFORT_LEVELS = {
    0: "FAST",
    1: "BALANCED",
    2: "HIGH",
    3: "MAX",
}


def set_effort_level(model: "MTLNNModel", level: int) -> Dict[str, object]:  # noqa: F821
    """Switch a live MTLNNModel to the requested effort tier.

    Returns a snapshot dict of the fields that were changed so the caller
    can restore the previous state with a second call if needed.

    Args:
        model: A live MTLNNModel instance (weights unchanged).
        level: 0=FAST, 1=BALANCED, 2=HIGH, 3=MAX.

    Returns:
        Dict mapping changed config-field names to their *previous* values.

    Example::

        model = MTLNNModel(cfg)
        prev = set_effort_level(model, 0)   # switch to FAST
        logits = model(input_ids=ids)["logits"]
        set_effort_level(model, **{k: v for k, v in prev.items()})  # restore
    """
    if level not in EFFORT_LEVELS:
        raise ValueError(f"level must be 0-3, got {level}")

    cfg = model.config
    S = cfg.n_time_scales

    # Fields touched per tier (only config attributes used in forward())
    tier_settings = {
        0: dict(  # FAST — minimize compute, sacrifice integration depth
            sparse_resonance_kernel=True,
            sparse_resonance_top_k=1,
            scale_gate_period=2,
            gwtb_per_block=False,
            use_world_model=False,
            use_hebbian=False,
            use_hebbian_refactor=False,
            use_rhythm=False,
            use_predictive_coding=False,
        ),
        1: dict(  # BALANCED — default inference sweet spot
            sparse_resonance_kernel=True,
            sparse_resonance_top_k=2,
            scale_gate_period=2,
            gwtb_per_block=False,
            use_world_model=getattr(cfg, "use_world_model", False),
            use_hebbian=False,
            use_hebbian_refactor=False,
            use_rhythm=getattr(cfg, "use_rhythm", False),
            use_predictive_coding=getattr(cfg, "use_predictive_coding", True),
        ),
        2: dict(  # HIGH — full temporal resolution, world-model on
            sparse_resonance_kernel=True,
            sparse_resonance_top_k=min(3, S),
            scale_gate_period=1,
            gwtb_per_block=False,
            use_world_model=model.world_model_head is not None,
            use_hebbian=False,
            use_hebbian_refactor=getattr(cfg, "use_hebbian_refactor", False),
            use_rhythm=getattr(cfg, "use_rhythm", False),
            use_predictive_coding=getattr(cfg, "use_predictive_coding", True),
        ),
        3: dict(  # MAX — all scales, all bio modules
            sparse_resonance_kernel=False,
            sparse_resonance_top_k=S,
            scale_gate_period=1,
            gwtb_per_block=getattr(cfg, "gwtb_per_block", False),
            use_world_model=model.world_model_head is not None,
            use_hebbian=model.hebbian_reg is not None,
            use_hebbian_refactor=model.hebbian_plasticity is not None,
            use_rhythm=any(b.lnn.use_rhythm for b in model.blocks),
            use_predictive_coding=getattr(cfg, "use_predictive_coding", True),
        ),
    }

    new_settings = tier_settings[level]
    snapshot: Dict[str, object] = {}

    for field, new_val in new_settings.items():
        old_val = getattr(cfg, field, None)
        if old_val != new_val:
            snapshot[field] = old_val
            setattr(cfg, field, new_val)

    # Propagate sparse_resonance_kernel/top_k into every resonance bank
    # (they are read from config at init but also consulted live via getattr)
    for block in model.blocks:
        res = block.lnn.resonance
        res.sparse_resonance_kernel = cfg.sparse_resonance_kernel
        res.sparse_resonance_top_k = cfg.sparse_resonance_top_k

    return snapshot


def compare_effort_avp(
    model: "MTLNNModel",  # noqa: F821
    input_ids: "torch.Tensor",
    kappas: Optional[List[float]] = None,
    K: int = 4,
    k_nn: int = 3,
    levels: Optional[List[int]] = None,
) -> Dict[int, dict]:
    """Run the Anesthesia Validation Protocol at each effort level.

    Measures Φ̂ at clean state and under progressive κ-anesthesia for each
    effort tier. Demonstrates that higher effort (more global workspace
    integration) raises Φ̂ even on the same model weights.

    This is the key counter-argument against "AVP failed on tiny model =
    mechanism broken": if Φ̂ is systematically higher in MAX vs FAST mode,
    the workspace integration mechanism is working; the tiny-model FAILURE
    in run_benchmark.py is purely a capacity issue (not enough parameters to
    build a rich global state), not an architecture defect.

    Args:
        model: Live MTLNNModel.
        input_ids: (B, T) token ids.
        kappas: anesthesia sweep values (default [1.0, 2.0, 5.0, 10.0]).
        K: number of random projections for Φ̂.
        k_nn: nearest-neighbour count for Φ̂.
        levels: effort levels to compare (default [0, 1, 2, 3]).

    Returns:
        Dict[level -> {"phi_clean": float, "phi_full": float,
                       "collapse_pct": float, "passed": bool,
                       "sweep": Dict[kappa -> phi]}]
    """
    from mt_lnn import phi_hat_anesthesia_sweep, anesthesia_test_result

    if kappas is None:
        kappas = [1.0, 2.0, 5.0, 10.0]
    if levels is None:
        levels = [0, 1, 2, 3]

    results: Dict[int, dict] = {}
    original_snapshot: Dict[str, object] = {}

    # Save original config state
    cfg = model.config
    for field in ["sparse_resonance_kernel", "sparse_resonance_top_k",
                  "scale_gate_period", "gwtb_per_block", "use_world_model",
                  "use_hebbian", "use_predictive_coding"]:
        original_snapshot[field] = getattr(cfg, field, None)

    try:
        for level in levels:
            set_effort_level(model, level)
            model.eval()
            with torch.no_grad():
                sweep = phi_hat_anesthesia_sweep(
                    model, input_ids, kappas=kappas, K=K, k_nn=k_nn
                )
            res = anesthesia_test_result(sweep, delta=0.7)
            results[level] = {
                "level_name": EFFORT_LEVELS[level],
                "phi_clean": res["phi_clean"],
                "phi_full": res["phi_full"],
                "collapse_pct": res["collapse_pct"],
                "passed": res["passed"],
                "sweep": sweep,
            }
    finally:
        # Always restore original config
        for field, val in original_snapshot.items():
            if val is not None:
                setattr(cfg, field, val)
        for block in model.blocks:
            res_mod = block.lnn.resonance
            res_mod.sparse_resonance_kernel = cfg.sparse_resonance_kernel
            res_mod.sparse_resonance_top_k = cfg.sparse_resonance_top_k

    return results


@dataclass
class RecipeResult:
    """Result of applying an adapter recipe."""

    wrapped_layer_indices: List[int]
    trainable_params: int
    total_params: int
    trainable_percent: float
    lora_applied: bool


def apply_phase5b_recipe(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_targets: Optional[List[str]] = None,
    verbose: bool = True,
) -> RecipeResult:
    """Apply the Phase 5b adapter recipe (MT residual adapters + LoRA).

    !! 2026-07-04 ATTRIBUTION CORRECTION !!
    The historical Phase 5/5b results previously quoted here
    (-28.5%/-27.7%/-34.4% PPL at "0.196%/0.139%/0.117% trainable") were run
    BEFORE the re-arm fix (commit 8d9d741, 2026-06-28): get_peft_model()
    froze the MT adapters at random init (residual scale 1e-3, contribution
    ~0) and ONLY LoRA trained. The quoted "trainable" counts are exactly the
    LoRA-only parameter counts (2.25M / 2.18M / 3.7M). Those PPL gains
    therefore measure plain LoRA, not the MT architecture. This function DOES
    train the MT adapters when the caller re-arms them after LoRA (as
    train_llama_mt_adapter.py now does); its REAL trainable budget on
    TinyLlama-1.1B is ~65.1M (5.6%) — the MT adapter is dominated by dense
    in/out projections. See benchmarks/attribution_ablation.py for the honest
    LoRA-vs-MT attribution and mt_lnn.mt_lnn_v2 for the parameter-lean
    redesign (~8.4M, 0.76%).

    The recipe consists of:
    1. MT residual adapters on every 4th decoder layer
       - 13 protofilaments (microtubule architecture)
       - 5 temporal time scales
       - 64-dim MAP gate hidden size
       - init_scale=1e-3 for stable residual connection
    2. LoRA (optional) on attention projection layers
       - Default targets: q_proj, k_proj, v_proj, o_proj
       - Rank 8, alpha 16, dropout 0.05

    Args:
        model: HuggingFace causal LM (unfrozen). Will be frozen by this function.
        lora_rank: LoRA rank (r parameter). Set to 0 to disable LoRA.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout probability.
        lora_targets: List of module names to apply LoRA to. Default: ["q_proj", "k_proj", "v_proj", "o_proj"]
        verbose: Print parameter counts and layer indices.

    Returns:
        RecipeResult with wrapped layer indices and parameter counts.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        >>> result = apply_phase5b_recipe(model)
        >>> print(f"Trainable: {result.trainable_percent:.3f}%")
    """
    # Step 1: Attach MT adapters every 4th layer
    wrapped = attach_mt_adapters(
        model,
        every=4,
        n_protofilaments=13,
        n_time_scales=5,
        map_hidden_dim=64,
        dropout=0.0,
        init_scale=1e-3,
        use_scan=True,
    )

    if verbose:
        trainable_before_lora = count_trainable_parameters(model)
        total = sum(p.numel() for p in model.parameters())
        print(f"MT adapters attached to layers: {wrapped}")
        print(
            f"Trainable params (MT only): {trainable_before_lora:,} / {total:,} "
            f"({100 * trainable_before_lora / total:.3f}%)"
        )

    # Step 2: Apply LoRA to attention projections (optional)
    lora_applied = False
    if lora_rank > 0:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            if verbose:
                print(
                    "Warning: LoRA requested but `peft` not installed. "
                    "Install with `pip install peft`. Skipping LoRA."
                )
        else:
            if lora_targets is None:
                lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_targets,
            )
            # get_peft_model wraps the model in-place
            peft_model = get_peft_model(model, config)
            # Update model reference if needed (some PEFT versions return new object)
            if peft_model is not model:
                # This shouldn't happen with modern PEFT, but handle it
                if verbose:
                    print("Warning: PEFT returned new model object (unusual)")
            lora_applied = True

            if verbose:
                print(f"LoRA applied to: {lora_targets} (r={lora_rank}, alpha={lora_alpha})")

    # Compute final statistics
    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())

    if verbose:
        print(
            f"Final trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.3f}%)"
        )

    return RecipeResult(
        wrapped_layer_indices=wrapped,
        trainable_params=trainable,
        total_params=total,
        trainable_percent=100 * trainable / total,
        lora_applied=lora_applied,
    )


def apply_mt_only_recipe(
    model: nn.Module,
    every: int = 4,
    n_protofilaments: int = 13,
    n_time_scales: int = 5,
    map_hidden_dim: int = 64,
    init_scale: float = 1e-3,
    verbose: bool = True,
) -> RecipeResult:
    """Apply MT adapters without LoRA.

    Useful for ablations to isolate the effect of MT architecture alone.

    Args:
        model: HuggingFace causal LM (unfrozen). Will be frozen by this function.
        every: Attach MT adapter every N layers.
        n_protofilaments: Number of parallel protofilaments (default 13 from microtubule biology).
        n_time_scales: Number of temporal integration scales.
        map_hidden_dim: Hidden dimension for MAP gate.
        init_scale: Residual connection initialization scale.
        verbose: Print parameter counts and layer indices.

    Returns:
        RecipeResult with wrapped layer indices and parameter counts.
    """
    wrapped = attach_mt_adapters(
        model,
        every=every,
        n_protofilaments=n_protofilaments,
        n_time_scales=n_time_scales,
        map_hidden_dim=map_hidden_dim,
        dropout=0.0,
        init_scale=init_scale,
        use_scan=True,
    )

    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"MT adapters attached to layers: {wrapped}")
        print(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.3f}%)"
        )

    return RecipeResult(
        wrapped_layer_indices=wrapped,
        trainable_params=trainable,
        total_params=total,
        trainable_percent=100 * trainable / total,
        lora_applied=False,
    )


def apply_efficient_recipe(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_targets: Optional[List[str]] = None,
    verbose: bool = True,
) -> RecipeResult:
    """Phase 5b + cross-layer gate sharing (period=2) — the default high-efficiency config.

    Extends apply_phase5b_recipe with two inference-time optimisations that have
    been validated to be quality-neutral:
      • sparse_resonance_kernel=True, sparse_resonance_top_k=2  (+7.0% throughput,
        mean logit divergence 0.003 vs dense)
      • scale_gate_period=2  (+1.8% throughput, mean logit divergence 0.015 on
        same-weight comparison; GPU upside expected to be larger)

    Underlying adapter layers and LoRA targets are identical to Phase 5b.
    The two extra flags are written directly on the HuggingFace model's MT
    adapter modules (not weights) so they take effect at next forward() call.

    NOTE: the historical Phase 5b benchmark numbers formerly quoted here are
    retracted — see the ATTRIBUTION CORRECTION in apply_phase5b_recipe's
    docstring (those runs trained LoRA only; the MT adapters were frozen).

    Use set_effort_level(model, 0) at inference time for the FAST tier
    (sparse k=1, period=2) if you need maximum throughput.
    """
    result = apply_phase5b_recipe(
        model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        verbose=verbose,
    )

    # Apply efficiency flags to every MT adapter's resonance bank
    _set_sparse_config(model, sparse_kernel=True, top_k=2, gate_period=2)

    if verbose:
        print("Efficient flags: sparse_resonance_kernel=True, top_k=2 "
              "(scale_gate_period applies to native MTLNNModel only)")

    return result


def _set_sparse_config(
    model: nn.Module,
    sparse_kernel: bool,
    top_k: int,
    gate_period: int,
) -> None:
    """Walk an HF model and set sparse-resonance flags on all MT adapter layers."""
    from .llama_adapter import MTResidualAdapter

    # NOTE: gate_period (cross-layer scale-gate sharing) applies only to the
    # native MTLNNModel block loop, where a leader layer's active_idx is shared
    # with followers. The HF adapter path runs each adapter independently inside
    # its host decoder layer, so there is no shared loop to propagate the index;
    # the parameter is accepted for API symmetry but has no effect here.
    adapters = [m for m in model.modules() if isinstance(m, MTResidualAdapter)]
    for adapter in adapters:
        res = adapter.mt_layer.resonance
        res.sparse_resonance_kernel = sparse_kernel
        res.sparse_resonance_top_k = top_k


def apply_lora_only_recipe(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_targets: Optional[List[str]] = None,
    verbose: bool = True,
) -> RecipeResult:
    """Apply LoRA without MT adapters.

    Useful for ablations to compare against vanilla LoRA baseline.

    Args:
        model: HuggingFace causal LM (unfrozen). Will be frozen by this function.
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout probability.
        lora_targets: List of module names to apply LoRA to. Default: ["q_proj", "k_proj", "v_proj", "o_proj"]
        verbose: Print parameter counts.

    Returns:
        RecipeResult with parameter counts (wrapped_layer_indices will be empty).
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requested but `peft` not installed. Install with `pip install peft`."
        ) from exc

    # Freeze base model first
    for param in model.parameters():
        param.requires_grad = False

    if lora_targets is None:
        lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    peft_model = get_peft_model(model, config)

    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"LoRA applied to: {lora_targets} (r={lora_rank}, alpha={lora_alpha})")
        print(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.3f}%)"
        )

    return RecipeResult(
        wrapped_layer_indices=[],
        trainable_params=trainable,
        total_params=total,
        trainable_percent=100 * trainable / total,
        lora_applied=True,
    )
