"""Pre-configured MT-LNN adapter recipes.

Solidified configurations that have been validated across multiple bases.
Each recipe is a self-contained function that applies MT adapters + optional
LoRA with documented hyperparameters.

(The runtime effort-tier helpers `set_effort_level` / `compare_effort_avp`
were removed 2026-07-13 as dead code — zero callers anywhere in serve/
benchmarks/scripts/train/tests, and `compare_effort_avp` depended on the
now-inert Φ̂/anesthesia path. `EFFORT_LEVELS` and the `apply_*_recipe`
builders below remain.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
