import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module, config) -> None:
    """GPT-2-style init with MT-specific overrides."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    # MT-specific overrides applied by name after full model init
    # (called from MTLNNModel.__init__ via named_modules loop)


def init_mt_params(model: nn.Module, config) -> None:
    """
    Apply MT-specific parameter initialisations after standard init.

    Strategy: keep the model behaving like a standard Transformer at step 0,
    then let microtubule dynamics emerge during training.
      - τ initialised to `tau_init` (≈1.0) so closed-form LTC is well-conditioned
      - polarity ≈ 0  → attention is approximately symmetric/causal at start
      - GTP γ small  → attention spans far positions initially
      - lateral W_lat = eye → protofilaments independent at start
      - rmc_gate = 0 → RMC contribution starts at zero, ramps up during training
      - coherence_scale = 0.05 (small) → global coherence layer near-identity
      - out_proj of MTLNNLayer scaled small (std=0.01) → block residual is small
    """
    for name, param in model.named_parameters():
        # log_tau lives in VectorizedMultiScaleResonance and is initialised
        # PER-SCALE in the constructor (using config.resonance_freqs). We
        # deliberately do NOT override it here so multi-scale variety survives.
        if name.endswith("polarity_direction"):
            nn.init.uniform_(param, -0.05, 0.05)
        elif name.endswith("W_lat"):
            nn.init.eye_(param)
            param.data += torch.randn_like(param) * 0.005
        elif name.endswith("rmc_gate"):
            nn.init.zeros_(param)               # sigmoid(0)=0.5 — but we re-zero below
            # Actually want sigmoid(rmc_gate) ≈ 0 at start; use large negative init
            nn.init.constant_(param, -3.0)      # sigmoid(-3) ≈ 0.047
        elif name.endswith("coherence_scale"):
            nn.init.constant_(param, 0.05)
        elif name.endswith("collapse_threshold"):
            nn.init.constant_(param, 0.5)
        elif name.endswith("blend_weights"):
            nn.init.zeros_(param)               # softmax(zeros) = uniform
        elif name.endswith("kappa_gate.weight"):
            nn.init.normal_(param, mean=0.0, std=0.02)
        elif name.endswith("kappa_gate.bias"):
            nn.init.constant_(param, getattr(config, "scale_gate_init_bias", 2.0))
        elif name.endswith("gtp_gamma"):
            # In MicrotubuleAttention this is in raw (softplus⁻¹) space → keep ctor init.
            # In MTLNNLayer this is in real space → re-set to a small positive value.
            if param.dim() == 0:
                nn.init.constant_(param, config.gamma_init)
        # MTLNNLayer.out_proj small init for residual stability
        # (matches names like "blocks.0.lnn.out_proj.weight")
        elif "lnn.out_proj.weight" in name:
            nn.init.normal_(param, mean=0.0, std=0.01)
        elif "lnn.out_proj.bias" in name:
            nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_module: Dict[str, int] = {}
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        by_module[name] = n
    return {"total": total, "trainable": trainable, "by_module": by_module}


# ---------------------------------------------------------------------------
# Causal mask helper
# ---------------------------------------------------------------------------

def get_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """(T, T) lower-triangular boolean mask; True = keep."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


# ---------------------------------------------------------------------------
# Cosine helpers (shared convention: map cos ∈ [-1, 1] → unit interval [0, 1])
# ---------------------------------------------------------------------------

def unit_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Cosine similarity of two 1-D vectors, affinely mapped to [0, 1].

    0.0 = anti-aligned, 0.5 = orthogonal, 1.0 = aligned. Returns a plain float
    so it is JSON-serialisable and free of autograd state. Used by the Phase B
    causal-consistency checker; the world model uses the complementary
    ``1 - cos`` (surprise) form inline because it reduces over a batch.
    """
    a_n = F.normalize(a.unsqueeze(0), dim=-1, eps=eps).squeeze(0)
    b_n = F.normalize(b.unsqueeze(0), dim=-1, eps=eps).squeeze(0)
    raw = torch.dot(a_n, b_n).item()                  # ∈ [-1, 1]
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


# ---------------------------------------------------------------------------
# Learning-rate scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self._step = 0
        # Store base LRs per param group
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self._step += 1
        s = self._step
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            if s < self.warmup_steps:
                lr = base_lr * s / max(1, self.warmup_steps)
            else:
                progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            pg["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str,
    config=None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if config is not None:
        import dataclasses
        payload["config"] = dataclasses.asdict(config)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"[checkpoint] missing keys initialised from model defaults: {missing}")
    if unexpected:
        print(f"[checkpoint] unexpected keys ignored: {unexpected}")
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ---------------------------------------------------------------------------
# AdamW param groups for MT-LNN (separate LRs for dynamical constants)
# ---------------------------------------------------------------------------

def make_param_groups(model: nn.Module, base_lr: float) -> list:
    """
    Separate param groups so ODE constants (τ, γ) train at 0.33× base_lr
    and polarity can move faster at 1.67× base_lr.
    """
    ode_names = {"log_tau", "gtp_gamma"}
    polarity_names = {"polarity_direction"}
    lateral_names = {"W_lat"}

    ode_params, polarity_params, lateral_params = [], [], []
    main_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        base = name.split(".")[-1]
        if base in ode_names:
            ode_params.append(param)
        elif base in polarity_names:
            polarity_params.append(param)
        elif base in lateral_names:
            lateral_params.append(param)
        elif param.ndim < 2 or "embed" in name.lower():
            # Standard AdamW hygiene: never decay 1-d params (LayerNorm
            # weight/bias, biases, scalar gates) or embeddings — decaying a
            # LayerNorm scale toward 0 fights the normalisation itself.
            no_decay_params.append(param)
        else:
            main_params.append(param)

    groups = [
        {"params": main_params,     "lr": base_lr,          "weight_decay": 0.1},
        {"params": no_decay_params, "lr": base_lr,          "weight_decay": 0.0},
        {"params": ode_params,      "lr": base_lr * 0.33,   "weight_decay": 0.0},
        {"params": polarity_params, "lr": base_lr * 1.67,   "weight_decay": 0.0},
        {"params": lateral_params,  "lr": base_lr * 0.33,   "weight_decay": 0.01},
    ]
    # Drop empty groups
    return [g for g in groups if len(g["params"]) > 0]
