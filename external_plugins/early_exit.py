"""early_exit.py — Plugin C: settling-based early-exit DIAGNOSTIC.

Idea (see README.md §2-C)
-------------------------
As the hidden state passes through the block stack it (ideally) *settles* toward
a stable representation. This plugin treats the per-block top-level states for
the LAST token as a relaxation trajectory and reuses the trunk's model-free
``mt_lnn.attractor_ops`` instrumentation to ask: **at which block does the
representation enter a tolerance band around its final value?** That block index
is where a genuine early-exit *could* terminate inference.

Why observe-only
----------------
A real early-exit must SKIP later blocks — that needs control inside the trunk
forward loop, which the hard rules forbid. As an external plugin we can only
*measure* the settling block and report the would-be compute saving. Wiring the
actual skip is a separate, trunk-level decision (kept out of scope here).

This plugin never modifies the forward output (mode is always "observe").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .hooks import PluginHook

# Read-only reuse of the trunk's attractor instrumentation. Imported lazily in
# _compute so that a missing/renamed symbol degrades to a cosine fallback rather
# than breaking plugin import.
try:
    from mt_lnn.attractor_ops import settling_time as _settling_time
    _HAVE_ATTRACTOR = True
except Exception:  # pragma: no cover - defensive
    _HAVE_ATTRACTOR = False


@dataclass
class SettlingEarlyExit(PluginHook):
    """Measure the block index at which the last-token state settles.

    Parameters
    ----------
    tol :
        Tolerance band (relative to ``||x_0 - x_final||`` when ``relative``).
    relative :
        Use a relative band (recommended; absolute scale of hidden states
        varies a lot across configs).
    """

    tol: float = 0.05
    relative: bool = True
    name: str = "early_exit"
    mode: str = "observe"

    last_diagnostics: Dict[str, float] = field(default_factory=dict, init=False)
    _states: List[torch.Tensor] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self.last_diagnostics = {}
        self._states = []

    def step(self, layer_idx: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        # A new forward begins at block 0 — clear the per-forward trajectory so
        # diagnostics describe ONE forward, not an accumulation across chunks.
        if layer_idx == 0:
            self._states = []
        # last-token state per block: (B, d_model) → mean over batch for a
        # single representative trajectory (cheap, model-free diagnostic).
        last_tok = x[:, -1, :].detach().to(torch.float32).mean(dim=0)  # (d_model,)
        self._states.append(last_tok)
        self._compute()
        return None  # strict non-intrusion

    def _compute(self) -> None:
        if len(self._states) < 2:
            return
        traj = torch.stack(self._states, dim=0)  # (n_blocks_seen, d_model)
        n = traj.shape[0]
        if _HAVE_ATTRACTOR:
            settle_idx = int(
                _settling_time(traj, target=None, tol=self.tol, relative=self.relative)
            )
        else:  # cosine fallback to the final state
            final = traj[-1]
            d = (traj - final).norm(dim=-1)
            band = self.tol * d[0] if self.relative else self.tol
            inside = d <= band
            settle_idx = int(inside.to(torch.int8).argmax()) if bool(inside.any()) else n - 1
        self.last_diagnostics["settle_block"] = settle_idx
        self.last_diagnostics["n_blocks"] = n
        # fraction of blocks that *could* be skipped after settling
        self.last_diagnostics["skippable_frac"] = max(0.0, (n - 1 - settle_idx) / max(1, n - 1))
