"""gwtb_pull.py — Plugin B: GWT top-down PULL (SCAFFOLD — not yet implemented).

Idea (see README.md §2-B)
-------------------------
Use a small "driver" cluster of units to PULL the global-workspace
representation toward a top-down target, as an inference-time replacement for
part of the attention-based modulation. Conceptually this couples a handful of
driver oscillators/units to the workspace state and lets that coupling bias the
broadcast — closer to a cortical top-down feedback loop than concatenating
control tokens.

Why this is a stub for now
--------------------------
* Done right, this should hook the TOP-LEVEL workspace, i.e. the
  ``CompetitiveGWTBLayer`` at ``model.gwtb`` (not the per-block blocks), and
  read its bid/broadcast tensors. That coupling point needs to be pinned
  against the real ``mt_lnn/gwtb.py`` interface before we commit code, to keep
  the "zero trunk edit / detach-to-restore" guarantee.
* Plugin A (Kuramoto) + Plugin C (early-exit) are the first ablation; B lands
  after A's smoke + ablation confirms the phase machinery and hook plumbing.

When implemented it will:
  * subclass :class:`PluginHook` with ``mode="intervene"``;
  * derive a low-rank driver bias from a small fixed cluster;
  * apply it to the workspace output via the same removable forward-hook path
    (attach to ``model.gwtb`` instead of ``model.blocks`` — PluginRunner will
    gain a ``target_attr`` option), under ``torch.no_grad()``, param-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from .hooks import PluginHook


@dataclass
class GWTBPull(PluginHook):
    """Placeholder. Instantiation is allowed (so configs can reference it), but
    :meth:`step` is a strict no-op until the workspace coupling is pinned."""

    name: str = "gwtb_pull"
    mode: str = "observe"
    last_diagnostics: Dict[str, float] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self.last_diagnostics = {}

    def step(self, layer_idx: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        # Intentionally inert. See module docstring for the planned design.
        return None
