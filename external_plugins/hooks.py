"""hooks.py — removable forward-hook plumbing for inference-only plugins.

Design contract
---------------
* We attach :func:`torch.nn.Module.register_forward_hook` handles to EXISTING
  modules of an already-built ``MTLNNModel`` (typically each ``MTLNNBlock`` in
  ``model.blocks``). We never subclass, monkey-patch, or edit the trunk.
* Every hook body runs inside ``torch.no_grad()``. Plugins must not create
  autograd graph or touch ``model.parameters()`` in-place.
* ``PluginRunner`` is a context manager: ``__enter__`` registers all hooks,
  ``__exit__`` removes every handle, so the model is byte-identical afterwards
  (verified in the ablation smoke test).

Hook semantics (PyTorch)
------------------------
A ``forward_hook(module, inputs, output)`` may *return* a value; if it does,
PyTorch replaces the module's output with it. We exploit this ONLY in
"intervene" mode. In "observe" mode the hook returns ``None`` → output is left
untouched → strict non-intrusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# A plugin is any object exposing:
#   .name : str
#   .mode : "observe" | "intervene"
#   .reset() -> None
#   .step(layer_idx, x) -> Optional[torch.Tensor]
#       x is the block output hidden state (B, T, d_model).
#       Return None to leave the output unchanged (observe), or a same-shape
#       tensor to replace it (intervene). Diagnostics are stashed on the plugin.
class PluginHook:
    """Minimal protocol marker / base class for hookable inference plugins."""

    name: str = "plugin"
    mode: str = "observe"

    def reset(self) -> None:  # called once at the start of each forward
        pass

    def step(self, layer_idx: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError


@dataclass
class PluginRunner:
    """Attach a list of :class:`PluginHook` objects to a model's blocks.

    Parameters
    ----------
    model :
        An already-constructed ``MTLNNModel`` (or anything exposing a
        ``blocks`` ``nn.ModuleList``). Left untouched except for transient
        forward-hook handles.
    plugins :
        Hooks to run, in order, on every block output.
    block_attr :
        Name of the ``nn.ModuleList`` attribute holding the per-layer blocks.
    """

    model: nn.Module
    plugins: List[PluginHook]
    block_attr: str = "blocks"
    _handles: List[Any] = field(default_factory=list, init=False)

    # -- lifecycle -------------------------------------------------------- #
    def _blocks(self) -> nn.ModuleList:
        blocks = getattr(self.model, self.block_attr, None)
        if blocks is None:
            raise AttributeError(
                f"model has no '{self.block_attr}' attribute; cannot attach hooks"
            )
        return blocks

    def attach(self) -> "PluginRunner":
        if self._handles:
            raise RuntimeError("PluginRunner already attached; detach() first")
        blocks = self._blocks()
        for idx, block in enumerate(blocks):
            handle = block.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)
        for p in self.plugins:
            p.reset()
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        for p in self.plugins:
            p.reset()

    # -- the hook body ---------------------------------------------------- #
    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, inputs: Tuple, output: Any):
            # MTLNNBlock returns (x, new_cache); plain blocks may return x.
            if isinstance(output, tuple):
                x = output[0]
                rest = output[1:]
            else:
                x = output
                rest = None

            new_x = x
            with torch.no_grad():
                for p in self.plugins:
                    res = p.step(layer_idx, new_x)
                    if res is not None:
                        if p.mode != "intervene":
                            # A guard: observe-mode plugins must never alter output.
                            raise RuntimeError(
                                f"plugin '{p.name}' returned a tensor in "
                                f"mode='{p.mode}'; only mode='intervene' may "
                                f"modify the forward output"
                            )
                        new_x = res

            if new_x is x:
                return None  # strict non-intrusion: leave output as-is
            # intervene: rebuild the original output container
            if rest is not None:
                return (new_x, *rest)
            return new_x

        return hook

    # -- context-manager sugar ------------------------------------------- #
    def __enter__(self) -> "PluginRunner":
        return self.attach()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.detach()

    # -- diagnostics collation ------------------------------------------- #
    def diagnostics(self) -> Dict[str, Any]:
        """Gather each plugin's stashed per-forward diagnostics by name."""
        out: Dict[str, Any] = {}
        for p in self.plugins:
            out[p.name] = getattr(p, "last_diagnostics", None)
        return out
