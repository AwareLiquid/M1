"""external_plugins — inference-only, zero-intrusion add-ons for the O1 native
MTLNNModel.

HARD RULES (see external_plugins/README.md):
  * O1 trunk (``mt_lnn/``) is FROZEN — these plugins never edit trunk source.
  * Plugins act at INFERENCE only: everything runs under ``torch.no_grad()``,
    no gradients, no training, no weight modification.
  * Plugins attach via removable ``register_forward_hook`` on existing modules.
    Detaching every hook restores the native model bit-for-bit.
  * Two modes per plugin:
      - "observe"  : read activations, compute diagnostics, DO NOT alter output
                     (strict non-intrusion; output bit-identical to native).
      - "intervene": opt-in for ablation; the hook returns a modified activation
                     to measure the effect on PPL. Still no_grad, still removable.

Nothing here is imported by the trunk. Plugins are assembled from the outside
around an already-constructed model.
"""

from .hooks import PluginRunner, PluginHook
from .kuramoto_coupling import KuramotoCoupling
from .early_exit import SettlingEarlyExit

__all__ = [
    "PluginRunner",
    "PluginHook",
    "KuramotoCoupling",
    "SettlingEarlyExit",
]
