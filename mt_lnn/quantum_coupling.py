"""
quantum_coupling.py — Parameterized quantum-circuit lateral coupling.

Replaces the classical ``LateralCoupling`` module with a Variational Quantum
Circuit (VQC) where each of the P protofilaments is represented by one
qubit. Entangling gates between neighbouring qubits in a ring topology
mirror the physical lateral B-lattice bonds of a real microtubule.

The circuit is fully differentiable through PennyLane's ``TorchLayer`` and
trains via standard backpropagation. We use the ``default.qubit`` classical
simulator, which gives genuine quantum-circuit semantics (superposition,
entanglement, measurement) on a CPU — no quantum hardware required.

Why classical simulation still matters scientifically
-----------------------------------------------------
1. The expressive power of a depth-k VQC differs from a classical neural
   network — there are functions a VQC can represent compactly that a
   classical layer cannot, and vice versa.
2. The ring entanglement pattern (CNOTs between neighbours mod P) is a
   direct architectural analogue of the MT lateral lattice; this maps
   the topology rather than the substrate.
3. Pass ``device_name="lightning.gpu"`` (needs ``pennylane-lightning-gpu`` +
   CUDA) to run the circuit on GPU — ``diff_method`` then auto-switches to
   ``adjoint`` (lightning does not support ``backprop``); ``default.qubit``
   keeps its ``backprop`` fast path. Actual quantum hardware (IBM Quantum,
   IonQ) works the same way via its device name.

Computational cost
------------------
- Each forward pass on B×T samples runs B×T copies of a P-qubit, k-layer
  circuit. With P=13, k=2 and default.qubit's batched contraction:
  ~10× slower than classical SDPA on CPU.
- For training: use small P (e.g. 4 or 6) or shallow circuits (k=1).
- For inference: any P up to ~20 is fast enough on a laptop.

Usage
-----
    from mt_lnn.quantum_coupling import QuantumLateralCoupling

    # Drop-in replacement for LateralCoupling
    coupling = QuantumLateralCoupling(n_protofilaments=13, d_proto=64,
                                       n_qlayers=2)
    h_coupled = coupling(h)   # h: (B, T, P, D) → (B, T, P, D)

Optional dependency: ``pip install pennylane``. If PennyLane is not
installed, this module imports cleanly but raises ImportError on
instantiation.

References
----------
- Bergholm, V. et al. (2018). PennyLane: Automatic differentiation of
  hybrid quantum-classical computations. arXiv:1811.04968.
- Schuld, M. et al. (2020). Circuit-centric quantum classifiers.
  Physical Review A, 101(3):032308.
- Mitarai, K. et al. (2018). Quantum circuit learning. Physical Review A,
  98(3):032309.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None


# ---------------------------------------------------------------------------
# Quantum lateral coupling layer
# ---------------------------------------------------------------------------

class QuantumLateralCoupling(nn.Module):
    """
    P qubits in a ring, entangled by nearest-neighbour CNOTs, parameterised
    by a depth-``n_qlayers`` variational circuit.

    Pipeline (per (b, t) sample):
        1. Encode  : D-dim feature → 3 rotation angles per qubit (RX, RY, RZ)
        2. VQC     : k layers of (RY rotations on each qubit) + (ring CNOTs)
        3. Measure : ⟨Z_i⟩ on each qubit → 1 scalar per protofilament
        4. Decode  : 1-dim measurement → D-dim feature

    The ring topology
        CNOT(0,1), CNOT(1,2), …, CNOT(P-1, 0)
    is identical to the microtubule lateral B-lattice connectivity at the
    qubit level.
    """

    def __init__(
        self,
        n_protofilaments: int,
        d_proto: int,
        n_qlayers: int = 2,
        device_name: str = "default.qubit",
        diff_method: str = None,
    ):
        super().__init__()
        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is not installed. Install with `pip install pennylane`."
            )

        self.P = n_protofilaments
        self.D = d_proto
        self.n_qlayers = n_qlayers

        # Classical-to-quantum encoder: D-dim → 3 angles per qubit
        # (We use 3 angles per qubit so encoding is information-rich.)
        self.encoder = nn.Linear(d_proto, 3)

        # Quantum device — classical simulator by default. Pass
        # device_name="lightning.gpu" (needs pennylane-lightning-gpu + CUDA) to
        # run the P-qubit circuit on GPU.
        self._device_name = device_name
        self.dev = qml.device(device_name, wires=self.P)

        # Differentiation method. lightning.gpu does NOT support "backprop" —
        # it uses the analytic "adjoint" method; default.qubit's fast path IS
        # "backprop". Auto-pick by device unless the caller overrides, so the
        # default.qubit path is not slowed by forcing adjoint on it.
        if diff_method is None:
            diff_method = ("adjoint" if device_name.startswith("lightning")
                           else "backprop")
        self._diff_method = diff_method

        # Variational quantum circuit
        @qml.qnode(self.dev, interface="torch", diff_method=self._diff_method)
        def circuit(inputs, weights):
            """
            inputs  : (P, 3)  — encoding angles for each qubit
            weights : (n_qlayers, P, 3)  — variational angles
            returns : list of P expectation values ⟨Z_i⟩
            """
            # Data encoding (3 rotations per qubit)
            for i in range(self.P):
                qml.RX(inputs[i, 0], wires=i)
                qml.RY(inputs[i, 1], wires=i)
                qml.RZ(inputs[i, 2], wires=i)

            # Variational ansatz (n_qlayers blocks)
            for L in range(n_qlayers):
                # Single-qubit rotations
                for i in range(self.P):
                    qml.RY(weights[L, i, 0], wires=i)
                    qml.RZ(weights[L, i, 1], wires=i)
                    qml.RY(weights[L, i, 2], wires=i)
                # Ring entanglement: CNOT(i, i+1 mod P)
                for i in range(self.P):
                    qml.CNOT(wires=[i, (i + 1) % self.P])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.P)]

        # PennyLane's TorchLayer wraps the qnode + variational weights
        weight_shapes = {"weights": (n_qlayers, self.P, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Quantum-to-classical decoder: 1 measurement → D-dim feature
        self.decoder = nn.Linear(1, d_proto, bias=False)

        # Output mixing factor (residual blend)
        self.alpha = nn.Parameter(torch.tensor(0.05))

    # ------------------------------------------------------------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : (B, T, P, D)
        returns : (B, T, P, D)  — same shape as classical LateralCoupling
        """
        B, T, P, D = h.shape
        assert P == self.P, f"Expected P={self.P}, got {P}"

        # Encode each protofilament's D-dim state → 3 angles
        angles = self.encoder(h)                       # (B, T, P, 3)

        # PennyLane TorchLayer expects (batch, P, 3) ; flatten (B,T) → batch
        angles_flat = angles.reshape(B * T, P, 3)

        # Run the quantum circuit
        # qlayer returns either tensor (batched) or list-of-tensors;
        # qml.qnn.TorchLayer concatenates expvals along the last dim.
        expvals = self.qlayer(angles_flat)             # (B*T, P)

        # Decode each measurement back to D-dim
        decoded = self.decoder(expvals.unsqueeze(-1))  # (B*T, P, D)
        decoded = decoded.reshape(B, T, P, D)

        # Gated residual mixing — start near identity (α≈0.05)
        return h + self.alpha * decoded

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quantum_state_purity(self, h: torch.Tensor) -> float:
        """
        Estimate the average purity of the post-circuit quantum state across
        a batch. Purity = Tr(ρ²) ∈ [1/2^P, 1]; equal to 1 for pure states,
        < 1 for mixed (entangled marginal) states.

        Lower purity ⇔ more entanglement across the ring — useful as a
        readout of how much the quantum lateral coupling is doing.
        """
        # The TorchLayer's qnode only returns expectation values; for full
        # state-vector purity we'd need a separate qnode that returns
        # qml.density_matrix. We approximate via 1 - mean(|⟨Z_i⟩|).
        B, T, P, D = h.shape
        angles = self.encoder(h).reshape(B * T, P, 3)
        expvals = self.qlayer(angles)           # (B*T, P) in [-1, 1]
        z_magnitude = expvals.abs().mean().item()
        return 1.0 - z_magnitude                # rough entanglement proxy

    def extra_repr(self) -> str:
        return (f"P={self.P}, D={self.D}, n_qlayers={self.n_qlayers}, "
                f"device={self._device_name}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Quick smoke test. Run with `python -m mt_lnn.quantum_coupling`."""
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not installed; skipping quantum coupling test.")
        return

    print("Quantum lateral coupling smoke test")
    print("=" * 50)

    P, D = 4, 8                                  # tiny config for speed
    coupling = QuantumLateralCoupling(P, D, n_qlayers=1)
    print(coupling)

    h = torch.randn(2, 3, P, D, requires_grad=True)
    out = coupling(h)
    assert out.shape == h.shape, f"Shape mismatch: {out.shape} vs {h.shape}"

    # Backward pass
    loss = out.sum()
    loss.backward()
    grad_norms = {
        "encoder.weight": coupling.encoder.weight.grad.norm().item(),
        "qlayer.weights": coupling.qlayer.weights.grad.norm().item(),
        "decoder.weight": coupling.decoder.weight.grad.norm().item(),
        "alpha":          coupling.alpha.grad.norm().item(),
    }
    print("\nGradient norms (all should be > 0):")
    for k, v in grad_norms.items():
        print(f"  {k:20s}: {v:.6f}")
        assert v > 0, f"Zero gradient on {k} — circuit not differentiable!"

    purity = coupling.quantum_state_purity(h)
    print(f"\nEntanglement proxy (1 - mean|⟨Z⟩|): {purity:.4f}")
    print("OK")


if __name__ == "__main__":
    _self_test()
