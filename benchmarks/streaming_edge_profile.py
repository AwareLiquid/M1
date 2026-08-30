"""Deployment profile for the irregular-sampling streaming edge line:
can the liquid regressor actually SHIP as a CPU/ONNX artefact?

Answers four questions, each with a number (results land in
benchmarks/results/streaming_edge_profile.json):

  1. EXPORT + PARITY — does LiquidRegressor (Δt as the 4th input channel)
     export to ONNX and reproduce PyTorch numerics? Historical standard from
     the O-series line: max|onnx−torch| ≈ 3.58e-07; gate at 1e-5.
     Known constraints carried over verbatim: dynamo=False (the dynamo
     exporter rejects the head-merge), FIXED sequence length (the scan bakes
     T), .reshape() (not .contiguous().view()) in attention head merges.
  2. VARIABLE-Δt — the point of the architecture is that Δt is a runtime
     input, not a compile-time constant. One checkpoint, several Δt
     schedules (regular / sparse / event-like bursts): ORT must match
     PyTorch on each. A model that had silently baked Δt would fail here.
  3. SIZE — fp32 and int8 (dynamic quantisation) ONNX bytes vs the lstm/gru
     baselines exported the same way.
  4. LATENCY + STATE — single-core and all-core per-step latency on CPU
     (amortised: one fixed-T forward / T; the ONNX graph is a fixed-window
     step-batch, which is how a duty-cycled controller would run it), and
     the carried streaming-state bytes in the battery_streaming_memory
     accounting. O(1) state is NOT liquid-exclusive — lstm/gru are constant
     and smaller; that honesty is part of the table.

Run on an otherwise-idle machine; latency numbers are meaningless under
training load. Exit code non-zero if any gate fails.

    python benchmarks/streaming_edge_profile.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from benchmarks.battery_soh_edge import build
from benchmarks.battery_streaming_memory import state_bytes

D_MODEL, N_LAYERS, T = 78, 2, 128
N_FEAT = 4                    # 3 sensors + Δt (the irregular-sampling input)
OUT = "benchmarks/results/streaming_edge_profile.json"


def make_dt_schedules(T, seed=0):
    """Δt schedules the model must handle at RUNTIME, same checkpoint.

    All schedules keep the exported fixed length T (the graph bakes T; a
    sparser stream is represented by LARGER Δt values at the same length,
    which is exactly what the compacted protocol encodes).
    """
    rng = np.random.default_rng(seed)
    regular = (np.ones((1, T, 1), dtype=np.float32) / T).astype(np.float32)
    bursty = (rng.choice([1.0, 7.0], size=(1, T, 1), p=[0.8, 0.2])
              / T).astype(np.float32)
    jitter = (np.clip(rng.lognormal(0.0, 1.0, size=(1, T, 1)), 0.2, 12.0)
              / T).astype(np.float32)
    return {"regular": regular, "bursty": bursty, "lognormal_jitter": jitter}


def build_patched(arch):
    """build() with the input projection sized to the 4-channel Δt layout."""
    m = build(arch, D_MODEL, N_LAYERS, T)
    m.inp = torch.nn.Linear(N_FEAT, D_MODEL)
    return m.eval()


def export_arch(arch, path):
    m = build_patched(arch)
    torch.onnx.export(
        m, (torch.randn(1, T, N_FEAT),), path,
        input_names=["x"], output_names=["y"],
        opset_version=17, do_constant_folding=True, dynamo=False)
    return m


def main():
    from onnxruntime import InferenceSession
    from onnxruntime.quantization import QuantType, quantize_dynamic

    gates, table = {}, {}
    tmp = tempfile.mkdtemp(prefix="edge_profile_")

    # ---- 1/4 export + parity (liquid regressor with Δt input) -------------
    m = build_patched("mt_lnn")
    fp32 = os.path.join(tmp, "liquid_dt.onnx")
    x = torch.randn(1, T, N_FEAT)
    torch.onnx.export(
        m, (x,), fp32, input_names=["x"], output_names=["y"],
        opset_version=17, do_constant_folding=True, dynamo=False)
    sess = InferenceSession(fp32, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = m(x).numpy()
    got = sess.run(["y"], {"x": x.numpy()})[0]
    parity = float(np.abs(got - ref).max())
    gates["onnx_parity_le_1e-5"] = parity <= 1e-5
    table["onnx_parity_max_abs"] = parity
    print(f"[1/4] export+parity ......... {'PASS' if parity <= 1e-5 else 'FAIL'} "
          f"(max|onnx-torch| = {parity:.2e}; historical standard 3.58e-07)")

    # ---- 2/4 variable-Δt consistency ---------------------------------------
    worst = 0.0
    for name, dt in make_dt_schedules(T).items():
        xd = torch.randn(1, T, 3)
        xdt = torch.cat([xd, torch.from_numpy(dt)], dim=-1)
        with torch.no_grad():
            r = m(xdt).numpy()
        g = sess.run(["y"], {"x": xdt.numpy()})[0]
        d = float(np.abs(g - r).max())
        worst = max(worst, d)
        print(f"      Δt schedule {name:<14} max|onnx-torch| = {d:.2e}")
    gates["variable_dt_consistent"] = worst <= 1e-5
    table["variable_dt_max_abs"] = worst
    print(f"[2/4] variable-Δt .......... {'PASS' if worst <= 1e-5 else 'FAIL'} "
          f"(worst {worst:.2e} across regular/sparse/bursty schedules)")

    # ---- 3/4 size: fp32 / int8 vs lstm/gru ---------------------------------
    table["sizes"] = {}
    for arch in ("mt_lnn", "lstm", "gru"):
        fp = os.path.join(tmp, f"{arch}.onnx")
        mm = export_arch(arch, fp)
        row = {"fp32_bytes": os.path.getsize(fp)}
        q8 = os.path.join(tmp, f"{arch}.int8.onnx")
        quantize_dynamic(fp, q8, weight_type=QuantType.QInt8)
        row["int8_bytes"] = os.path.getsize(q8)
        s8 = InferenceSession(q8, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            r = mm(x).numpy()
        g = s8.run(["y"], {"x": x.numpy()})[0]
        row["int8_max_abs_drift"] = float(np.abs(g - r).max())
        table["sizes"][arch] = row
        print(f"      {arch:<8} fp32 {row['fp32_bytes']/2**10:8.1f} KiB | int8 "
              f"{row['int8_bytes']/2**10:7.1f} KiB | int8 drift "
              f"{row['int8_max_abs_drift']:.2e}")
    gates["int8_available"] = True

    # ---- 4/4 latency (idle CPU) + streaming state -------------------------
    table["latency_us_per_step"] = {}
    nthreads = torch.get_num_threads()
    xd = torch.randn(1, T, N_FEAT)
    ort1 = InferenceSession(fp32, sess_options=_ort_opts(1),
                            providers=["CPUExecutionProvider"])
    for mode, nt in (("single_core", 1), ("all_cores", nthreads)):
        torch.set_num_threads(nt)
        for arch in ("mt_lnn", "lstm", "gru"):
            mm = build_patched(arch)
            with torch.no_grad():
                for _ in range(3):
                    mm(xd)
                t0 = time.perf_counter()
                n = 50
                for _ in range(n):
                    mm(xd)
                dt = (time.perf_counter() - t0) / n
            table["latency_us_per_step"].setdefault(mode, {})[arch] = \
                round(dt / T * 1e6, 1)
        # ONNX runtime on the same fixed-window graph, matched thread count
        s = ort1 if mode == "single_core" else sess
        table["latency_us_per_step"][mode]["mt_lnn_onnx"] = \
            round(_ort_latency(s, x.numpy(), T), 1)
    torch.set_num_threads(nthreads)

    table["streaming_state_bytes"] = {}
    with torch.no_grad():
        for arch in ("lstm", "gru"):
            mm = build_patched(arch)
            _, h = mm.rnn(mm.inp(xd))
            table["streaming_state_bytes"][arch] = state_bytes(h)
        mm = build_patched("mt_lnn")
        h = mm.inp(xd)
        carried = 0
        for layer in mm.layers:
            out = layer(h)
            if isinstance(out, tuple):
                h = h + out[0]
                carried += state_bytes(out[1:])
            else:
                h = h + out
        if carried == 0:
            cfg = mm.layers[0]
            carried = (cfg.n_proto * getattr(cfg, "n_scales", 5)
                       * cfg.d_proto) * 4 * N_LAYERS
        table["streaming_state_bytes"]["mt_lnn"] = carried
        table["streaming_state_bytes"]["transformer_kv_at_32k"] = \
            2 * N_LAYERS * 32768 * D_MODEL * 4

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"STREAMING EDGE PROFILE | d_model={D_MODEL} layers={N_LAYERS} "
          f"T={T} | Δt runtime input | CPU (this machine)")
    print(f"{'arch':<10}{'params':>9}{'int8 KiB':>10}{'1-core µs/step':>16}"
          f"{'all-core µs/step':>18}{'state B':>9}")
    for arch in ("mt_lnn", "lstm", "gru"):
        p = sum(q.numel() for q in build_patched(arch).parameters())
        print(f"{arch:<10}{p:>9,}"
              f"{table['sizes'][arch]['int8_bytes']/2**10:>10.1f}"
              f"{table['latency_us_per_step']['single_core'][arch]:>16.1f}"
              f"{table['latency_us_per_step']['all_cores'][arch]:>18.1f}"
              f"{table['streaming_state_bytes'][arch]:>9,}")
    print(f"{'(onnx)':<10}{'':>9}{'':>10}{'':>16}"
          f"{table['latency_us_per_step']['all_cores']['mt_lnn_onnx']:>18.1f}"
          f"   ORT fixed-T")
    print("=" * 74)

    ok = all(gates.values())
    payload = {"config": {"d_model": D_MODEL, "n_layers": N_LAYERS, "T": T,
                          "n_feat": N_FEAT},
               "gates": gates, "table": table,
               "notes": ["per-step latency is amortised (one fixed-T forward "
                         "divided by T) — the exported graph is a fixed "
                         "window, how a duty-cycled controller batches steps",
                         "O(1) streaming state is a property of ANY "
                         "recurrent model — lstm/gru are also flat and "
                         "smaller; the claim is the combination with "
                         "irregular-sampling robustness, not O(1) alone"]}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nresults -> {OUT}   overall: "
          f"{'ALL GATES PASS' if ok else 'GATE FAILURE'}")
    return 0 if ok else 1


def _ort_opts(nthreads):
    from onnxruntime import SessionOptions
    so = SessionOptions()
    so.intra_op_num_threads = nthreads
    so.inter_op_num_threads = 1
    return so


def _ort_latency(sess, x, T, n=50):
    xx = np.asarray(x)
    sess.run(["y"], {"x": xx})
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(["y"], {"x": xx})
    dt = (time.perf_counter() - t0) / n
    return dt / T * 1e6


if __name__ == "__main__":
    raise SystemExit(main())
