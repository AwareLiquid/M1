"""Export a trained O-series checkpoint to ONNX (+ int8) for browser inference.

Why: the hosted demo goes down whenever its inference backend does (the Modal
workspace behind awareliquid.ai is currently disabled, so every /v1/model call
404s). Shipping the model itself to the page — onnxruntime-web on the WebGPU
execution provider — removes the server from the critical path entirely.

Feasibility was established separately by
`benchmarks/check_onnx_webgpu_feasibility.py`: the liquid core exports and runs
under onnxruntime with max|onnx-torch| ≈ 3.6e-07, at ~5.5 MB per 1M params fp32.
This script is the production version of that path, for a real checkpoint.

Two model-side properties this relies on (do not regress them):
  * attention head-merge uses .reshape(), not .contiguous().view()
  * parallel_scan._next_pow2 coerces its argument to int
Both are required for torch.onnx.export to trace the graph at all.

Usage:
    py -3.11 benchmarks/export_o1_for_browser.py --ckpt checkpoints/o1_48m.pt
    py -3.11 benchmarks/export_o1_for_browser.py --ckpt <path> --seq-len 256 --int8

The exported graph has a FIXED sequence length; the browser pads/truncates to
it. Making it dynamic would require the scan/mask to stop baking in a Python int
and is not needed for a demo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def load_checkpoint(path):
    """Checkpoints written by serve/server.py embed their own config."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"{path}: expected a dict checkpoint, got {type(ckpt)}")
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    cfg_d = ckpt.get("config") or ckpt.get("cfg")
    return state, cfg_d, ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="trained O-series .pt")
    ap.add_argument("--out-dir", default="serve/static/model")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="fixed context window baked into the graph")
    ap.add_argument("--int8", action="store_true",
                    help="also emit a dynamically-quantised int8 graph (~4x smaller)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    from mt_lnn.config import MTLNNConfig
    from mt_lnn.model import MTLNNModel

    state, cfg_d, raw = load_checkpoint(args.ckpt)
    if cfg_d is None:
        raise SystemExit(
            f"{args.ckpt} has no embedded config — cannot rebuild the model. "
            f"Keys present: {sorted(raw.keys())[:12]}")
    cfg = MTLNNConfig(**cfg_d) if isinstance(cfg_d, dict) else cfg_d
    cfg.max_seq_len = max(getattr(cfg, "max_seq_len", args.seq_len), args.seq_len)

    m = MTLNNModel(cfg)
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first: {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first: {unexpected[:3]})")
    m.eval()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"loaded {args.ckpt}: {n_params/1e6:.1f}M params, "
          f"d_model={cfg.d_model}, layers={cfg.n_layers}, vocab={cfg.vocab_size}")

    os.makedirs(args.out_dir, exist_ok=True)
    fp32_path = os.path.join(args.out_dir, "o1.onnx")

    ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len))
    with torch.no_grad():
        ref = m(ids)
    ref = (ref["logits"] if isinstance(ref, dict) else ref).numpy()

    # dynamo=False: torch>=2.9's default exporter rejects the head-merge even
    # with reshape (its decomposition re-introduces aten.view). The legacy
    # TorchScript tracer handles it.
    torch.onnx.export(
        m, (ids,), fp32_path,
        input_names=["input_ids"], output_names=["logits"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    fp32_mb = os.path.getsize(fp32_path) / 2**20
    print(f"exported fp32 -> {fp32_path} ({fp32_mb:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input_ids": ids.numpy()})[0]
    d = float(np.abs(got - ref).max())
    print(f"verify fp32: max|onnx-torch| = {d:.2e} "
          f"{'OK' if d < 1e-3 else 'FAIL — graph does not match PyTorch'}")
    if d >= 1e-3:
        return 1

    manifest = {"model": "o1.onnx", "seq_len": args.seq_len,
                "vocab_size": cfg.vocab_size, "d_model": cfg.d_model,
                "n_layers": cfg.n_layers, "params": n_params,
                "fp32_mb": round(fp32_mb, 1)}

    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        int8_path = os.path.join(args.out_dir, "o1.int8.onnx")
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        int8_mb = os.path.getsize(int8_path) / 2**20
        s8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
        got8 = s8.run(["logits"], {"input_ids": ids.numpy()})[0]
        # int8 is lossy by construction; report the drift rather than gate on
        # an arbitrary tolerance, and compare argmax agreement which is what
        # actually matters for generated text.
        d8 = float(np.abs(got8 - ref).max())
        agree = float((got8.argmax(-1) == ref.argmax(-1)).mean())
        print(f"exported int8 -> {int8_path} ({int8_mb:.1f} MB, "
              f"{fp32_mb/int8_mb:.1f}x smaller)")
        print(f"verify int8: max|int8-torch| = {d8:.3f}, "
              f"argmax agreement = {agree*100:.1f}%")
        manifest.update({"int8_model": "o1.int8.onnx",
                         "int8_mb": round(int8_mb, 1),
                         "int8_argmax_agreement": round(agree, 4)})

    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {mpath}")
    print("\nnext: load this with onnxruntime-web (WebGPU EP) from the demo page; "
          "no inference server required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
