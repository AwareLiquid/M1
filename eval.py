"""
eval.py — Evaluation utilities for MT-LNN.

What this script does
---------------------
1. Standard perplexity (PPL) on a held-out token stream.
2. **Long-context PPL** via sliding-window evaluation — exercises the MT-LNN's
   recurrent h_prev state and per-head GTP windows past the training length.
3. **Collapse-gate activation statistics** for the GlobalCoherenceLayer
   (Orch-OR collapse rate; ideal ~30–70%, not 0 or 100).
4. **Microtubule structural diagnostics** — τ / γ / polarity / RMC gate /
   W_lat coupling matrix. W_lat is dumped per-layer for inspection
   (optionally as a PNG heatmap if matplotlib is available).

Usage
-----
    python eval.py --ckpt checkpoints/final.pt --diagnostics
    python eval.py --ckpt checkpoints/final.pt --eval_data
    python eval.py --ckpt checkpoints/final.pt --eval_data --long_ctx 2048
    python eval.py --ckpt checkpoints/final.pt --heatmap_dir analysis/
"""

import argparse
import dataclasses
import json
import math
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mt_lnn import (
    MTLNNConfig, MTLNNModel,
    compute_phi_hat_from_model, phi_hat_anesthesia_sweep, anesthesia_test_result,
)
from mt_lnn.utils import load_checkpoint


# ---------------------------------------------------------------------------
# Self-contained dataset (no dependency on train.py — fixes prior bug)
# ---------------------------------------------------------------------------

class MemoryDataset(Dataset):
    """A flat in-RAM token tensor sliced into (seq_len + 1) windows."""

    def __init__(self, tokens: torch.Tensor, seq_len: int, stride: Optional[int] = None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride or seq_len

    def __len__(self):
        return max(1, (len(self.tokens) - self.seq_len - 1) // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.tokens[start: start + self.seq_len + 1]
        # Labels must be ALIGNED with inputs: MTLNNModel.forward shifts
        # internally (shift_logits = logits[:, :-1] vs shift_labels =
        # labels[:, 1:]). Returning chunk[1:] here would DOUBLE-shift into a
        # 2-ahead objective, reporting an artificially low (wrong) PPL that
        # masks a checkpoint which cannot actually do next-token prediction.
        # See tests/test_label_alignment.py and train.py BinDataset.
        x = chunk[:-1]
        return x, x.clone()


# ---------------------------------------------------------------------------
# Standard perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(model: MTLNNModel, dataloader: DataLoader, device: str) -> float:
    model.eval()
    total_nll, n_tokens = 0.0, 0
    for inp, lbl in dataloader:
        inp, lbl = inp.to(device), lbl.to(device)
        out = model(inp, labels=lbl)
        total_nll += out["loss"].item() * lbl.numel()
        n_tokens += lbl.numel()
    return math.exp(total_nll / max(n_tokens, 1))


# ---------------------------------------------------------------------------
# Long-context sliding-window PPL — uses the model's dual cache so we
# evaluate context lengths beyond the training seq_len.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_long_context_ppl(
    model: MTLNNModel,
    tokens: torch.Tensor,            # (T_total,)
    context_len: int,
    chunk_size: int = 256,
    device: str = "cuda",
    max_chunks: int = 50,
) -> float:
    """
    Stream tokens through the model in chunks of `chunk_size`, threading both
    the KV cache and the LNN recurrent state. PPL is computed on every
    predicted token across the full `context_len` window.

    This is the test the MT-LNN should win on: the recurrent h_prev should
    let it extrapolate past its training seq_len.
    """
    model.eval()
    ids = tokens[:context_len].to(device).unsqueeze(0)        # (1, context_len)
    T = ids.shape[1]

    total_nll, n = 0.0, 0
    cache = None

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = ids[:, start:end]
        if chunk.shape[1] < 2:
            break

        # Predict token t+1 from t. Labels must be ALIGNED with inputs because
        # MTLNNModel.forward shifts internally (shift_labels = labels[:, 1:]);
        # passing chunk[:, 1:] here double-shifts into a 2-ahead objective and
        # reports an artificially low (wrong) PPL. See tests/test_label_alignment.py.
        inp = chunk[:, :-1]
        lbl = inp.clone()
        if inp.shape[1] == 0:
            break

        out = model(inp, cache=cache, labels=lbl, use_cache=True)
        cache = out["cache"]
        total_nll += out["loss"].item() * lbl.numel()
        n += lbl.numel()

        if (start // chunk_size) + 1 >= max_chunks:
            break

    return math.exp(total_nll / max(n, 1))


# ---------------------------------------------------------------------------
# Collapse-gate statistics over a corpus
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_collapse_stats(
    model: MTLNNModel,
    dataloader: DataLoader,
    device: str,
    max_batches: int = 50,
    fire_threshold: float = 0.5,
) -> dict:
    """
    Track the global-coherence collapse-gate value across a corpus.

    Returns:
      mean_gate    — average gate value ∈ [0, 1]
      activation_rate — fraction of batches where gate > fire_threshold
      gate_values  — raw per-batch values (numpy)
    """
    model.eval()
    coherence_layer = model.coherence
    gate_history = []
    for i, (inp, _) in enumerate(dataloader):
        if i >= max_batches:
            break
        inp = inp.to(device)
        model(inp)                                            # forward triggers gate write
        gate_history.append(coherence_layer.last_gate.item())
    arr = np.asarray(gate_history)
    return {
        "mean_gate": float(arr.mean()) if arr.size else 0.0,
        "activation_rate": float((arr > fire_threshold).mean()) if arr.size else 0.0,
        "gate_min": float(arr.min()) if arr.size else 0.0,
        "gate_max": float(arr.max()) if arr.size else 0.0,
        "n_batches": int(arr.size),
    }


# ---------------------------------------------------------------------------
# W_lat dump / heatmap
# ---------------------------------------------------------------------------

def dump_lateral_coupling(model: MTLNNModel, out_dir: Optional[str] = None) -> dict:
    """
    Per-layer W_lat matrix. Optionally renders a PNG heatmap per layer.
    Returns a dict {layer_idx: {'matrix': np.ndarray, 'off_diag_norm': float}}.
    """
    result = {}
    for i, block in enumerate(model.blocks):
        W = block.lnn.lateral.W_lat.detach().cpu().float().numpy()
        eye = np.eye(W.shape[0])
        off_diag_norm = float(np.linalg.norm(W - eye))
        result[i] = {"matrix": W, "off_diag_norm": off_diag_norm}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for i, info in result.items():
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(info["matrix"], cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_title(f"Layer {i} W_lat  ‖off-diag‖={info['off_diag_norm']:.3f}")
                ax.set_xlabel("protofilament j")
                ax.set_ylabel("protofilament i")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"W_lat_layer{i:02d}.png"), dpi=110)
                plt.close(fig)
            print(f"  W_lat heatmaps → {out_dir}/")
        except ImportError:
            print("  (matplotlib not installed; skipping heatmap render)")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    valid = {f.name for f in dataclasses.fields(MTLNNConfig)} - {"d_proto", "d_proto_total"}
    config = MTLNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = MTLNNModel(config).to(device)
    load_checkpoint(ckpt_path, model)
    model.eval()
    return model, config


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config = load_model(args.ckpt, device)
    print(f"Loaded {model.get_num_params()/1e6:.1f}M param MT-LNN "
          f"(d_model={config.d_model}, n_layers={config.n_layers}, "
          f"vocab={config.vocab_size}, max_seq_len={config.max_seq_len})")

    # ---- MT structural diagnostics ----
    if args.diagnostics:
        diag = model.get_mt_diagnostics()
        print("\n=== MT diagnostics ===")
        for k, v in sorted(diag.items()):
            print(f"  {k:36s}: {v:+.4f}")

    # ---- W_lat dump + (optional) heatmap render ----
    if args.heatmap_dir or args.dump_w_lat:
        print("\n=== Lateral coupling W_lat ===")
        lateral = dump_lateral_coupling(model, args.heatmap_dir)
        for i, info in lateral.items():
            print(f"  layer {i:2d}  ‖W_lat - I‖ = {info['off_diag_norm']:.4f}")
            if args.dump_w_lat:
                print(np.round(info["matrix"], 3))

    # ---- Φ̂ + Anesthesia Validation Protocol ----
    if args.phi_hat or args.anesthesia_test:
        # Use a random or supplied prompt batch
        N_samples = args.phi_batch
        ids = torch.randint(0, config.vocab_size,
                             (1, min(N_samples, config.max_seq_len)),
                             device=device)

        if args.phi_hat:
            phi = compute_phi_hat_from_model(model, ids, K=args.phi_K, k_nn=args.phi_k_nn)
            print(f"\n=== Φ̂ (information integration) ===")
            print(f"  K={args.phi_K}, k_nn={args.phi_k_nn}, n_samples={ids.numel()}")
            print(f"  Φ̂ = {phi:.4f}  (higher = more integrated)")

        if args.anesthesia_test:
            kappas = args.anesthesia_kappas or [1.0, 2.0, 5.0, 10.0]
            sweep = phi_hat_anesthesia_sweep(model, ids, kappas=kappas,
                                              K=args.phi_K, k_nn=args.phi_k_nn)
            result = anesthesia_test_result(sweep, delta=args.anesthesia_delta)
            print(f"\n=== Anesthesia Validation Protocol ===")
            print(f"  Φ̂ vs anesthesia level:")
            for kappa, phi in sweep.items():
                print(f"    κ = {kappa:>4.1f}   Φ̂ = {phi:+.4f}")
            print(f"  Φ̂(κ=1)         = {result['phi_clean']:+.4f}")
            print(f"  Φ̂(κ=max)       = {result['phi_full']:+.4f}")
            print(f"  abs change      = {result['abs_change']:+.4f}")
            print(f"  signed rel chg  = {result['signed_relative_change']*100:+.1f}%")
            print(f"  collapse        = {result['collapse_pct']:.1f}% "
                  f"(threshold {args.anesthesia_delta*100:.0f}%)")
            print(f"  monotone decr.  = {result['monotone_decrease']}")
            print(f"  TEST {'✓ PASSED' if result['passed'] else '✗ FAILED'}")

    # ---- Perplexity + collapse stats + long-context PPL ----
    if args.eval_data:
        # Tokeniser must match the one used at training time.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        if tok.vocab_size != config.vocab_size:
            print(f"WARNING: tokenizer vocab_size {tok.vocab_size} != "
                  f"config.vocab_size {config.vocab_size}. PPL will be meaningless.")

        # Token stream
        from datasets import load_dataset
        ds = load_dataset(args.dataset, args.dataset_config)
        split = "test" if "test" in ds else "validation"
        text = " ".join(ds[split]["text"])
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        print(f"  test corpus: {len(ids):,} tokens")

        # 1) Standard PPL at training context length
        test_ds = MemoryDataset(ids, config.max_seq_len)
        loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)
        ppl = evaluate_perplexity(model, loader, device)
        print(f"\n=== PPL @ seq_len={config.max_seq_len}: {ppl:.2f} ===")

        # 2) Collapse-gate stats over the same corpus
        coll = evaluate_collapse_stats(model, loader, device,
                                        max_batches=args.collapse_batches)
        print("=== Collapse-gate stats ===")
        for k, v in coll.items():
            print(f"  {k:18s}: {v}")
        if coll["activation_rate"] in (0.0, 1.0):
            print("  ⚠ collapse gate is stuck — check coherence_scale / collapse_threshold")

        # 3) Long-context PPL (sliding window with dual cache)
        if args.long_ctx:
            for L in args.long_ctx:
                ppl_long = evaluate_long_context_ppl(
                    model, ids, context_len=L,
                    chunk_size=args.chunk_size, device=device,
                    max_chunks=args.long_ctx_max_chunks,
                )
                print(f"=== Long-context PPL @ {L}: {ppl_long:.2f} "
                      f"(chunk_size={args.chunk_size}) ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True, help="Path to checkpoint .pt file")
    p.add_argument("--diagnostics", action="store_true", help="Print MT structural diagnostics")
    p.add_argument("--eval_data",   action="store_true", help="Run PPL + collapse stats")
    p.add_argument("--long_ctx",    type=int, nargs="*", default=None,
                                       help="Extra context lengths to evaluate, e.g. --long_ctx 2048 4096")
    p.add_argument("--chunk_size",  type=int, default=256,
                                       help="Sliding-window chunk size for long-context PPL")
    p.add_argument("--long_ctx_max_chunks", type=int, default=50)
    p.add_argument("--collapse_batches", type=int, default=50)
    p.add_argument("--batch",       type=int, default=8)
    p.add_argument("--dataset",     default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--tokenizer",   default="gpt2")
    p.add_argument("--heatmap_dir", default=None,
                                       help="If set, render W_lat heatmaps to this directory")
    p.add_argument("--dump_w_lat",  action="store_true", help="Print each W_lat matrix to stdout")
    # Φ̂ and AVP
    p.add_argument("--phi_hat",     action="store_true",
                                       help="Compute Φ̂ information-integration proxy")
    p.add_argument("--anesthesia_test", action="store_true",
                                       help="Run the Anesthesia Validation Protocol")
    p.add_argument("--phi_K",       type=int, default=4)
    p.add_argument("--phi_k_nn",    type=int, default=3)
    p.add_argument("--phi_batch",   type=int, default=512,
                                       help="N samples (token positions) for Φ̂ estimation")
    p.add_argument("--anesthesia_kappas", type=float, nargs="*", default=None)
    p.add_argument("--anesthesia_delta",  type=float, default=0.7,
                                       help="Minimum collapse fraction to pass AVP")
    main(p.parse_args())
