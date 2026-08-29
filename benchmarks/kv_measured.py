"""Measured validation of the KV-frontier ledger on real models.

The frontier ledger (benchmarks/kv_frontier.py) is exact analytic arithmetic;
this script re-measures its three structural claims on REAL model runs:

  arm 1  real HF forward -> actual fp16 cache nbytes must equal the analytic
         formula EXACTLY (the cache is contiguous, no hidden metadata);
  arm 2  the REAL cache tensors, quantized with KIVI's asymmetric scheme
         (per-channel K / per-token V, fp16 scale per group, zero-point at
         data width, bit-packed) -> measured bytes at the ledger's grouping
         (one group per channel/token = the published floor) and with
         KIVI-style g=32 sub-grouping (the real metadata overhead);
  arm 3  eviction = the sink+window slice of the real cache -> measured
         constant bytes beyond the cap, fp16 and 2-bit.

Models: TinyLlama-1.1B (GQA=4, usually already cached) and
unsloth/Llama-3.2-1B (GQA=8, ungated mirror). Zero new dependencies,
CPU/MPS only, no training.

Honest direction: any measured excess over the ledger floor (sub-grouping
scales, zero-point bytes) makes the real opponent BIGGER than published, so
the ledger's ARR advantages are LOWER BOUNDS. The fp16 row validates the
formula exactly; the g=32 row bounds how much the floor understates a real
KIVI deployment.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODELS = "TinyLlama/TinyLlama-1.1B-Chat-v1.0,unsloth/Llama-3.2-1B"
_DEFAULT_CONTEXTS = "512,2048,8192"
_SINK, _WINDOWS = 4, (512, 4096)
_SCALE_BYTES = 2                    # fp16 scale per quantization group
_ZERO_BYTES = 1                     # zero-point at data width (2-4 bit)
_K_GROUP_AXIS, _V_GROUP_AXIS = 2, 3  # cache tensor layout (B, H, T, D)


def main():
    args = _parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    contexts = [int(x) for x in args.contexts.split(",")]
    payload = {"device": device,
               "scale_accounting": "fp16 scale/group + zero-point at data width",
               "models": [measure_model(m, contexts, device)
                          for m in args.models.split(",") if m]}
    write_outputs(payload, args.out_dir)
    _report(payload)
    return 0


def measure_model(model_id, contexts, device):
    """Load one real model, run every arm at every context, return rows."""
    model, dtype_name = load_model(model_id, device)
    geo = geometry(model.config) | {"model": model_id, "dtype": dtype_name}
    rows = []
    for T in contexts:
        tensors = forward_cache(model, T, device)
        row = geo | {"context": T,
                     **arm_fp16(tensors, geo["n_layers"], geo["n_kv"],
                                geo["head_dim"], T)}
        row |= arm_kivi(tensors, bits=2)
        row |= arm_kivi(tensors, bits=4)
        if T >= _SINK + max(_WINDOWS):
            row["evict"] = arm_evict(tensors)
        rows.append(row)
    del model
    return geo | {"rows": rows}


def load_model(model_id, device):
    """Load in fp16 with a float32 CPU fallback; return (model, dtype name)."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16).to(device)
        return model.eval(), "fp16"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32).to(device)
        return model.eval(), "fp32"


def geometry(cfg):
    """(layers, kv heads, head dim) straight from the model's own config."""
    heads = cfg.num_attention_heads
    return {"n_layers": cfg.num_hidden_layers,
            "n_kv": getattr(cfg, "num_key_value_heads", heads),
            "head_dim": getattr(cfg, "head_dim",
                                cfg.hidden_size // heads)}


def forward_cache(model, T, device):
    """Real forward over T tokens -> per-layer (K, V) CPU tensors, own dtype."""
    ids = torch.randint(0, model.config.vocab_size, (1, T), device=device)
    with torch.no_grad():
        out = _forward(model, ids)
    past = out.past_key_values
    layers = getattr(past, "layers", None)
    pairs = ([(l.keys, l.values) for l in layers] if layers is not None
             else list(zip(past.key_cache, past.value_cache)))
    return [(k.detach().cpu(), v.detach().cpu()) for k, v in pairs]


def _forward(model, ids):
    try:
        return model(input_ids=ids, use_cache=True, logits_to_keep=1)
    except TypeError:
        return model(input_ids=ids, use_cache=True)


def arm_fp16(tensors, L, n_kv, d, T):
    """Arm 1: actual cache bytes vs the exact analytic formula."""
    measured = sum(t.numel() * t.element_size() for k, v in tensors
                   for t in (k, v))
    formula = 2 * L * n_kv * d * T * tensors[0][0].element_size()
    return {"fp16_measured": measured, "fp16_formula": formula,
            "fp16_dev_pct": round(100 * (measured - formula) / formula, 4)}


def arm_kivi(tensors, bits):
    """Arm 2: real tensors quantized KIVI-style; floor grouping and g=32."""
    out = {}
    for tag, g in (("floor", None), ("g32", 32)):
        out[f"{bits}bit_{tag}"] = sum(
            kivi_bytes(k, bits, _K_GROUP_AXIS, g)
            + kivi_bytes(v, bits, _V_GROUP_AXIS, g) for k, v in tensors)
    out[f"{bits}bit_g32_extra_pct"] = round(
        100 * (out[f"{bits}bit_g32"] / out[f"{bits}bit_floor"] - 1), 2)
    return out


def arm_evict(tensors):
    """Arm 3: sink+window slice of the real cache, fp16 and 2-bit."""
    keep = _SINK + max(_WINDOWS)
    return {"keep": keep,
            "evict_fp16": sum(t[..., :keep, :].numel() * t.element_size()
                              for k, v in tensors for t in (k, v)),
            "evict_2bit": sum(
                kivi_bytes(k[..., :keep, :], 2, _K_GROUP_AXIS, None)
                + kivi_bytes(v[..., :keep, :], 2, _V_GROUP_AXIS, None)
                for k, v in tensors)}


def kivi_bytes(t, bits, axis, group):
    """Asymmetric-quant bytes for one cache tensor, grouped along `axis`.

    (B, H, S, O): groups partition the S side into ceil(S/group) chunks per
    (head, O-row) — K quantizes per channel (groups along T), V per token
    (groups along D). group=None = one group per (head, O-row), the ledger
    floor. Storage: numel*bits bit-packed + per-group fp16 scale + zero-point
    at data width.
    """
    s, o = t.shape[axis], t.shape[3] if axis == 2 else t.shape[2]
    n_groups = t.shape[1] * o * (1 if group is None else -(-s // group))
    return (t.numel() * bits + 7) // 8 + n_groups * (_SCALE_BYTES + _ZERO_BYTES)


def write_outputs(payload, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kv_measured.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, "kv_measured.md"), "w") as f:
        f.write(render_md(payload))


def render_md(payload):
    lines = ["# KV ledger — measured validation on real models", "",
             f"Device: `{payload['device']}` — " + payload["scale_accounting"],
             "", "fp16 rows must match the formula exactly; floor vs g32 "
             "bounds the metadata the ledger's floor omits (real KIVI sits "
             "between them; the excess enlarges the opponent, so published "
             "ARR advantages are lower bounds).", ""]
    for m in payload["models"]:
        lines += [f"## {m['model']}", "",
                  f"L={m['n_layers']}, n_kv={m['n_kv']}, "
                  f"head_dim={m['head_dim']}, dtype={m['dtype']}", "",
                  "| T | fp16 real | formula | dev | 2-bit floor | 2-bit g32 "
                  "| g32 extra |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['context']} | {r['fp16_measured']:,} | "
                  f"{r['fp16_formula']:,} | {r['fp16_dev_pct']}% | "
                  f"{r['2bit_floor']:,} | {r['2bit_g32']:,} | "
                  f"+{r['2bit_g32_extra_pct']}% |"
                  for r in m["rows"]]
        ev = [r for r in m["rows"] if "evict" in r]
        if ev:
            e = ev[-1]["evict"]
            lines += ["", f"Eviction @T={ev[-1]['context']}, keep={e['keep']}: "
                      f"fp16 {e['evict_fp16']:,} B · "
                      f"2-bit {e['evict_2bit']:,} B", ""]
    return "\n".join(lines)


def _report(payload):
    for m in payload["models"]:
        devs = ", ".join(f"T={r['context']}: {r['fp16_dev_pct']}%"
                         for r in m["rows"])
        print(f"{m['model']}  fp16 dev [{devs}]  "
              f"2-bit floor={m['rows'][-1]['2bit_floor']:,}B "
              f"g32={m['rows'][-1]['2bit_g32']:,}B")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=_DEFAULT_MODELS)
    ap.add_argument("--contexts", default=_DEFAULT_CONTEXTS)
    ap.add_argument("--out_dir", default=os.path.join(_ROOT, "results"))
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
