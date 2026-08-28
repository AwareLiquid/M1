"""KV-compression frontier: exact byte ledger of quantized / evicted / GQA KV
caches vs the ARR O(1) carried state, plus crossover-point solving.

The repo's O(1) claim ("1008x @128k / 8063x @1M") was measured against a
fp16, GQA=1 KV cache. External compression attacks that baseline from the
flank: KIVI-style 2-bit asymmetric quantization, 4/8-bit KV, GQA=8 head
counts, and StreamingLLM-style sink+window eviction all shrink the O(T)
term. This script re-audits the claim against those strongest
configurations instead of only the weakest one:

* KV bytes are exact integer arithmetic, no estimates:
    base   2 (K and V) * L * n_kv * d_head * T * bits, bit-packed to bytes
    +V per-token fp16 scales   L * n_kv * T * 2        (KIVI asymmetric, bits<16)
    +K per-channel fp16 scales L * n_kv * d_head * 2    (constant term, bits<16)
  Zero-points cost 0 bytes: they fold into the attention algebra.
* Eviction caps the cached length at sink+window tokens, so the cache is
  constant in T beyond the cap (StreamingLLM/SnapKV-style floor).
* The ARR line is the MEASURED flat snapshot-byte sum loaded from the
  decode.json that `scaling_comparison.py --mode decode` writes, together
  with the geometry it was measured at (L, d_head) — nothing is hand-copied
  from old documents.
* Hybrid honest rows: the M-series hybrid contains attention too, so its KV
  gets the same 2-bit treatment as the opponent's and is charged
  kv@2bit + the measured liquid state (the ledger must not weight only the
  opponent's side).

Crossover T* = the smallest context at which a config's carried bytes EXCEED
the ARR state — the lower bound of the interval where ARR is strictly
smaller. Any config with T* inside the common range (<=128k), or whose
constant memory never exceeds ARR at all, is a claim-boundary row that must
be mirrored in RESULTS.md's "do NOT claim" section.

Outputs benchmarks/results/kv_frontier_ledger.json and kv_frontier.md.
Pure ledger: no models, no training, runs in seconds on CPU.
"""
from __future__ import annotations

import argparse
import json
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DECODE_JSON = os.path.join(_ROOT, "results", "decode.json")
_DEFAULT_OUT_DIR = os.path.join(_ROOT, "results")
_SINK = 4                      # StreamingLLM's canonical attention-sink tokens
_BIT_WIDTHS = (16, 8, 4, 2)    # KV word sizes; 2 = KIVI's floor
_GQA = (1, 8)                  # 1 = repo's conservative ledger, 8 = realistic
_WINDOWS = (512, 4096)         # eviction sliding-window sizes

_SCALE_DOC = ("base = 2*L*n_kv*d_head*T*bits bit-packed; bits<16 adds KIVI-style "
              "asymmetric-quant metadata: per-token fp16 V scales "
              "(L*n_kv*T*2 B) + per-channel fp16 K scales (L*n_kv*d_head*2 B, "
              "constant); zero-points fold away at 0 B")

_FOOTNOTES = """Footnotes — the byte account is not the capability account:

1. Eviction rows that sit BELOW the ARR line win on carried bytes at every T,
   but pay in retrieval: attention is structurally 0.000 on cross-window
   recall while the fast-weight state scores 0.56 (RESULTS.md, "Cross-window
   associative recall"). A token evicted past the window is gone; a binding
   written into the state is not.
2. Hybrid rows apply the same audit to ourselves: the M-series hybrid's KV
   quantizes like anyone else's, so it is charged kv@2bit + the measured
   liquid stream, never the fp16 KV the old tables implied.
3. Every opponent number here is a byte-exact analytic carried-state floor,
   NOT a quality claim: 2-bit KV needs KIVI's asymmetric scheme to stay
   usable, eviction needs a sink, and ARR's own quality beyond 512 tokens is
   unproven (RESULTS.md out-of-window nulls)."""


def main():
    args = _parse_args()
    arr_bytes, geo = load_arr_state(args.decode_json)
    ctxs = [int(x) for x in args.contexts.split(",")]
    cfgs = build_configs()
    rel = os.path.relpath(args.decode_json, os.path.dirname(_ROOT))
    ledger = {"meta": {"decode_json": rel,
                       "arr_state_bytes": arr_bytes, "geometry": geo,
                       "sink": _SINK, "scale_accounting": _SCALE_DOC},
              "contexts": ctxs,
              "configs": [config_row(c, ctxs, arr_bytes, geo) for c in cfgs]}
    frontier = {"arr_state_bytes": arr_bytes,
                "configs": [frontier_row(c, arr_bytes, geo)
                            for c in cfgs if c["family"] == "kv"]}
    write_outputs(ledger, frontier, args.out_dir)
    _report(frontier)
    return 0


def build_configs():
    """The full (bits x GQA x eviction) grid, the ARR reference row, hybrid rows."""
    cfgs = [{"name": "arr_state_measured (O-series)", "family": "arr",
             "bits": None, "n_kv": None, "window": None}]
    for bits in _BIT_WIDTHS:
        for n_kv in _GQA:
            cfgs.append({"name": f"kv_{bits}bit_gqa{n_kv}", "family": "kv",
                         "bits": bits, "n_kv": n_kv, "window": None})
            for w in _WINDOWS:
                cfgs.append({"name": f"evict_sink{_SINK}_w{w}_{bits}bit_gqa{n_kv}",
                             "family": "kv", "bits": bits, "n_kv": n_kv,
                             "window": w})
    for n_kv in _GQA:
        cfgs.append({"name": f"hybrid_kv2bit_gqa{n_kv}", "family": "hybrid",
                     "bits": 2, "n_kv": n_kv, "window": None})
    return cfgs


def config_row(cfg, ctxs, arr_bytes, geo):
    """Ledger row: exact carried bytes (and MB) at every context length."""
    row = dict(cfg)
    row["bytes_at"] = {str(T): (arr_bytes if cfg["family"] == "arr"
                                else config_bytes(cfg, T, geo, arr_bytes))
                       for T in ctxs}
    row["mb_at"] = {k: round(v / 2**20, 3) for k, v in row["bytes_at"].items()}
    return row


def frontier_row(cfg, arr_bytes, geo):
    """Crossover boundary for one KV config: where does ARR stop being smaller?"""
    t_star = crossover(lambda T: config_bytes(cfg, T, geo), arr_bytes)
    ratios = {f"ratio_config_over_arr@{T}":
              round(config_bytes(cfg, T, geo) / arr_bytes, 2)
              for T in (131_072, 1_048_576)}
    row = dict(cfg, t_star=t_star, **ratios)
    row["arr_smaller_interval"] = f"T >= {t_star}" if t_star else "never"
    row["threat_in_common_range"] = t_star is None or t_star <= 131_072
    return row


def render_markdown(ledger, frontier):
    """One page: measured ARR line, the full grid, crossover boundaries, footnotes."""
    geo = ledger["meta"]["geometry"]
    arr_mb = mb(frontier["arr_state_bytes"])
    head = [f"# KV-compression frontier — exact carried-state byte ledger", "",
            f"ARR measured state: **{arr_mb} MB, flat** (loaded from "
            f"`{ledger['meta']['decode_json']}`, never hand-copied). Geometry: "
            f"L={geo['n_layers']}, d_head={geo['d_head']} "
            f"(kv-heads measured: {geo['n_kv_heads_measured']}).", ""]
    return "\n".join(head + ["## Full grid — carried MB, all configs x contexts", "",
                             _grid_table(ledger), "",
                             "## Crossover boundaries — where ARR stops being smaller",
                             "", _frontier_table(frontier), "", _FOOTNOTES, ""])


def _grid_table(ledger):
    ctxs = ledger["contexts"]
    rows = ["| config | bits | gqa | window | " +
            " | ".join(f"T={T}" for T in ctxs) + " |",
            "|" + "---|" * (4 + len(ctxs))]
    for c in ledger["configs"]:
        cells = [c["name"], _dash(c["bits"]), _dash(c["n_kv"]),
                 _dash(c["window"])]
        cells += [f"**{c['mb_at'][str(T)]}**" if c["family"] == "arr"
                  else str(c["mb_at"][str(T)]) for T in ctxs]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _frontier_table(frontier):
    rows = ["| config | ARR smaller for | config/ARR @128k | @1M | boundary threat |",
            "|---|---|---|---|---|"]
    for r in frontier["configs"]:
        flag = "T* <= 128k" if r["t_star"] and r["threat_in_common_range"] else (
            "never exceeds ARR" if r["t_star"] is None else "—")
        rows.append(f"| {r['name']} | {r['arr_smaller_interval']} | "
                    f"{r['ratio_config_over_arr@131072']}x | "
                    f"{r['ratio_config_over_arr@1048576']}x | {flag} |")
    return "\n".join(rows)


def write_outputs(ledger, frontier, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kv_frontier_ledger.json"), "w") as f:
        json.dump({"ledger": ledger, "frontier": frontier}, f, indent=2)
    with open(os.path.join(out_dir, "kv_frontier.md"), "w") as f:
        f.write(render_markdown(ledger, frontier))


def load_arr_state(path):
    """ARR measured flat state (bytes) + geometry, read from the decode JSON.

    This is the anti-hand-copy gate: the file must exist, every row must
    carry the SAME arr_state_mb (that IS the O(1) property), and the geometry
    the analytic side needs travels in the same file.
    """
    with open(path) as f:
        res = json.load(f)
    mbs = [r["arr_state_mb"] for r in res["rows"]]
    if not mbs or len(set(mbs)) != 1:
        raise ValueError(f"ARR state not flat across contexts in {path}: {mbs}")
    geo = {"n_layers": res["n_layers"], "d_head": res["d_head"],
           "n_kv_heads_measured": res["n_kv_heads"]}
    return int(round(mbs[0] * 2**20)), geo


def kv_bytes(T, bits, n_kv, geo, quant_scales=True):
    """Exact KV bytes: bit-packed K+V storage plus KIVI asymmetric scales."""
    L, d = geo["n_layers"], geo["d_head"]
    total = (2 * L * n_kv * d * T * bits + 7) // 8
    if quant_scales and bits < 16:
        total += L * n_kv * T * 2 + L * n_kv * d * 2
    return total


def config_bytes(cfg, T, geo, arr_bytes=0):
    """Carried bytes at context T: eviction caps T, the hybrid adds its state."""
    eff_t = min(T, _SINK + cfg["window"]) if cfg.get("window") else T
    total = kv_bytes(eff_t, cfg["bits"], cfg["n_kv"], geo)
    return total + arr_bytes if cfg.get("family") == "hybrid" else total


def crossover(bytes_fn, ref_bytes, hi=1 << 22):
    """Smallest T in [1, hi] with bytes_fn(T) > ref_bytes, else None.

    bytes_fn must be nondecreasing in T — true for linear KV and for capped
    eviction — so binary search is exact across the flat region too.
    """
    if bytes_fn(hi) <= ref_bytes:
        return None
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bytes_fn(mid) > ref_bytes:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decode_json", default=_DEFAULT_DECODE_JSON,
                    help="decode.json written by scaling_comparison --mode decode")
    ap.add_argument("--out_dir", default=_DEFAULT_OUT_DIR)
    ap.add_argument("--contexts", default="512,2048,8192,32768,131072,524288,1048576")
    return ap.parse_args()


def _report(frontier):
    print(f"ARR measured state: {frontier['arr_state_bytes']} bytes, flat")
    for r in frontier["configs"]:
        print(f"  {r['name']:38s} ARR smaller for {r['arr_smaller_interval']:>15s}"
              f"   config/ARR @1M = {r['ratio_config_over_arr@1048576']}x")


def mb(n_bytes):
    return round(n_bytes / 2**20, 3)


def _dash(v):
    return "—" if v is None else str(v)


if __name__ == "__main__":
    raise SystemExit(main())
