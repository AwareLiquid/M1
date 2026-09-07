"""Post-hoc verdict + tables for the irregular-sampling evidence line.

Reads the four result JSONs, recomputes the hand-written Welch t from the
stored mean/std/n (the same numpy formula the sweep scripts print), applies
the PRE-REGISTERED rules exactly as fixed in the producers' docstrings, and
prints (a) per-domain verdicts, (b) the overall 2-of-3 verdict, and (c)
markdown tables ready for BENCHMARKS.md. Nothing here re-runs training and
nothing here can move a threshold: the rules below are copied verbatim from
the producer scripts and must be kept in sync with them (drift = rework).

Where a producer's rule is ambiguous (air-quality: per-target granularity;
synthetic: how many cv=1 tiers), this file applies the CONSERVATIVE reading
— the one that makes a positive claim HARDEST — and says so next to the
verdict, so a negative cannot be laundered into a positive by ambiguity.

    python benchmarks/analyze_irregular_line.py
"""

from __future__ import annotations

import json
import math
import os

R = "benchmarks/results"
BATTERY = os.path.join(R, "battery_irregular_grud_10seed.json")
AIR = os.path.join(R, "airquality_irregular.json")
SYNTH = os.path.join(R, "synth_ct_control.json")
SYNTH_AD = os.path.join(R, "synth_ct_control_ad.json")
PROFILE = os.path.join(R, "streaming_edge_profile.json")

# Pre-registered rules, verbatim from the producer docstrings / JSON fields:
#   battery: "gru_d absorbs the mt_lnn advantage iff |t(mt_lnn,gru_d)| < 2
#            AND gru_d degradation within 10pp of mt_lnn at the sparsest
#            tier; otherwise the claim survives"
#   all:     "claim holds iff |t|>=2 vs gru_d AND lstm AND gru at the
#             sparsest / high-irregularity tier(s) on >=2 of 3 domains"
SPARSE_T = 2.0
DEG_PP = 10.0


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def welch_better(a, b):
    """t for 'a has lower RMSE than b' from stored stats; >2 ⇒ significant."""
    se = math.sqrt(a["rmse_std"] ** 2 / a["n"] + b["rmse_std"] ** 2 / b["n"])
    return (b["rmse_mean"] - a["rmse_mean"]) / se if se else 0.0


def deg_pct(d0, dN):
    return (dN["rmse_mean"] / d0["rmse_mean"] - 1.0) * 100.0


def fmt(m):
    return f"{m['rmse_mean']:.4f} ± {m['rmse_std']:.4f}"


# --------------------------------------------------------------------------
def battery_verdict(d):
    if d is None:
        return "battery: JSON missing — sweep not finished", None, None
    drops = d["drops"]
    d0, dN = d["results"][f"{drops[0]}"], d["results"][f"{drops[-1]}"]
    archs = [a for a in ("mt_lnn", "mt_lnn_dt", "lstm", "gru", "gru_d",
                         "transformer") if a in dN]
    lines = [f"### battery (held-out {d['test_cell']}, {d['seeds']} seeds, "
             f"drops {drops}) — sparsest tier {drops[-1]:.0%}", ""]
    lines += ["| arch | sparsest RMSE | degradation 0%→sparsest |", "|---|---|---|"]
    for a in archs:
        if a in d0 and a in dN:
            lines.append(f"| {a} | {fmt(dN[a])} | {deg_pct(d0[a], dN[a]):+.1f}% |")
    lines.append("")
    t_ml_grud = welch_better(dN["mt_lnn"], dN["gru_d"])
    t_ml_lstm = welch_better(dN["mt_lnn"], dN["lstm"])
    t_ml_gru = welch_better(dN["mt_lnn"], dN["gru"])
    deg_ml = deg_pct(d0["mt_lnn"], dN["mt_lnn"])
    deg_grud = deg_pct(d0["gru_d"], dN["gru_d"])
    absorbed = abs(t_ml_grud) < SPARSE_T and abs(deg_grud - deg_ml) < DEG_PP
    lines.append(f"- Welch t (positive = mt_lnn lower RMSE): vs gru_d "
                 f"**{t_ml_grud:+.2f}**, vs lstm **{t_ml_lstm:+.2f}**, vs gru "
                 f"**{t_ml_gru:+.2f}**")
    lines.append(f"- GRU-D absorption test (pre-registered): \\|t(mt_lnn,gru_d)\\| "
                 f"< 2 AND degradation gap < {DEG_PP:.0f}pp → "
                 f"{abs(t_ml_grud) < SPARSE_T} AND "
                 f"{abs(deg_grud - deg_ml) < DEG_PP} (gru_d {deg_grud:+.1f}pp vs "
                 f"mt_lnn {deg_ml:+.1f}pp)")
    verdict = ("**NEGATIVE — GRU-D absorbs the mt_lnn advantage** (pre-registered "
               "rule fired; record as null in RESULTS.md, no reframing)" if absorbed
               else "**mt_lnn survives the GRU-D control**")
    win = (t_ml_grud >= SPARSE_T and t_ml_lstm >= SPARSE_T
           and t_ml_gru >= SPARSE_T)
    lines.append(f"- Verdict: {verdict}")
    return "\n".join(lines), win, absorbed


def air_verdict(d):
    if d is None:
        return "air quality: JSON missing — sweep not finished", None
    drops = d["drops"]
    d0, dN = d["results"][f"{drops[0]}"], d["results"][f"{drops[-1]}"]
    archs = [a for a in ("mt_lnn", "lstm", "gru", "gru_d", "transformer")
             if a in dN]
    lines = [f"### air quality (held-out {d['test_station']}, {d['seeds']} "
             f"seeds, drops {drops}) — sparsest tier {drops[-1]:.0%}", ""]
    for t in d["targets"]:
        lines.append(f"| arch | {t} RMSE (raw units) |")
        lines.append("|---|---|")
        for a in archs:
            lines.append(f"| {a} | {dN[a][t]['rmse_mean']:.3f} ± "
                         f"{dN[a][t]['rmse_std']:.3f} |")
        lines.append("")
    # conservative reading: the win needs the margin on BOTH targets
    per_target = {}
    for t in d["targets"]:
        ts = {b: welch_better(dN["mt_lnn"][t], dN[b][t])
              for b in ("gru_d", "lstm", "gru")}
        per_target[t] = all(v >= SPARSE_T for v in ts.values())
        lines.append(f"- {t}: t vs gru_d {ts['gru_d']:+.2f}, lstm "
                     f"{ts['lstm']:+.2f}, gru {ts['gru']:+.2f} → "
                     f"{'all significant' if per_target[t] else 'NOT all significant'}")
    win = all(per_target.values())
    lines.append(f"- Verdict (conservative: BOTH targets): "
                 f"{'domain WIN' if win else 'domain not won'}")
    return "\n".join(lines), win


def synth_verdict(d):
    if d is None:
        return "synthetic control: JSON missing — sweep not finished", None
    keys = sorted(d["results"].keys())
    hi = [k for k in keys if d["results"][k].get("dt_cv") == 1.0]
    archs = [a for a in ("mt_lnn", "mt_lnn_dt", "mt_lnn_ad", "lstm", "gru",
                         "gru_d", "transformer") if a in d["results"][hi[0]]]
    lines = [f"### synthetic continuous-time control (Van der Pol μ={d['mu']}, "
             f"{d['seeds']} seeds) — all tiers", ""]
    lines.append("| tier | " + " | ".join(archs) + " |")
    lines.append("|---|" + "---|" * len(archs))
    for k in keys:
        row = [d["results"][k][a]["rmse_mean"] if a in d["results"][k] else None
               for a in archs]
        lines.append(f"| {k} | " +
                     " | ".join("-" if v is None else f"{v:.4f}" for v in row) +
                     " |")
    lines.append("")
    tier_wins = []
    for k in hi:
        ts = {b: welch_better(d["results"][k]["mt_lnn"], d["results"][k][b])
              for b in ("gru_d", "lstm", "gru")}
        w = all(v >= SPARSE_T for v in ts.values())
        tier_wins.append(w)
        lines.append(f"- {k}: t vs gru_d {ts['gru_d']:+.2f}, lstm {ts['lstm']:+.2f},"
                     f" gru {ts['gru']:+.2f} → {'WIN' if w else 'not won'}")
    # conservative reading: need the margin at a MAJORITY of the cv=1 tiers
    win = sum(tier_wins) * 2 > len(tier_wins)
    lines.append(f"- Verdict (conservative: majority of cv=1 tiers "
                 f"{sum(tier_wins)}/{len(tier_wins)}): "
                 f"{'domain WIN' if win else 'domain not won'}")
    mp = d.get("mechanism_probe")
    if mp:
        lines.append(f"- Mechanism probe (pre-registered R1/R2, recomputed "
                     f"from the stored pairwise): {mp['verdict']} "
                     f"(R1={mp['r1_in_distribution']} wins={mp['r1_wins']}, "
                     f"R2={mp['r2_distribution_shift']})")
    return "\n".join(lines), win


def synth_ad_verdict(d):
    """Independent recheck of the pre-registered mt_lnn_ad rules R2'/R1'.

    Recomputes Welch t from the stored mean/std/n (NOT from the producer's
    pairwise dict), applies the rules exactly as fixed in the producer's
    docstring, and compares against the verdict the producer stamped into
    the JSON. Missing arch/tier counts as not significant (conservative).
    """
    if d is None:
        return "adaptive-decay probe: JSON missing — run not finished", None
    res = d["results"]
    tiers = [k for k in res if not res[k].get("extrap")]
    hi = [k for k in tiers if res[k].get("dt_cv") == 1.0]
    ek = next((k for k in res if res[k].get("extrap")), None)
    archs = [a for a in ("mt_lnn", "mt_lnn_dt", "mt_lnn_ad", "lstm", "gru",
                         "gru_d", "transformer") if a in res[tiers[0]]]
    lines = [f"### adaptive-decay probe mt_lnn_ad (synth grid re-run, "
             f"μ={d['mu']}, {d['seeds']} seeds, synth_ct_control_ad.json)",
             "", "| tier | " + " | ".join(archs) + " |",
             "|---|" + "---|" * len(archs)]
    for k in sorted(res.keys()):
        row = [res[k][a]["rmse_mean"] if a in res[k] else None for a in archs]
        lines.append(f"| {k} | " + " | ".join(
            "-" if v is None else f"{v:.4f}" for v in row) + " |")
    lines.append("")

    def beats(tier, a, b):
        if tier not in res or a not in res[tier] or b not in res[tier]:
            return None
        return welch_better(res[tier][a], res[tier][b])

    r2_detail = {o: (beats(ek, "mt_lnn_ad", o) if ek else None)
                 for o in ("gru_d", "lstm", "gru", "mt_lnn_dt")}
    r2 = all(t is not None and t >= SPARSE_T for t in r2_detail.values())
    lines.append(f"- R2' (shift tier {ek}): " + ", ".join(
        f"vs {o} {'n/a' if t is None else f'{t:+.2f}'}"
        for o, t in r2_detail.items()) + f" → {'PASS' if r2 else 'FAIL'}")
    r1_wins = 0
    for k in hi:
        ts = {o: beats(k, "mt_lnn_ad", o) for o in ("gru_d", "lstm", "gru")}
        w = all(t is not None and t >= SPARSE_T for t in ts.values())
        r1_wins += int(w)
        lines.append(f"- R1' {k}: " + ", ".join(
            f"vs {o} {'n/a' if t is None else f'{t:+.2f}'}"
            for o, t in ts.items()) + f" → {'WIN' if w else 'not won'}")
    r1 = r1_wins >= 2
    if r1 and r2:
        v = "REOPENED — adaptive multi-scale decay (R2'+R1')"
    elif r2:
        v = "NARROW — shift-robustness only (R2')"
    elif r1:
        v = "PARTIAL — in-distribution only (R1')"
    else:
        v = "ARCHIVED FOR GOOD — double negative with mt_lnn_dt"
    lines.append(f"- Independent verdict (mapping as registered): {v}")
    mp = d.get("mechanism_probe_ad")
    if mp:
        match = (mp["r2_prime_distribution_shift"] == r2
                 and mp["r1_prime_in_distribution"] == r1)
        lines.append(f"- Producer JSON stamped R2'="
                     f"{mp['r2_prime_distribution_shift']}, R1'="
                     f"{mp['r1_prime_in_distribution']} (wins="
                     f"{mp['r1_prime_wins']}) → recheck "
                     f"{'MATCH' if match else 'MISMATCH'}")
        lines.append(f"  producer verdict: {mp['verdict']}")
    return "\n".join(lines), (r1, r2)


def profile_table(p):
    if p is None:
        return "deployment profile: JSON missing — chain not finished"
    sz, lat, st = p["table"]["sizes"], p["table"]["latency_us_per_step"], \
        p["table"]["streaming_state_bytes"]
    lines = ["### CPU / ONNX deployment profile", "",
             "| arch | int8 KiB | 1-core µs/step | all-core µs/step | "
             "streaming state B |", "|---|---|---|---|---|"]
    for a in ("mt_lnn", "lstm", "gru"):
        lines.append(f"| {a} | {sz[a]['int8_bytes']/2**10:.1f} | "
                     f"{lat['single_core'][a]:.1f} | {lat['all_cores'][a]:.1f} | "
                     f"{st[a]:,} |")
    lines.append(f"| mt_lnn via ORT | — | {lat['single_core']['mt_lnn_onnx']:.1f} "
                 f"| {lat['all_cores']['mt_lnn_onnx']:.1f} | — |")
    lines.append("")
    lines.append(f"ONNX parity max\\|onnx−torch\\| = "
                 f"{p['table']['onnx_parity_max_abs']:.2e} (gate ≤1e-5: "
                 f"{'PASS' if p['gates']['onnx_parity_le_1e-5'] else 'FAIL'}); "
                 f"variable-Δt worst = {p['table']['variable_dt_max_abs']:.2e} "
                 f"({'PASS' if p['gates']['variable_dt_consistent'] else 'FAIL'}). "
                 f"O(1) state is a property of any RNN — lstm/gru are also flat "
                 f"and smaller; the claim is the combination.")
    return "\n".join(lines)


def main():
    b, air, s, p = load(BATTERY), load(AIR), load(SYNTH), load(PROFILE)
    sa = load(SYNTH_AD)
    bt, bwin, absorbed = battery_verdict(b)
    at, awin = air_verdict(air)
    st, swin = synth_verdict(s)
    sat, _adflags = synth_ad_verdict(sa)
    print(bt, "\n")
    print(at, "\n")
    print(st, "\n")
    print(sat, "\n")
    print(profile_table(p), "\n")

    wins = [w for w in (bwin, awin, swin) if w is not None]
    n = len(wins)
    print("=" * 70)
    print(f"OVERALL (pre-registered, >=2 of available domains, {n} ready): "
          f"{sum(1 for w in wins if w)}/{n} domains won")
    if n < 3:
        print("…incomplete: not all three sweeps have landed their JSON yet.")
    elif sum(wins) >= 2:
        print("=> the irregular-sampling robustness claim SURVIVES the GRU-D "
              "control and generalises to >=2 of 3 task domains.")
    else:
        print("=> NEGATIVE overall: fewer than 2 of 3 domains meet the "
              "pre-registered bar. Record as negative in RESULTS.md; no "
              "reframing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
