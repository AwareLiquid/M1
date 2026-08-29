"""Unit tests for the KV-compression frontier ledger (exact byte accounting).

Spot-checks the analytic KV byte formula against hand-computed integers
(including the fp16 anchor that must reproduce the published 1.5 MB @512 /
3072 MB @1M rows), verifies the crossover solver on synthetic monotone
curves, asserts the ARR line loaded from the committed decode.json is flat
with the documented 0.381 MB value, and checks the frontier's defining
invariant config(T*-1) <= ARR < config(T*) on every real config.
All CPU-only, well under a minute.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.kv_frontier import (build_configs, config_bytes, crossover,
                                    kv_bytes, load_arr_state)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DECODE_JSON = os.path.join(_ROOT, "benchmarks", "results", "decode.json")
_GEO = {"n_layers": 12, "d_head": 64}   # the matched 832x12 decode geometry


class TestAnalyticBytes:
    def test_fp16_reproduces_published_rows(self):
        # 2(K,V) * 12 layers * 1 kv head * 64 dim * T * 2 B — the documented
        # 1.5 MB @512 and 3072 MB @1M rows in BENCHMARKS.md, as exact ints.
        assert kv_bytes(512, 16, 1, _GEO) == 1_572_864
        assert kv_bytes(1_048_576, 16, 1, _GEO) == 3_221_225_472

    def test_2bit_includes_kivi_asymmetric_scales(self):
        base = (2 * 12 * 1 * 64 * 512 * 2 + 7) // 8      # 196_608
        scales = 12 * 1 * 512 * 2 + 12 * 1 * 64 * 2      # V per-token + K per-channel
        assert kv_bytes(512, 2, 1, _GEO) == base + scales == 210_432
        assert kv_bytes(512, 2, 1, _GEO, quant_scales=False) == base


class TestCrossoverSolver:
    def test_linear_curve_crossing_bracket(self):
        t = crossover(lambda T: 408 * T, 399_503)         # 2-bit GQA1 slope
        assert t == 980
        assert 408 * (t - 1) <= 399_503 < 408 * t

    def test_capped_curve_below_reference_is_none(self):
        # flat at 516*614 = 317_224 < 399_503 -> never exceeds the reference
        assert crossover(lambda T: min(T, 516) * 614, 399_503) is None

    def test_capped_curve_above_reference_crosses_like_linear(self):
        fn = lambda T: min(T, 4100) * 3072                # fp16 GQA1, window 4096
        t = crossover(fn, 399_503)
        assert t == 131 and fn(t - 1) <= 399_503 < fn(t)


class TestArrFlatLine:
    def test_decode_json_is_flat_with_documented_value(self):
        arr_bytes, geo = load_arr_state(_DECODE_JSON)
        assert arr_bytes == int(round(0.381 * 2**20))
        assert geo["n_layers"] == 12 and geo["d_head"] == 64


class TestFrontierInvariant:
    def test_t_star_bracket_holds_on_every_real_config(self):
        arr, geo = load_arr_state(_DECODE_JSON)
        for cfg in build_configs():
            if cfg["family"] != "kv":
                continue
            t = crossover(lambda T: config_bytes(cfg, T, geo), arr)
            if t is None:
                assert config_bytes(cfg, 1 << 22, geo) <= arr
            else:
                assert config_bytes(cfg, t - 1, geo) <= arr < config_bytes(cfg, t, geo)
