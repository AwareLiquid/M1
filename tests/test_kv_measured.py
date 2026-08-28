"""Unit tests for the measured-KV validation harness (benchmarks/kv_measured.py).

Everything runs on a tiny RANDOM Llama (no downloads, CPU, <1s): one real
forward pass produces real cache tensors, then each arm's byte math is
checked against hand-computed integers — the fp16 arm must match the
analytic formula exactly, the KIVI floor and g=32 rows have hand-derived
byte counts, and the eviction slice arithmetic is pinned. Quantized rows are
byte accounting only (values are never read back), matching the script.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.kv_measured import arm_fp16, arm_kivi, kivi_bytes

_L, _H, _D, _T = 2, 2, 16, 64          # layers, kv heads, head dim, tokens
_ESIZE = 4                              # tiny model runs fp32 on CPU


def _tiny_model():
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, num_hidden_layers=_L,
                      num_attention_heads=4, num_key_value_heads=_H,
                      head_dim=_D, intermediate_size=128)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def _real_tensors(T=_T):
    model = _tiny_model()
    ids = torch.randint(0, 64, (1, T))
    with torch.no_grad():
        past = model(input_ids=ids, use_cache=True).past_key_values
    layers = getattr(past, "layers", None)
    pairs = ([(l.keys, l.values) for l in layers] if layers is not None
             else list(zip(past.key_cache, past.value_cache)))
    return [(k.detach().cpu(), v.detach().cpu()) for k, v in pairs]


class TestFp16Arm:
    def test_real_forward_matches_formula_exactly(self):
        tensors = _real_tensors()
        row = arm_fp16(tensors, _L, _H, _D, _T)
        assert row["fp16_measured"] == 2 * _L * _H * _D * _T * _ESIZE
        assert row["fp16_dev_pct"] == 0.0

    def test_cache_bytes_double_with_context(self):
        big = arm_fp16(_real_tensors(2 * _T), _L, _H, _D, 2 * _T)
        small = arm_fp16(_real_tensors(), _L, _H, _D, _T)
        assert big["fp16_measured"] == 2 * small["fp16_measured"]


class TestKiviArm:
    def setUp_tensors(self):
        return _real_tensors()

    def test_floor_grouping_hand_value(self):
        # 8192 elems * 2 bit = 2048 B base; K groups 2*16 / layer, V 2*64
        # / layer -> (2 layers) 320 groups * (2 scale + 1 zero) B
        row = arm_kivi(self.setUp_tensors(), bits=2)
        assert row["2bit_floor"] == 2048 + 320 * 3

    def test_g32_subgrouping_adds_metadata(self):
        # K groups double (ceil(64/32)=2), V unchanged -> +64 groups = +192 B
        row = arm_kivi(self.setUp_tensors(), bits=2)
        assert row["2bit_g32"] == row["2bit_floor"] + 192
        assert row["2bit_g32_extra_pct"] == round(100 * 192 / 3008.0, 2)


class TestEvictionSlice:
    def test_slice_bytes_hand_value(self):
        k = torch.zeros(1, _H, 616, _D)          # T=616 > sink4+w512
        kept = k[..., :516, :]
        assert kept.numel() * _ESIZE == 2 * 516 * 16 * 4
        assert kivi_bytes(kept, 2, 2, None) == 4128 + 32 * 3

    def test_eviction_never_exceeds_full_cache(self):
        tensors = _real_tensors()
        full = arm_kivi(tensors, bits=2)["2bit_floor"]
        sliced = sum(kivi_bytes(k[..., :48, :], 2, 2, None)
                     + kivi_bytes(v[..., :48, :], 2, 3, None)
                     for k, v in tensors)
        assert sliced == 2304 and sliced < full
