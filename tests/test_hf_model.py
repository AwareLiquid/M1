"""
tests/test_hf_model.py — O-series 的 HF 原生打包（mt_lnn/hf_model.py）。

四个断言，四件事：

1. config 序列化往返：``MTLNNConfig`` -> config.json -> ``MTLNNConfig``，
   逐字段相等；product_line 与图重建字段在盘上真的存在。
2. save/load 权重位等价（沿用仓库 bit-exact 测试口径：``torch.equal``，
   不是 allclose）。
3. 三行代码加载冒烟：``from_pretrained`` 出来的模型前向 == 直接构造的模型
   前向，逐位。
4. generate 一致性：外壳的 generate == 独立构造的 MTLNNModel.generate。

随机小权重是关键设计件（``mt_lnn.hf_model.tiny_o_model``）：vocab 64 /
2 层 / d_model 104，构造与前向在 CPU 上是毫秒级，整套远低于 2min 预算，
且不下载任何大模型。
"""
import dataclasses
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, ".")

pytest.importorskip("transformers", reason="HF 打包测试需要 transformers")

from mt_lnn.config import MTLNNConfig
from mt_lnn.hf_model import MTLNNConfigHF, MTLNNForCausalLM, tiny_o_config, tiny_o_model
from mt_lnn.model import MTLNNModel

VOCAB = 64


def _normalize(value):
    """tuple/list 视为相等（JSON 往返会把 tuple 变成 list）。"""
    if isinstance(value, (tuple, list)):
        return [_normalize(v) for v in value]
    return value


def test_config_roundtrip_preserves_native_fields_and_product_line(tmp_path):
    """config.json 必须能独立重建同一个计算图 —— 这是 O 系列的入场券。"""
    native = tiny_o_config()
    model = tiny_o_model(seed=0)
    model.save_pretrained(tmp_path)

    with open(os.path.join(tmp_path, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["model_type"] == "mtlnn"
    assert raw["product_line"] == "O"
    assert raw["mtlnn_config"]["n_layers"] == 2
    assert raw["mtlnn_config"]["use_gwtb"] is False, "O 系列图开关没写进 config.json"
    assert raw["graph"]["builder"] == "mt_lnn.model:MTLNNModel", "图重建入口没写进 config.json"

    loaded_cfg = MTLNNConfigHF.from_pretrained(tmp_path)
    assert loaded_cfg.product_line == "O"
    restored = loaded_cfg.to_mtlnn_config()
    for field_ in dataclasses.fields(MTLNNConfig):
        if not field_.init:
            continue                      # d_proto / d_proto_total 由 __post_init__ 推导
        want = _normalize(getattr(native, field_.name))
        got = _normalize(getattr(restored, field_.name))
        assert got == want, f"字段 {field_.name} 序列化往返不一致: {got!r} != {want!r}"


def test_save_load_weights_are_bit_exact(tmp_path):
    """保存再加载必须逐位相同；权重绑定也必须保持绑定。"""
    model = tiny_o_model(seed=0)
    model.save_pretrained(tmp_path)
    loaded = MTLNNForCausalLM.from_pretrained(tmp_path).eval()

    src, dst = dict(model.named_parameters()), dict(loaded.named_parameters())
    assert set(src) == set(dst), "参数名集合不一致（图的形状变了）"
    for name in src:
        assert torch.equal(src[name], dst[name]), f"权重 {name} 不是逐位相等"

    if model.config.tie_word_embeddings:
        assert (loaded.model.lm_head.weight.data_ptr()
                == loaded.model.embedding.token_embed.weight.data_ptr()), "加载后权重绑定丢了"


def test_non_persistent_buffers_survive_reload(tmp_path):
    """non-persistent buffer 必须逐位相同 —— 这是 transformers 5 的专属陷阱。

    5.x 的 from_pretrained 一律在 meta device 上建图，加载完用
    ``torch.empty_like`` 把所有 non-persistent buffer 搬回 CPU —— 值是未初始化
    的垃圾。库自带的重算只认 class 名含 RotaryEmbedding 且带 original_inv_freq
    的模块（RoPE 表、注意力距离掩码、GWTB causal mask 都不在其列），所以
    ``MTLNNForCausalLM`` 用 get_init_context + _init_weights 两道措施兜住。
    这条测试就是那两道措施的守卫：它挂了就意味着加载出来的模型会静默算错。
    """
    model = tiny_o_model(seed=0)
    model.save_pretrained(tmp_path)
    loaded = MTLNNForCausalLM.from_pretrained(tmp_path).eval()

    src = dict(model.named_buffers())          # named_buffers 含 non-persistent
    dst = dict(loaded.named_buffers())
    assert set(src) == set(dst), "buffer 名集合不一致"
    assert src, "tiny 模型居然没有 non-persistent buffer，这条测试失去意义"
    for name in src:
        assert torch.equal(src[name], dst[name]), f"buffer {name} 不是逐位相等"


def test_three_line_load_matches_direct_construction(tmp_path):
    """三行代码路径的输出 == 直接构造的输出（逐位）。"""
    source = tiny_o_model(seed=0)
    source.save_pretrained(tmp_path)

    # ---- 用户看到的三行 ----
    from mt_lnn.hf_model import MTLNNForCausalLM
    model = MTLNNForCausalLM.from_pretrained(tmp_path).eval()
    ids = torch.randint(0, VOCAB, (1, 6))
    # -----------------------

    with torch.no_grad():
        expected = source(ids).logits
        actual = model(ids).logits
    assert torch.equal(expected, actual), "加载路径与直接构造的输出不是逐位一致"


def test_generate_matches_native_decoder(tmp_path):
    """外壳的 generate 必须就是 MTLNNModel.generate（数学零改动）。"""
    source = tiny_o_model(seed=0)
    source.save_pretrained(tmp_path)
    loaded = MTLNNForCausalLM.from_pretrained(tmp_path).eval()

    native = MTLNNModel(tiny_o_config())
    native.load_state_dict(source.model.state_dict())     # 严格加载：形状必须完全一致
    native.eval()

    ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        via_wrapper = loaded.generate(ids, max_new_tokens=8, do_sample=False)
        via_native = native.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(via_wrapper, via_native)
    assert via_wrapper.shape == (1, 6 + 8)
