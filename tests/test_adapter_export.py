"""
tests/test_adapter_export.py — M-series adapter 的 Hub 就绪布局。

两个断言，一件事：导出的目录**自己能把自己加载回来**。

1. 布局校验：目录结构 + adapter_config.json 的契约字段（base_model_name_or_path
   + 图重建规格，键名对齐 serve/server_hf.py 读的 checkpoint args）。
2. 从目录加载位等价：adapter 权重逐位相同，前向输出逐位相同。

基座用 tests/test_llama_adapter.py 同款的 ``TinyBackbone``（4 层 / hidden 32），
不下载 Qwen/Llama；``load_mt_adapter_dir`` 接受 nn.Module 作为基座就是为
了让这条路径可测。
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")

from mt_lnn.adapter_export import (
    ADAPTER_CONFIG_NAME,
    MT_WEIGHTS_NAME,
    README_NAME,
    export_adapter_dir,
    read_adapter_config,
    validate_adapter_dir,
)
from mt_lnn.llama_adapter import attach_mt_adapters
from mt_lnn.recipes import load_mt_adapter_dir

# attach_mt_adapters() 的入参 与 checkpoint/图规格的键名 是两套命名，别混用。
ATTACH_KW = {"every": 2, "n_protofilaments": 4, "n_time_scales": 2,
             "map_hidden_dim": 8, "init_scale": 1e-2}
MT_ARGS = {"mt_every": 2, "mt_proto": 4, "mt_scales": 2, "mt_map_hidden": 8,
           "mt_init_scale": 1e-2}
BASE_REF = "stub/tiny-base"


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        return (self.proj(hidden_states),)


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size=32, n_layers=4):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [TinyDecoderLayer(hidden_size) for _ in range(n_layers)]
        )

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def _fresh_backbone(seed: int = 0) -> TinyBackbone:
    """基座权重只由这个种子决定，所以"源模型"和"重新加载的基座"逐位相同。"""
    torch.manual_seed(seed)
    return TinyBackbone()


def _trained_backbone(seed: int = 0) -> TinyBackbone:
    model = _fresh_backbone(seed)
    attach_mt_adapters(model, **ATTACH_KW)
    return model


def _export(model: nn.Module, out_dir: str):
    return export_adapter_dir(
        out_dir, base_model_name_or_path=BASE_REF,
        state_dict=model.state_dict(), args=dict(MT_ARGS), meta={"step": 123},
    )


def test_export_layout_is_hub_ready(tmp_path):
    """布局校验：文件齐全 + 元数据带得走"重建这张图"所需的全部信息。"""
    model = _trained_backbone()
    out_dir = str(tmp_path / "mt-adapter")
    manifest = _export(model, out_dir)

    report = validate_adapter_dir(out_dir)
    assert report["ok"], f"布局校验失败: {report['errors']}"
    assert {README_NAME, ADAPTER_CONFIG_NAME, MT_WEIGHTS_NAME}.issubset(set(report["files"]))
    assert report["has_peft_lora"] is False, "没开 LoRA 时不应出现 peft_lora/"

    cfg = read_adapter_config(out_dir)
    assert cfg["product_line"] == "M"
    assert cfg["base_model_name_or_path"] == BASE_REF
    for key in ("adapter", "mt_every", "mt_proto", "mt_scales", "mt_map_hidden",
                "mt_init_scale", "mt_no_scan", "v2_rank"):
        assert key in cfg["mt"], f"图重建规格缺字段 {key}"
    assert cfg["mt"]["mt_every"] == 2 and cfg["mt"]["mt_proto"] == 4

    assert manifest["mt_tensors"] > 0, "MT adapter 权重一个都没导出"
    assert manifest["lora_tensors"] == 0
    assert BASE_REF in (tmp_path / "mt-adapter" / README_NAME).read_text(encoding="utf-8")


def test_load_from_dir_restores_adapter_bit_exact(tmp_path):
    """从目录重建：adapter 权重逐位相同，前向输出逐位相同。"""
    source = _trained_backbone()
    out_dir = str(tmp_path / "mt-adapter")
    _export(source, out_dir)

    reloaded, info = load_mt_adapter_dir(_fresh_backbone(seed=0), out_dir, verbose=False)
    assert info["wrapped_layer_indices"] == [1, 3]
    assert info["missing_mt"] == [], f"有 adapter 权重没被灌进去: {info['missing_mt']}"
    assert info["unexpected_mt"] == [], f"有多余的 adapter 权重: {info['unexpected_mt']}"

    src = {k: v for k, v in source.state_dict().items() if "mt_adapter" in k}
    dst = {k: v for k, v in reloaded.state_dict().items() if "mt_adapter" in k}
    assert set(src) == set(dst), "adapter 参数名集合不一致（图重建错了）"
    for name in src:
        assert torch.equal(src[name], dst[name]), f"adapter 权重 {name} 不是逐位相等"

    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        assert torch.equal(source(x), reloaded(x)), "重建后的前向输出不是逐位一致"
