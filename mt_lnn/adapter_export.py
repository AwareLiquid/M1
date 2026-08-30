"""
mt_lnn/adapter_export.py — M-series adapter 的 HF Hub 就绪布局。

M 系列 = **冻结 HF 基座 + MT adapter**。基座是别人的（Qwen / Llama /
GLM…），我们只分发适配器，所以目录里不能、也不该有基地权重。目标是
"下载一个小目录 + 一句 from_pretrained 就能跑"。

布局（导出产物）
----------------
::

    <out_dir>/
      README.md                 # 带 YAML front-matter 的模型卡，Hub 直读
      adapter_config.json       # 本文件格式（base_model_name_or_path + 图重建字段）
      mt_adapter.pt             # 只含 MT adapter 权重（torch.save）
      peft_lora/                # 仅当训练时开了 LoRA：PEFT 原生子目录
        adapter_config.json
        adapter_model.bin

为什么要自己一份 adapter_config.json，而不是直接复用 PEFT 的
------------------------------------------------------------
因为 **MT adapter 不是 LoRA**。`MTResidualAdapter` 是注入进 decoder layer
的自定义残差模块（13 protofilaments × 多时间尺度共振 + 可选 fast-weight），
PEFT 的 LoRA/AdaLoRA/IA3 都表达不了它，硬塞进 peft_type 只会得到一份
PEFT 读不懂、我们也加载不回来的目录。所以：

* **LoRA 那一半走 PEFT 原生路径**（peft_lora/ 子目录，`PeftModel.from_pretrained`
  直接可用，社区工具链全部兼容）；
* **MT 那一半走我们自己的 `mt_adapter.pt` + `adapter_config.json`**，
  由 `mt_lnn.recipes.load_mt_adapter_dir()` 加载（约 20 行，内部复用
  `llama_adapter.attach_adapters_from_checkpoint` 的同款重建语义）。

这是"非标模块"必须付的税，写清楚比假装它是 LoRA 更有用。

字段与 serve/server_hf.py 的对齐
--------------------------------
``adapter_config.json["mt"]`` 的键名 **逐字照抄** server_hf.py 从 checkpoint
``args`` 里读的那套（mt_every / mt_proto / mt_scales / mt_map_hidden /
mt_init_scale / mt_dropout / mt_no_scan / adapter / v2_* / no_mt …），
也照抄 train_llama_mt_adapter.py 的 argparse 名。三条路径（训练、服务、
分发）因此共享同一份图重建规格，不会出现"服务能跑、导出的目录加载不回来"。

**不上传任何东西。** 本模块只写本地目录；`push_to_hub` 是外发动作，任何阶段
都由人工执行。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

PRODUCT_LINE_M = "M"

ADAPTER_CONFIG_NAME = "adapter_config.json"
MT_WEIGHTS_NAME = "mt_adapter.pt"
README_NAME = "README.md"
PEFT_SUBDIR = "peft_lora"

# 图重建规格的默认值。键名 = server_hf.py 读的 checkpoint args 名。
MT_GRAPH_FIELDS: Tuple[str, ...] = (
    "adapter", "no_mt",
    "mt_every", "mt_proto", "mt_scales", "mt_map_hidden",
    "mt_dropout", "mt_init_scale", "mt_no_scan",
    "v2_d_proto", "v2_rank", "v2_selective", "v2_no_fw", "v2_fw_dim", "v2_fw_heads",
    "sel_mode",
)
MT_GRAPH_DEFAULTS: Dict[str, Any] = {
    "adapter": "v1",
    "no_mt": False,
    "mt_every": 4,
    "mt_proto": 13,
    "mt_scales": 5,
    "mt_map_hidden": 64,
    "mt_dropout": 0.0,
    "mt_init_scale": 1e-3,
    "mt_no_scan": False,
    "v2_d_proto": 64,
    "v2_rank": 128,
    "v2_selective": False,
    "v2_no_fw": False,
    "v2_fw_dim": 64,
    "v2_fw_heads": 1,
    "sel_mode": "mamba",
}
LORA_FIELDS: Tuple[str, ...] = ("lora_r", "lora_alpha", "lora_dropout", "lora_targets")

__all__ = [
    "ADAPTER_CONFIG_NAME",
    "MT_WEIGHTS_NAME",
    "PEFT_SUBDIR",
    "README_NAME",
    "build_adapter_config",
    "export_adapter_dir",
    "export_from_checkpoint",
    "read_adapter_config",
    "render_readme",
    "split_adapter_state",
    "validate_adapter_dir",
]


def export_adapter_dir(
    out_dir: str,
    *,
    base_model_name_or_path: str,
    state_dict: Dict[str, torch.Tensor],
    args: Optional[Dict[str, Any]] = None,
    recipe: str = "phase5b",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把一份 adapter 权重写成 Hub 就绪目录，返回 manifest。

    ``state_dict`` 是**训练后模型的完整 state_dict**；本函数按 key 名把它切成
    MT 与 LoRA 两份，只落盘这两部分（基座权重不进目录）。``args`` 传训练时的
    argparse 字典（``vars(args)``）；缺省字段用 :data:`MT_GRAPH_DEFAULTS` 补。
    """
    mt_state, lora_state = split_adapter_state(state_dict)
    config = build_adapter_config(
        base_model_name_or_path=base_model_name_or_path,
        mt_args=_mt_args_from(args),
        lora_args=None if not lora_state else _lora_args_from(args),
        recipe=recipe,
        meta=meta,
    )
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, ADAPTER_CONFIG_NAME), config)
    torch.save({"state_dict": mt_state, "meta": config.get("meta", {})},
               os.path.join(out_dir, MT_WEIGHTS_NAME))
    with open(os.path.join(out_dir, README_NAME), "w", encoding="utf-8") as f:
        f.write(render_readme(config))
    if lora_state:
        _write_peft_subdir(os.path.join(out_dir, PEFT_SUBDIR), config, lora_state)
    return {
        "dir": out_dir,
        "config": config,
        "mt_tensors": len(mt_state),
        "mt_params": int(sum(v.numel() for v in mt_state.values())),
        "lora_tensors": len(lora_state),
        "lora_params": int(sum(v.numel() for v in lora_state.values())),
        "files": sorted(os.listdir(out_dir)),
    }


def export_from_checkpoint(
    ckpt_path: str,
    out_dir: str,
    *,
    base_model_name_or_path: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """从 ``train_llama_mt_adapter.py`` 的 ``.pt`` 导出 Hub 就绪目录。

    训练产物是 ``{"step", "model", "state_dict", "args"}``，其中 state_dict 已
    经只含 mt_adapter / lora_ 键、args 已经是 argparse 字典 —— 这里只是把
    ``ckpt["model"]`` 当作 base_model_name_or_path 的默认值搬过来。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = dict(ckpt.get("args") or {})
    base = base_model_name_or_path or args.get("model") or ""
    if not base:
        raise ValueError(
            "无法确定 base_model_name_or_path：checkpoint 没有 args['model']，"
            "请显式传入 base_model_name_or_path。"
        )
    meta = {**(meta or {}), "source_checkpoint": ckpt_path, "step": ckpt.get("step")}
    return export_adapter_dir(
        out_dir, base_model_name_or_path=base,
        state_dict=ckpt.get("state_dict", {}), args=args,
        recipe=str(args.get("recipe", "phase5b")), meta=meta,
    )


def validate_adapter_dir(adapter_dir: str) -> Dict[str, Any]:
    """校验目录结构与元数据完整性，返回报告（不抛异常，交给调用方决定）。

    报告里 ``errors`` 为空即"布局可用"。校验的是**契约**而不是权重内容：
    字段是否存在、指向的权重文件是否真的在盘上、product_line 是否是 M。
    """
    errors: list = []
    cfg = read_adapter_config(adapter_dir) if os.path.isfile(
        os.path.join(adapter_dir, ADAPTER_CONFIG_NAME)) else None
    if cfg is None:
        errors.append(f"缺少 {ADAPTER_CONFIG_NAME}")
        cfg = {}
    if cfg.get("product_line") != PRODUCT_LINE_M:
        errors.append(f"product_line 应为 {PRODUCT_LINE_M!r}，实际 {cfg.get('product_line')!r}")
    if not cfg.get("base_model_name_or_path"):
        errors.append("缺少 base_model_name_or_path（基座引用是 M 系列的加载前提）")
    mt_path = os.path.join(adapter_dir, MT_WEIGHTS_NAME)
    if not os.path.isfile(mt_path):
        errors.append(f"缺少 {MT_WEIGHTS_NAME}")
    if not os.path.isfile(os.path.join(adapter_dir, README_NAME)):
        errors.append(f"缺少 {README_NAME}")
    missing_fields = [k for k in MT_GRAPH_FIELDS if k not in (cfg.get("mt") or {})]
    if missing_fields and not (cfg.get("mt") or {}).get("no_mt", False):
        errors.append(f"mt 图规格缺字段: {missing_fields}")
    peft_dir = os.path.join(adapter_dir, PEFT_SUBDIR)
    has_peft = os.path.isdir(peft_dir)
    if has_peft and not os.path.isfile(os.path.join(peft_dir, ADAPTER_CONFIG_NAME)):
        errors.append(f"{PEFT_SUBDIR}/ 缺少 {ADAPTER_CONFIG_NAME}")
    return {
        "ok": not errors,
        "dir": adapter_dir,
        "config": cfg,
        "errors": errors,
        "has_peft_lora": has_peft,
        "files": sorted(os.listdir(adapter_dir)) if os.path.isdir(adapter_dir) else [],
    }


def read_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    with open(os.path.join(adapter_dir, ADAPTER_CONFIG_NAME), "r", encoding="utf-8") as f:
        return json.load(f)


def build_adapter_config(
    *,
    base_model_name_or_path: str,
    mt_args: Dict[str, Any],
    lora_args: Optional[Dict[str, Any]] = None,
    recipe: str = "phase5b",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """组装 ``adapter_config.json``。``mt`` 块是图重建规格（键名对齐 server_hf）。"""
    return {
        "product_line": PRODUCT_LINE_M,
        "base_model_name_or_path": base_model_name_or_path,
        "recipe": recipe,
        "mt": mt_args,
        "lora": lora_args,
        "graph_backend": "mt_lnn.recipes:load_mt_adapter_dir",
        "meta": dict(meta or {}),
    }


def split_adapter_state(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """按 key 名把完整 state_dict 切成 (MT adapter, LoRA) 两份。

    与 train_llama_mt_adapter.py 的保存过滤条件一致（``"mt_adapter" in k or
    "lora_" in k``），保证"能存就能取"。
    """
    mt = {k: v.detach().cpu() for k, v in state_dict.items() if "mt_adapter" in k}
    lora = {k: v.detach().cpu() for k, v in state_dict.items()
            if "lora_" in k and "mt_adapter" not in k}
    return mt, lora


def render_readme(config: Dict[str, Any]) -> str:
    """Hub 模型卡。front-matter 用 HF 认识的键，正文给出三行加载示例。"""
    base = config["base_model_name_or_path"]
    mt = config.get("mt") or {}
    lora = config.get("lora")
    tags = "\n".join([
        "  - mt-lnn",
        "  - microtubule-adapter",
        "  - peft" if lora else "  - residual-adapter",
    ])
    peft_line = (
        "\nLoRA 部分是标准 PEFT 适配器，可单独 `PeftModel.from_pretrained(base, "
        f"<dir>/{PEFT_SUBDIR})` 加载（不含 MT 模块）。" if lora else ""
    )
    # 没有 LoRA 时 library_name 写 peft 是误导：Hub 会以为这是标准 PEFT 适配器。
    library = "peft" if lora else "transformers"
    return f"""---
base_model: {base}
library_name: {library}
tags:
{tags}
---

# MT-LNN adapter for {base}

MT 残差适配器（{mt.get('adapter', 'v1')}）：每 {mt.get('mt_every', 4)} 层注入一个
{mt.get('mt_proto', 13)} protofilaments × {mt.get('mt_scales', 5)} 时间尺度的
微管共振模块，基座权重完全冻结。{peft_line}

## 加载

```python
from mt_lnn.recipes import load_mt_adapter_dir
model, info = load_mt_adapter_dir("{base}", "./this-dir")
```

## 内容

| 文件 | 说明 |
| --- | --- |
| `{ADAPTER_CONFIG_NAME}` | 基座引用 + 图重建规格（键名对齐 `serve/server_hf.py`） |
| `{MT_WEIGHTS_NAME}` | MT adapter 权重（不含基座） |
| `{PEFT_SUBDIR}/` | LoRA 部分（PEFT 原生格式，仅当训练时启用） |

## 边界

* 需要 `mt_lnn` 在 Python 路径中：MT adapter 是自定义模块，PEFT 无法独立加载它。
* 基座权重不在本目录，加载时按 `base_model_name_or_path` 单独下载。
* 未做量化导出；服务端部署路线见仓库 `docs/SERVING_ROADMAP.md`。
"""


def _mt_args_from(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """训练 args -> 图重建规格：只留 :data:`MT_GRAPH_FIELDS`，缺的用默认值补。"""
    src = dict(args or {})
    out: Dict[str, Any] = {}
    for key in MT_GRAPH_FIELDS:
        out[key] = src.get(key, MT_GRAPH_DEFAULTS[key])
    return out


def _lora_args_from(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """训练 args -> PEFT 原生 ``LoraConfig`` 字段（键名是 PEFT 的，不是我们的）。"""
    src = dict(args or {})
    targets = src.get("lora_targets", "q_proj,k_proj,v_proj,o_proj")
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(src.get("lora_r", 8)),
        "lora_alpha": int(src.get("lora_alpha", 16)),
        "lora_dropout": float(src.get("lora_dropout", 0.05)),
        "bias": "none",
        "target_modules": list(targets),
    }


def _write_peft_subdir(peft_dir: str, config: Dict[str, Any],
                       lora_state: Dict[str, torch.Tensor]) -> None:
    """写 PEFT 原生子目录：``PeftModel.from_pretrained(base, dir)`` 可直接吃。"""
    os.makedirs(peft_dir, exist_ok=True)
    peft_cfg = dict(config.get("lora") or {})
    peft_cfg["base_model_name_or_path"] = config["base_model_name_or_path"]
    _write_json(os.path.join(peft_dir, ADAPTER_CONFIG_NAME), peft_cfg)
    torch.save(lora_state, os.path.join(peft_dir, "adapter_model.bin"))


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
