"""mt_lnn/quantization.py — weight-only int8 量化（液态裸 Parameter + einsum 路径）

背景（scripts/quant_smoke.py 2026-08-28 打样）
---------------------------------------------
``torch.ao.quantization.quantize_dynamic`` 只覆盖 ``nn.Linear``（注意力投影），
而液态核心的权重是裸 ``nn.Parameter`` 直接进 ``einsum`` —— 打样实测液态参数
全部落在量化覆盖之外，2B「fp32 7.6 GB → 全 int8 1.9 GB 进 7.2GB VPS」的部署
数学缺最后一环。本模块补上这一环。

设计
----
``Int8Weight`` 是 ``torch.Tensor`` 子类：**int8 字节做持久存储**（+ 每张量一个
fp32 scale），任何算子（einsum / matmul / add …）触到它时在
``__torch_dispatch__`` 里惰性反量化成 fp32 再参与计算。因此：

* 液态层 forward 代码**零改动** —— ``self.W_in`` 依然是"一个能进 einsum 的
  张量"；
* 常驻内存是 int8（1 byte/elem）；fp32 形态只在算子内部瞬态存在，
  用完即释放 —— 逐层瞬态，峰值 ≈ int8 总量 + 单层 fp32；
* ``model.parameters()`` 不再包含被量化权重（量化模型只用于推理），int8 载荷
  与 scale 以 buffer 形式保留，``state_dict`` 照常可存取。

**刻意不量化的路径**：tau / decay / λ / blend / sel_b / log_tau 等动力学小参数
保持 fp32（2026-08-28 的教训：softplus/exp 衰减链对数值精度敏感，bf16 都会让
训练 PPL 恶化 3–7.5×，int8 只会更糟）。本模块只碰"大矩阵乘权重"，名单硬编码
见 ``QUANTIZABLE_WEIGHT_NAMES``。

用法
----
    from mt_lnn.quantization import quantize_mtlnn_int8
    model = load_ckpt(...).to(device).eval()
    model, report = quantize_mtlnn_int8(model)   # 返回值可能是新模块树！
    model(input_ids=...)                          # 照常推理

``quantize_dynamic``（Linear 部分）会**重建模块树并返回新对象**，所以本函数
返回 ``(model, report)``，调用方必须使用返回的 model。顺序上必须先 dynamic-
Linear、后 Int8Weight（重建会丢已挂的普通属性句柄；函数内部已按此顺序）。
``load_state_dict`` 对 buffer 是原位 copy，句柄与 buffer 共享存储，加载后无需
重建；若手工换过 buffer 张量对象，调用 ``rebuild_int8_handles(model)``。
"""

from __future__ import annotations

import torch
from torch._C import _DisableTorchDispatch  # noqa: PLC2701 - 无公开等价物
import torch.nn as nn

# 液态核心的"大矩阵乘权重"名单（mt_lnn_layer.py + mt_lnn_v2.py）。
# 不在名单 = 永不量化（动力学参数、bias、门控小参数）。
QUANTIZABLE_WEIGHT_NAMES = frozenset({
    "W_in",        # (P,S,D,D) 输入转移
    "W_pred",      # (P,S-1,D,D) 预测编码
    "hh_w",        # (P,rank,D,D) Householder 非对角
    "dp_u_w", "dp_v_w",  # (P,rank,D,D) DeltaProduct
    "sel_w",       # (P,S,D) selective_decay 门
    "W_mix", "W_dt", "W_gate",  # mt_lnn_v2
    "fc1_weight", "fc2_weight",  # MAP gate MLP（小，但纯矩阵）
})

# 低于该元素数的权重不值得量化（scale 开销 > 收益）。
MIN_NUMEL = 4096


class Int8Weight(torch.Tensor):
    """int8 存储 + per-tensor scale 的惰性反量化权重张量。

    对外行为 = fp32 反量化值（任何算子经 ``__torch_dispatch__`` 反量化后执行）；
    对内存储 = int8（``element_size()==1``）。
    """

    scale: torch.Tensor   # fp32 标量张量（0-dim），与载荷同设备

    @staticmethod
    def from_float(w: torch.Tensor) -> "Int8Weight":
        w = w.detach().float()
        scale = (w.abs().max() / 127.0).clamp(min=1e-12)
        q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
        handle = q.as_subclass(Int8Weight)
        handle.scale = scale
        return handle

    def dequantized(self) -> torch.Tensor:
        """fp32 反量化（绕过 dispatch，避免自递归）。"""
        with _DisableTorchDispatch():
            return self.to(torch.float32) * self.scale

    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def tree(obj):
            if isinstance(obj, Int8Weight):
                return obj.dequantized()
            if isinstance(obj, (tuple, list)):
                return type(obj)(tree(o) for o in obj)
            if isinstance(obj, dict):
                return {k: tree(v) for k, v in obj.items()}
            return obj

        return func(*tree(args), **(tree(kwargs) if kwargs else {}))


def _quantize_linears_robust(model: nn.Module) -> tuple:
    """nn.Linear → 动态量化，逐模块容错。

    整树 ``quantize_dynamic`` 一把梭在某些引擎上会被个别形状卡死（实测
    qnnpack 不支持 1×1 Linear 的 ``linear_prepack``，FBGEMM 支持），所以
    整树失败时降级为逐模块替换、跳过引擎不支持的 —— 宁可留个别 fp32
    也不让整个量化路径失败。返回 (model, n_swapped, n_skipped)。
    """
    from torch.ao.quantization import quantize_dynamic
    try:
        new = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        n = sum(1 for m in new.modules()
                if isinstance(m, torch.nn.quantized.dynamic.Linear))
        return new, n, 0
    except RuntimeError:
        pass
    # 逐模块替换（swap 后的对象树不变名，只换叶子）
    n_swapped = n_skipped = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        try:
            q = quantize_dynamic(mod, {nn.Linear}, dtype=torch.qint8)
        except RuntimeError:
            n_skipped += 1
            continue
        # name 形如 "blocks.0.attn.q_proj"：挂回父模块
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, q)
        n_swapped += 1
    return model, n_swapped, n_skipped


def _module_weight_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def _linear_dyn_supported() -> bool:
    """本机能否跑 nn.Linear 动态量化（需要 QEngine：x86=FBGEMM/oneDNN，
    arm64 macOS/部分构建上没有 —— 报 ``quantized::linear_prepack NoQEngine``）。
    探测一次并缓存；不支持时优雅降级为"只量化液态裸 Parameter"。"""
    if getattr(_linear_dyn_supported, "_cached", None) is not None:
        return _linear_dyn_supported._cached
    try:
        from torch.ao.quantization import quantize_dynamic
        probe = quantize_dynamic(nn.Linear(8, 8), {nn.Linear},
                                 dtype=torch.qint8)
        with torch.no_grad():
            probe(torch.randn(1, 8))
        ok = True
    except Exception:  # noqa: BLE001 - 任何 engine 缺失都降级
        ok = False
    _linear_dyn_supported._cached = ok
    return ok


def quantize_mtlnn_int8(
    model: nn.Module,
    *,
    quantize_linear: bool = True,
    verbose: bool = False,
) -> tuple:
    """就地 weight-only int8 量化（液态裸 Parameter）+ 可选 nn.Linear 动态量化。

    必须在 ``.to(device)`` 与 ``.eval()`` 之后、开始推理之前调用。
    返回 ``(model, report)`` —— Linear 部分经 ``quantize_dynamic`` 重建模块树，
    返回的 model 可能是新对象，调用方必须改用它。
    """
    before_bytes = _module_weight_bytes(model)
    quantized_names = []

    # 1) nn.Linear → torch 动态量化（会重建模块树，必须先做）。
    #    QEngine 不可用（如 arm64 macOS）时降级为只量化液态裸 Parameter。
    linear_dyn = quantize_linear and _linear_dyn_supported()
    linear_skipped = 0
    if quantize_linear and not linear_dyn:
        print("[quant] WARNING: no QEngine for nn.Linear dynamic quantization "
              "on this build — Linear layers stay fp32, liquid raw Parameters "
              "are still quantized.")
    if linear_dyn:
        model, _, linear_skipped = _quantize_linears_robust(model)
        if linear_skipped:
            print(f"[quant] {linear_skipped} Linear layer(s) skipped by the "
                  "engine (unsupported shape, e.g. 1×1 on qnnpack) — left fp32.")

    # 2) 液态裸 Parameter → Int8Weight（就地替换，模块树不动）
    for module in model.modules():
        for name in list(getattr(module, "_parameters", {})):
            if name not in QUANTIZABLE_WEIGHT_NAMES:
                continue
            p = module._parameters[name]
            if p is None or p.numel() < MIN_NUMEL or p.dtype != torch.float32:
                continue
            w = p.detach()
            scale = (w.abs().max() / 127.0).clamp(min=1e-12)
            q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
            # Parameter 移除；int8 载荷与 scale 以 buffer 入 state_dict；
            # 同名普通属性挂 Int8Weight 句柄（与 buffer 共享存储）。
            del module._parameters[name]
            module.register_buffer(f"{name}_q", q)
            module.register_buffer(f"{name}_scale", scale)
            handle = q.as_subclass(Int8Weight)
            handle.scale = scale
            setattr(module, name, handle)
            quantized_names.append(f"{type(module).__name__}.{name}")

    after_bytes = _module_weight_bytes(model)
    report = {
        "weights_quantized": quantized_names,
        "n_quantized": len(quantized_names),
        "linear_dynamic": linear_dyn,
        "linear_skipped": linear_skipped,
        "bytes_before": before_bytes,
        "bytes_after": after_bytes,
        "reduction": 1.0 - after_bytes / max(before_bytes, 1),
    }
    if verbose:
        print(f"[quant] {report['n_quantized']} liquid weights → int8; "
              f"{before_bytes/1e6:.0f} MB → {after_bytes/1e6:.0f} MB "
              f"({report['reduction']:.0%})")
    return model, report


def rebuild_int8_handles(model: nn.Module) -> int:
    """按 buffer 重建 Int8Weight 句柄。

    ``load_state_dict`` 是原位 copy、句柄与 buffer 共享存储，通常无需调用；
    只有手工替换过 ``*_q`` buffer 张量对象时才需要。
    """
    n = 0
    for module in model.modules():
        for buf_name in list(getattr(module, "_buffers", {})):
            if not buf_name.endswith("_q"):
                continue
            base = buf_name[:-2]
            if base not in QUANTIZABLE_WEIGHT_NAMES:
                continue
            q = module._buffers[buf_name]
            scale = module._buffers.get(f"{base}_scale")
            if q is None or scale is None:
                continue
            handle = q.as_subclass(Int8Weight)
            handle.scale = scale
            setattr(module, base, handle)
            n += 1
    return n
