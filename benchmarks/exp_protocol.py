"""MT-LNN 实验协议统计模块 (E0)。

固化"诚实实验协议"，从今以后禁止单种子结论：
  - 好配方默认（beta2=0.999 / clip=0）——beta2=0.95 和 clip=1.0 各自独立地
    阻止 parity 类 grokking（2026-08-05 bisect），历史默认值保留仅为可比性。
  - 强制多种子：任何结论引用前 >= 3 seeds。
  - 配对检验：sel vs stock 的 grok 率用 Fisher 精确检验，逐种子 delta 用符号检验。
  - grokking 双峰检测：固定难度探针在 grok 掷硬币双峰区时，单 seed 结论不可信。

数据来源：benchmarks/results/reasoning_depth.jsonl 每行一个实验记录。
acc 字段可能是 float（固定难度）或嵌套 dict（per-k 评估，如
{"1": {"1": 1.0, "2": 1.0, "4": 1.0}}）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


# ── 单检验原语 ───────────────────────────────────────────────────────────────

def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """2x2 列联表 Fisher 精确检验（双尾）。

    表布局：
        [a  b]   上排 = 实验组（sel）
        [c  d]   下排 = 对照组（stock）
        a = sel 且 grok，b = sel 且非 grok
        c = stock 且 grok，d = stock 且非 grok
    返回双尾 p 值。p < 0.05 表示两臂 grok 率差异显著。
    """
    oddsr, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(p)


def paired_sign_test(deltas: list[float]) -> tuple[float, int, int]:
    """配对符号检验（零假设：逐种子 delta = 0 的中位数为 0）。

    剔除零差值后，正/负个数用二项分布双侧检验。
    返回 (p, n_pos, n_neg)。
    """
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n = n_pos + n_neg
    if n == 0:
        return (1.0, 0, 0)
    p = stats.binomtest(n_pos, n, 0.5).pvalue
    return (float(p), n_pos, n_neg)


def grok_rate(accs: list[float], threshold: float = 0.9) -> float:
    """acc >= threshold 的种子占比（grok 率）。"""
    if not accs:
        return 0.0
    return sum(1 for a in accs if a >= threshold) / len(accs)


# ── grokking 双峰检测 ────────────────────────────────────────────────────────

def bimodality_check(accs: list[float], chance: float = 0.5,
                     high: float = 0.9) -> dict[str, Any]:
    """检测 acc 分布是否落在 grokking 双峰区。

    grokking 的典型现象：acc 要么卡在 chance（~0.5），要么跃升到 ~1.0，
    两个峰并存 = "掷硬币双峰区"——有的种子 grok 了、有的没有，此时任何单
    seed 结论都不可信（可能是种子运气而非机制差异）。

    判据：chance 峰和 grok 峰同时非空，且中间灰区不占主导
    （mid_frac <= 0.3，否则是"训练中途"而非 grokking 双峰）。
    """
    if not accs:
        return {"bimodal": False, "verdict": "empty", "mid_frac": 0.0}
    n = len(accs)
    n_chance = sum(1 for a in accs if abs(a - chance) <= 0.05)
    n_grok = sum(1 for a in accs if a >= high)
    n_mid = n - n_chance - n_grok
    mid_frac = n_mid / n
    bimodal = (n_grok > 0 and n_chance > 0 and mid_frac <= 0.3)
    return {
        "bimodal": bimodal,
        "verdict": ("BIMODAL: single-seed unreliable" if bimodal else "clean"),
        "n_chance": n_chance,
        "n_grok": n_grok,
        "n_mid": n_mid,
        "n_chance_frac": n_chance / n,
        "n_grok_frac": n_grok / n,
        "mid_frac": mid_frac,
    }


# ── 汇总报告 ────────────────────────────────────────────────────────────────

def protocol_report(sel_accs: list[float], stock_accs: list[float],
                    chance: float = 0.5,
                    grok_threshold: float = 0.9) -> dict[str, Any]:
    """sel vs stock 两臂分离裁决（决策门 G1）。

    输入：两臂各自的逐种子 acc 列表（长度应相等且种子对齐）。
    裁决规则：
      - fisher_p < 0.05 且两臂均非双峰 → "SEPARATED"，decision_gate_passed=True
      - 任一臂双峰 → "BIMODAL_ZONE_UNRELIABLE"（不可信，需更多种子/预算）
      - 否则 → "NOT_SEPARATED"
    """
    sel_grok = int(sum(1 for a in sel_accs if a >= grok_threshold))
    stock_grok = int(sum(1 for a in stock_accs if a >= grok_threshold))
    n_sel = len(sel_accs)
    n_stock = len(stock_accs)
    fisher_p = fisher_exact_2x2(
        sel_grok, n_sel - sel_grok, stock_grok, n_stock - stock_grok)

    deltas = [s - t for s, t in zip(sel_accs, stock_accs)]
    sign_p, n_pos, n_neg = paired_sign_test(deltas)

    sel_bi = bimodality_check(sel_accs, chance, grok_threshold)
    stock_bi = bimodality_check(stock_accs, chance, grok_threshold)

    bimodal = sel_bi["bimodal"] or stock_bi["bimodal"]
    if bimodal:
        verdict = "BIMODAL_ZONE_UNRELIABLE"
    elif fisher_p < 0.05:
        verdict = "SEPARATED"
    else:
        verdict = "NOT_SEPARATED"

    return {
        "n_sel": n_sel,
        "n_stock": n_stock,
        "sel_grok_rate": grok_rate(sel_accs, grok_threshold),
        "stock_grok_rate": grok_rate(stock_accs, grok_threshold),
        "sel_grok": sel_grok,
        "stock_grok": stock_grok,
        "fisher_p": fisher_p,
        "sign_test_p": sign_p,
        "sign_test_pos": n_pos,
        "sign_test_neg": n_neg,
        "sel_bimodality": sel_bi,
        "stock_bimodality": stock_bi,
        "verdict": verdict,
        "decision_gate_passed": (fisher_p < 0.05 and not bimodal),
    }


# ── jsonl 数据提取 ───────────────────────────────────────────────────────────

def load_jsonl_rows(path: str) -> list[dict]:
    """读 jsonl 每行转 dict（跳过空行）。"""
    p = Path(path)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _flatten_acc(val: Any, depth_key: str) -> float:
    """把 jsonl 里的 acc 值规约为单标量。

    兼容三种形态：float / {"1": float} / {"1": {"1": float, "2": float, ...}}
    嵌套 dict 时对 depth_key 对应的所有叶子取均值。
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        if depth_key in val:
            inner = val[depth_key]
            if isinstance(inner, dict):
                vals = [float(v) for v in inner.values() if isinstance(v, (int, float))]
                if not vals:
                    return float("nan")
                return float(np.mean(vals))
            return float(inner)
        # depth_key 缺失：取全部叶子均值
        all_vals: list[float] = []
        for v in val.values():
            if isinstance(v, dict):
                all_vals.extend(float(x) for x in v.values()
                                if isinstance(x, (int, float)))
            elif isinstance(v, (int, float)):
                all_vals.append(float(v))
        if not all_vals:
            return float("nan")
        return float(np.mean(all_vals))
    return float("nan")


def aggregate_by_tag(rows: list[dict], tag_sel: str, tag_stock: str,
                     acc_field: str = "mtlnn_acc_by_depth",
                     depth_key: str = "1") -> tuple[list[float], list[float]]:
    """从 rows 提取两个 tag 的 acc 列表（按 seed 排序对齐）。

    返回 (sel_accs, stock_accs)，两者长度相等且种子对齐。
    """
    def extract(tag: str) -> list[float]:
        sel_rows = sorted([r for r in rows if r.get("tag") == tag],
                          key=lambda r: r.get("seed", 0))
        return [_flatten_acc(r.get(acc_field, {}), depth_key) for r in sel_rows]

    sel = extract(tag_sel)
    stock = extract(tag_stock)
    # 对齐到共同种子（以 stock 顺序为准，sel 补 NaN）
    n = min(len(sel), len(stock))
    return sel[:n], stock[:n]
