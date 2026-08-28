"""experiment_protocol.py — E0 实验协议固化（DEVELOPMENT_PLAN 第一天项）

背景：项目两次被"单 seed 结论"咬过 —— GQA 配额 g0 单 seed 1.0000，本地复现
seeds 1,2 = 0.183/0.165（grokking 掷硬币双峰区）；text_selective v4 的 null
又因配方没训起来而无效。DEVELOPMENT_PLAN E0 要求把这些纪律**代码化**：
从今以后不再产出单 seed 结论。

三条硬纪律（全部零依赖, 纯 python + math）：

1. ``require_min_seeds(per_seed_values, min_seeds=3)``
   不足 N seeds → ProtocolViolation（引用前必须过这关）。
2. ``bimodality_1d(values)`` — grokking 双峰检测：1-D 二分聚类，
   报告 {bimodal, split_at, gap_ratio}。双峰数据**不得引用**
   （HANDOFF §2.5 第 5 条的协议）。
3. ``paired_sign_test(a, b)`` / ``fisher_exact_2x2(table)`` — 配对检验，
   E1 的 p=0.0022 就是这个口径（不依赖 scipy，跨环境可复现）。

``publishable(per_seed_a, per_seed_b)`` — 引用门槛一键判定：
≥3 seeds、非双峰、配对检验 p<0.05，返回 (ok, report)。

排版约定：公共 API 在前、私有 helper 在后，各自按逻辑行数降序（大函数在上、
小函数在下）。

用法::

    from benchmarks.experiment_protocol import publishable
    ok, report = publishable(sel_accs, stock_accs, tag="E1-d16")
    # ok=False 时 report["reasons"] 说明差哪关 —— 结果只能入档，不能引用
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


class ProtocolViolation(RuntimeError):
    """违反 E0 协议（seeds 不足 / 双峰区数据 / 引用门槛未过）。"""


def fisher_exact_2x2(table: Sequence[Sequence[int]]) -> Dict:
    """2×2 Fisher 精确检验（双侧, hypergeometric 尾概率和法）。

    E1 的 grok 率对比 (selective 5/6 vs stock 0/6) 用这个口径。
    """
    ((a, b), (c, d)) = table
    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) == 0:
        return {"p": 1.0, "odds_ratio": float("nan")}

    def p_exact(x):
        return (math.comb(a + c, x) * math.comb(b + d, a + b - x)
                / math.comb(n, a + b))

    p_obs = p_exact(a)
    # 双侧: 所有 ≤ 观察概率的表求和
    lo = max(0, (a + b) - (b + d))
    hi = min(a + b, a + c)
    p = sum(p_exact(x) for x in range(lo, hi + 1) if p_exact(x) <= p_obs + 1e-15)
    or_ = (a * d) / (b * c) if b and c else float("inf")
    return {"p": min(1.0, p), "odds_ratio": or_}


def paired_sign_test(a: Sequence[float], b: Sequence[float]) -> Dict:
    """配对符号检验（精确二项, 双侧）。

    H0: P(a_i > b_i) = 0.5。并列对丢弃。无 scipy 依赖 —— 跨 Windows/
    Kaggle/AutoDL 环境结果一致。
    """
    if len(a) != len(b):
        raise ValueError("paired_sign_test: a/b 长度不一致")
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    n = wins + losses
    if n == 0:
        return {"p": 1.0, "wins": 0, "losses": 0, "n_effective": 0}
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return {"p": min(1.0, p), "wins": wins, "losses": losses,
            "n_effective": n}


def require_min_seeds(values: Sequence[float], min_seeds: int = 3,
                      what: str = "result") -> List[float]:
    vals = [float(v) for v in values]
    if len(vals) < min_seeds:
        raise ProtocolViolation(
            f"{what}: {len(vals)} seed(s) < required {min_seeds} — "
            "E0 协议禁止引用单/双 seed 结论 (HANDOFF §2.5 第 5 条)")
    if any(v is None or math.isnan(v) for v in vals):
        raise ProtocolViolation(f"{what}: per-seed 值含 NaN/None")
    return vals


def bimodality_1d(values: Sequence[float]) -> Dict:
    """1-D 二分聚类双峰检测（简化 dip：最优二分 + 间隙比）。

    返回 {"bimodal": bool, "split_at": float|None, "gap_ratio": float,
          "groups": [[...], [...]]}。判定：类间间隙 / 类内散度 > 3.0
    （grokking 的 chance/满分两团中间是空带，间隙比极大；噪声则比值小）。
    阈值 3.0 为经验值 —— 宁可多报（bimodal=True 只是"不得引用"，
    不是"扔数据"）。
    """
    vals = sorted(float(v) for v in values)
    if len(vals) < 4:
        return _unimodal_result(vals)
    best = _best_split(vals)
    if best is None:                       # 全同值: 无间隙, 单峰
        return _unimodal_result(vals)
    ratio, i = best
    return {"bimodal": ratio > 3.0, "split_at": (vals[i - 1] + vals[i]) / 2,
            "gap_ratio": round(ratio, 3), "groups": [vals[:i], vals[i:]]}


def forbid_bimodal(values: Sequence[float], what: str = "result") -> Dict:
    b = bimodality_1d(values)
    if b["bimodal"]:
        raise ProtocolViolation(
            f"{what}: per-seed 分布双峰 (gap_ratio={b['gap_ratio']}, "
            f"split@{b['split_at']:.3f}) — grokking 掷硬币区, 数据不得引用; "
            "加 seeds/换课程任务后重跑")
    return b


def publishable(a: Sequence[float], b: Sequence[float],
                min_seeds: int = 3, alpha: float = 0.05,
                tag: str = "") -> tuple:
    """对照结果的"可引用"判定：≥min_seeds + 双臂非双峰 + 配对 p<alpha。

    返回 (ok, report)。ok=False 的结果**只能入档**（jsonl + ABLATIONS），
    不得写进 README/论文/对外材料。这是 E0 的全部执行语义。
    """
    report: Dict = {"tag": tag}
    reasons = _gate_reasons(a, b, min_seeds, alpha, tag, report)
    report["ok"] = not reasons
    report["reasons"] = reasons
    return (not reasons), report


def _best_split(vals: List[float]):
    """最优二分点：间隙/类内散度 最大的切分。返回 (ratio, idx)；全同值 None。"""
    best = None
    for i in range(1, len(vals)):
        gap = vals[i] - vals[i - 1]
        if gap <= 0:
            continue
        lo, hi = vals[:i], vals[i:]
        # 类内散度: 两组合并的 mean absolute deviation（尺度稳定）
        spread = sum(abs(v - _mean(g)) for g in (lo, hi) for v in g) / len(vals)
        ratio = gap / max(spread, 1e-12)
        if best is None or ratio > best[0]:
            best = (ratio, i)
    return best


def _gate_reasons(a, b, min_seeds, alpha, tag, report) -> List[str]:
    """三道门槛（seeds / 双峰 / 配对检验）的未过原因收集。"""
    reasons = []
    for name, vals in (("arm-a", a), ("arm-b", b)):
        try:
            require_min_seeds(vals, min_seeds, what=f"{tag} {name}")
        except ProtocolViolation as e:
            reasons.append(str(e))
    reasons += _bimodality_reasons(a, b, report)
    reasons += _sign_test_reasons(a, b, alpha, report)
    return reasons


def _bimodality_reasons(a, b, report) -> List[str]:
    reasons = []
    for name, vals in (("arm-a", a), ("arm-b", b)):
        if len(vals) < 4:
            continue
        bm = bimodality_1d(vals)
        report[f"bimodality_{name}"] = bm
        if bm["bimodal"]:
            reasons.append(f"{name} 双峰 (gap_ratio={bm['gap_ratio']})")
    return reasons


def _sign_test_reasons(a, b, alpha, report) -> List[str]:
    if len(a) != len(b) or len(a) < 1:
        return ["两臂 seeds 不齐, 无法配对"]
    st = paired_sign_test(a, b)
    report["sign_test"] = st
    if st["p"] >= alpha:
        return [f"配对符号检验 p={st['p']:.3f} ≥ {alpha}"]
    return []


def _unimodal_result(vals: List[float]) -> Dict:
    return {"bimodal": False, "split_at": None, "gap_ratio": 0.0,
            "groups": [vals, []]}


def _mean(xs):
    return sum(xs) / len(xs)
