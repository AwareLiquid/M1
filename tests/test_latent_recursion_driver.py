"""latent_recursion.py 驱动的不变量测试 (全 CPU, <30s)。

锁定三类易回归点:
  1. resume 语义 — 有效行跳过、损坏/缺失行重跑 (断点续跑的正确性)。
  2. 原子写 — 落盘行可完整解析, 无 .tmp 残留。
  3. 预注册判决数学 — H 成立 / 平坦 Null / 双峰门槛 三条路径,
     含"3 seeds 时配对符号检验 p 最小 0.25、必不过 0.05"的协议事实
     (这就是默认 6 seeds 的原因, 防止有人改回 3 seeds 还指望可引用)。

Run:  python -m pytest tests/test_latent_recursion_driver.py -v
"""

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

from benchmarks.latent_recursion import (DEPTHS, MODES, judge_decision,
                                         pending_configs, save_row)


def _args(tmp, steps=100):
    return argparse.Namespace(out_dir=str(tmp), seeds=[0, 1, 2, 3, 4, 5],
                              modes=MODES, steps=steps)


def _row(mode, depth, seed, acc):
    return {"mode": mode, "depth": depth, "seed": seed, "mean_acc": acc,
            "acc_by_k": {"8": acc}, "steps": 100, "protocol": "test"}


def _rows_for(tier_accs, mode="stack", seeds=6):
    """tier_accs: {depth: acc 或 [per-seed accs]} → 行列表。"""
    rows = []
    for depth, acc in tier_accs.items():
        vals = acc if isinstance(acc, list) else [acc] * seeds
        rows += [_row(mode, depth, s, v) for s, v in enumerate(vals)]
    return rows


def test_resume_skips_valid_and_retries_corrupt(tmp_path):
    args = _args(tmp_path)
    path = os.path.join(str(tmp_path),
                        "latent_recursion_stack_d1_s0.json")
    save_row(path, _row("stack", 1, 0, 0.5))
    todo = pending_configs(args)
    assert ("stack", 1, 0) not in todo
    assert ("stack", 1, 1) in todo and ("core", 8, 5) in todo
    with open(path + ".tmp", "w") as f:      # 模拟中断留下的坏行
        f.write('{"mode": "stack", "depth": 1, "seed": 0, "mean_a')
    os.replace(path + ".tmp", path)          # 覆盖成损坏文件
    assert ("stack", 1, 0) in pending_configs(args)


def test_resume_requires_matching_budget(tmp_path):
    """低预算行 (Stage-1 验证 1000 步) 不得让正式 30k 跑跳过该配置。"""
    args = _args(tmp_path)
    args.steps = 30000
    stale = dict(_row("stack", 1, 0, 0.5), steps=1000)
    save_row(os.path.join(str(tmp_path),
                          "latent_recursion_stack_d1_s0.json"), stale)
    assert ("stack", 1, 0) in pending_configs(args), \
        "steps 不匹配的旧行必须重跑"
    fresh = dict(_row("core", 2, 1, 0.5), steps=30000)
    save_row(os.path.join(str(tmp_path),
                          "latent_recursion_core_d2_s1.json"), fresh)
    assert ("core", 2, 1) not in pending_configs(args)


def test_save_row_atomic_roundtrip(tmp_path):
    path = os.path.join(str(tmp_path), "row.json")
    save_row(path, _row("core", 4, 2, 0.75))
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["mean_acc"] == 0.75
    assert not os.path.exists(path + ".tmp")


def test_judge_h_supported_requires_six_seeds(tmp_path):
    """单调 + 增益≥2σ + 6 seeds 全胜 → H 成立; 同数据 3 seeds 必判负。"""
    tiers = {1: 0.20, 2: 0.40, 4: 0.65, 8: 0.90}
    v6 = judge_decision(_rows_for(tiers), DEPTHS, MODES)
    assert v6["h_supported"] is True
    assert v6["h_components"]["monotonic"] is True
    assert v6["diagnosis"]["kind"] == "H_supported"
    v3 = judge_decision(_rows_for(tiers, seeds=3), DEPTHS, MODES)
    assert v3["h_supported"] is False, "3 seeds 时符号检验 p>=0.25, 不可引用"


def test_judge_null_flat_iterations_ignored():
    tiers = {1: 0.50, 2: 0.51, 4: 0.49, 8: 0.50}
    v = judge_decision(_rows_for(tiers), DEPTHS, MODES)
    assert v["h_supported"] is False
    assert v["diagnosis"]["kind"] == "flat_iterations_ignored"
    assert v["gate"]["ok"] is False


def test_judge_bimodal_grokking_gate():
    """深度 8 半数种子 grok 半数随机 → 双峰区, 判负且不可引用。"""
    tiers = {1: 0.5, 2: 0.5, 4: 0.5, 8: [1.0, 1.0, 1.0, 0.06, 0.06, 0.06]}
    v = judge_decision(_rows_for(tiers), DEPTHS, MODES)
    assert v["h_supported"] is False
    assert v["diagnosis"]["kind"] == "bimodal_grokking_zone"
    assert any("双峰" in r for r in v["gate"]["reasons"])


def test_parse_args_namespace_complete():
    """main/run_config 用到的每个 args 属性都必须由 _parse_args 提供
    (回归: args.modes AttributeError 曾让启动即崩)。"""
    import benchmarks.latent_recursion as lr
    args = lr._parse_args(["--steps", "5"])
    for attr in ("steps", "batch", "lr", "difficulty", "n_values", "seeds",
                 "device", "out_dir", "probe", "probe_steps", "log_every"):
        assert hasattr(args, attr), f"missing args.{attr}"


def test_judge_core_only_runs_do_not_crash():
    """core 单模式收尾不得 KeyError: 'stack' — 2026-08-31 A100 事故回归点。

    core 是阴性对照臂, 单独运行时无裁决对象: 判决如实降级为
    no_stack_arms 非判决账本 (行 JSON 本就逐配置原子写, 不受影响)。
    服务器残留 verdict (n_rows=1) 即此 bug 的现场。
    """
    rows = _rows_for({1: 0.30, 2: 0.35, 4: 0.40, 8: 0.45}, mode="core")
    v = judge_decision(rows, DEPTHS, ("core",))
    assert v["h_supported"] is False
    assert v["diagnosis"]["kind"] == "no_stack_arms"
    assert v["h_components"]["monotonic"] is None
    assert v["gate"]["ok"] is False
    assert v["n_rows"] == 24


def test_judge_stack_only_still_adjudicates():
    """stack 单模式(服务器上 stack 进程的实际形态)保持可裁决 — 冻结路径不回归。"""
    rows = _rows_for({1: 0.20, 2: 0.40, 4: 0.65, 8: 0.90}, mode="stack")
    v = judge_decision(rows, DEPTHS, ("stack",))
    assert v["h_supported"] is True
    assert v["diagnosis"]["kind"] == "H_supported"
