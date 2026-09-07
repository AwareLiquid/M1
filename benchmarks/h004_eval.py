#!/usr/bin/env python3
"""benchmarks/h004_eval.py — H004 评测与记账脚手架（机制轨，零主张）

Bet A 的评测基建中可以在 GPU 之前写好并测试的部分：

- CostLedger      成本记账（compute_accounting 口径：phase/gpu_hours/单价 → 合计），
                  H004 判据 3 的账本载体；
- cse_weights     采样偏置重要性加权（CSE, arXiv:2608.17293）——修正"标称上下文
                  被采样分布偏置"的行业问题，评测口径的严谨性差异化；
- build_replay    跨会话持久记忆流式回放任务生成器（确定性可种子化，模型无关）：
                  会话 A 写事实 → 填充干扰 → 会话 B 探针召回。这是 S3 王牌任务族
                  的流式版骨架，也是 benchmark 子贡献的任务源；
- write_prereg    预注册判据落档（pre_registered_rule 字段 + git 哈希 + 时间戳，
                  运行前写死——kb/RESULTS 三件套纪律的前端）。

零主张：本文件不携带任何实验数字；判据的具体阈值在开题日写死并回显 JSON。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import subprocess
from dataclasses import dataclass, field

SCHEMA_LEDGER = "h004_cost_ledger/1"
SCHEMA_PREREG = "h004_prereg/1"


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ── 成本账本 ────────────────────────────────────────────────────────────────

@dataclass
class CostLedger:
    """训练/推理成本记账。金额单位人民币；gpu_hours × unit_price_cny 逐条累加。"""

    entries: list[dict] = field(default_factory=list)

    def add(self, phase: str, gpu_hours: float, unit_price_cny: float, note: str = "") -> dict:
        if gpu_hours < 0 or unit_price_cny < 0:
            raise ValueError("成本项不允许负数（记账诚实化）")
        e = {"phase": phase, "gpu_hours": round(gpu_hours, 4),
             "unit_price_cny": unit_price_cny,
             "cny": round(gpu_hours * unit_price_cny, 2), "note": note}
        self.entries.append(e)
        return e

    def total(self) -> dict:
        return {
            "gpu_hours": round(sum(e["gpu_hours"] for e in self.entries), 4),
            "cny": round(sum(e["cny"] for e in self.entries), 2),
            "n_entries": len(self.entries),
        }

    def to_report(self) -> dict:
        return {"schema": SCHEMA_LEDGER, "entries": list(self.entries),
                "total": self.total(), "git_hash": git_hash(), "timestamp_utc": now_iso()}

    def dump(self, path: str) -> dict:
        report = self.to_report()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report


# ── CSE 采样偏置修正 ────────────────────────────────────────────────────────

def cse_weights(freqs: dict[str, float]) -> dict[str, float]:
    """重要性加权：w_i ∝ 1/f_i（归一化到和为 1）。

    freqs = 各评测桶的采样频率（如按上下文长度分桶的出现概率）。
    均匀采样 → 均匀权重；过采样桶被降权。零频/缺失桶不参与。
    重建"真实分布下的期望表现"，防"标称 1M 只在易桶上测"式虚标。
    """
    positive = {k: f for k, f in freqs.items() if f > 0}
    if not positive:
        raise ValueError("cse_weights: 至少需要一个正采样频率")
    raw = {k: 1.0 / f for k, f in positive.items()}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


# ── 跨会话流式回放任务生成器 ────────────────────────────────────────────────

CODES = ("alpha", "bravo", "delta", "echo", "kilo", "lima", "nova", "oscar",
         "papa", "quebec", "romeo", "tango", "uniform", "victor", "whiskey",
         "xray", "yankee", "zulu")


def build_replay(seed: int, *, n_sessions: int = 3, facts_per_session: int = 5,
                 fillers_per_session: int = 20, n_probes: int = 6) -> dict:
    """生成跨会话持久记忆回放任务（确定性：同 seed 同任务）。

    结构：每个会话写入 facts_per_session 条事实（值含唯一 code 词）+
    fillers 干扰流；探针在**后续会话**中要求召回**更早会话**的事实——
    "跨会话"性质由构造保证（probe.session > fact.session）。
    模型无关：产出纯文本任务，评测 runner 挂任意模型。
    """
    rng = random.Random(seed)
    if n_sessions < 2:
        raise ValueError("跨会话任务至少需要 2 个会话")
    n_facts = facts_per_session * n_sessions
    # 探针按 code 词查询，因此 code 必须全局唯一：池不够时用序号后缀拼接
    codes = [CODES[i] if i < len(CODES) else f"{CODES[i % len(CODES)]}-{i // len(CODES) + 1}"
             for i in range(n_facts)]

    sessions, fact_index, ci = [], {}, 0
    for sid in range(1, n_sessions + 1):
        facts = []
        for _ in range(facts_per_session):
            code = codes[ci]
            key, value = f"fact::s{sid}::{ci}", f"项目 {code} 的登记码是 {ci:04d}"
            facts.append({"key": key, "value": value})
            fact_index[key] = {"value": value, "session": sid, "code": code}
            ci += 1
        fillers = [f"例行日志 {sid}-{j}: {rng.choice(CODES)} 常规巡检无异常"
                   for j in range(fillers_per_session)]
        sessions.append({"session_id": sid, "facts": facts, "fillers": fillers})

    # 探针：从"已有后续会话"的事实里取，保证跨会话
    eligible = [k for k, v in fact_index.items() if v["session"] < n_sessions]
    rng.shuffle(eligible)
    probes = []
    for k in eligible[:n_probes]:
        meta = fact_index[k]
        probes.append({"query": f"项目 {meta['code']} 的登记码是多少？",
                       "expected_key": k,
                       "expected_value": meta["value"],
                       "asked_in_session": n_sessions,
                       "fact_from_session": meta["session"]})

    return {
        "schema": "h004_replay_task/1",
        "seed": seed,
        "n_sessions": n_sessions,
        "n_probes": len(probes),
        "sessions": sessions,
        "probes": probes,
        "generator_git_hash": git_hash(),
        "timestamp_utc": now_iso(),
    }


def check_replay_integrity(task: dict) -> dict:
    """任务自检：每个探针的答案必须恰好出现在其事实来源中（ground-truth 完整性）。"""
    values_by_key = {}
    for s in task["sessions"]:
        for f in s["facts"]:
            values_by_key[f["key"]] = f["value"]
    bad = [p["expected_key"] for p in task["probes"]
           if values_by_key.get(p["expected_key"]) != p["expected_value"]]
    cross = all(p["asked_in_session"] > p["fact_from_session"] for p in task["probes"])
    return {"probes_total": len(task["probes"]),
            "ground_truth_ok": not bad, "bad_probes": bad,
            "all_probes_cross_session": cross}


# ── 预注册判据落档 ──────────────────────────────────────────────────────────

def write_prereg(path: str, rule: dict) -> dict:
    """把预注册判据写死落档（运行前调用；落档后判据不得移动——kb 纪律）。"""
    doc = {"schema": SCHEMA_PREREG, "pre_registered_rule": rule,
           "git_hash": git_hash(), "timestamp_utc": now_iso()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="H004 评测与记账脚手架（零主张）")
    p.add_argument("--replay", action="store_true", help="生成示例回放任务并自检")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="benchmarks/results/h004_replay_sample.json")
    args = p.parse_args(argv)

    if args.replay:
        task = build_replay(args.seed)
        integrity = check_replay_integrity(task)
        task["integrity"] = integrity
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(task, f, ensure_ascii=False, indent=2)
        print(json.dumps(integrity, ensure_ascii=False))
        return 0 if integrity["ground_truth_ok"] and integrity["all_probes_cross_session"] else 1
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
