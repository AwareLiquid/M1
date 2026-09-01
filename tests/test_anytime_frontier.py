"""anytime_frontier.py 的不变量测试 (全 CPU, <60s)。

锁定:
  1. CoT 批标签语义 — 只监督链区, 链标签 == token 本身, 链长随 k/g 正确。
  2. FLOPs 横轴映射 — cot_mean_flops 的闭式期望与单调性 (粗粒度更省),
     g=∞ 退化为 account_tfm(N=0)。
  3. 预注册压制规则 — 平局算压制 (对潜空间从严), 严格更差不压制。
  4. resume 预算匹配 — steps 不匹配的旧行必须重跑。

Run:  python -m pytest tests/test_anytime_frontier.py -v
"""

import argparse
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, ".")

from benchmarks.anytime_frontier import (DIFFICULTY, PROMPT_T_COT,
                                         _cot_marks, _done, _row_path,
                                         cot_mean_flops, make_cot_batch)
from benchmarks.compute_accounting import PROBE, account_tfm
from benchmarks.reasoning_tasks import vocab_size


def test_cot_batch_labels_cover_only_chain():
    """标签 = 链区全部位置 (-100 其余); 链标签与 token 一致 (teacher-forced)。
    m 由实际序列长推断 (make_cot_batch 内部 k 是每批随机抽的)。"""
    rng = np.random.default_rng(0)
    for g in (1000, 4, 2, 1):
        ids, labels = make_cot_batch(g, 8, rng, "cpu")
        m = ids.shape[1] - 53
        assert 1 <= m <= len(_cot_marks(DIFFICULTY, g)), f"g={g}: m={m}"
        assert torch.equal(ids[:, -m:], labels[:, -m:]), \
            f"g={g}: 链区必须全监督"
        assert (labels[:, :-m] == -100).all(), f"g={g}: 链外必须 -100"
    # 变 k 语义: 同 g 下不同 draw 的链长随 k 变 (mix 课程)
    lens = {make_cot_batch(2, 4, np.random.default_rng(s), "cpu")[0].shape[1]
            for s in range(8)}
    assert len(lens) > 1, "mix 口径 k 必须变化 → 链长必须变化"


def test_cot_mean_flops_closed_form_and_monotone():
    sh = dict(PROBE, T=PROMPT_T_COT, vocab=vocab_size(16))
    # 闭式: g=∞ → 每题 m=1 → account_tfm(N=0)
    assert cot_mean_flops(sh, 1000) == account_tfm(sh, 0, "cot")["flops"]
    # g=1 → m(k)=k → E[m]=4.5, 解码步 = m → 逐 k 求和闭式
    expect = sum(account_tfm(sh, k - 1, "cot")["flops"]
                 for k in range(1, DIFFICULTY + 1)) / DIFFICULTY
    assert cot_mean_flops(sh, 1) == expect
    # 单调性: 粒度越细 (g 越小) token 越多 → FLOPs 越大
    seq = [cot_mean_flops(sh, g) for g in (1000, 4, 2, 1)]
    assert seq == sorted(seq) and len(set(seq)) == len(seq)


def test_dominance_rule_tie_is_dominated():
    from benchmarks.anytime_frontier import _dominated
    cot_cheap = {"arm": "cot", "mflops": 10.0, "acc_mean": 0.50}
    cot_far = {"arm": "cot", "mflops": 200.0, "acc_mean": 0.90}
    pts = [cot_cheap, cot_far]
    tie = {"arm": "latent", "mflops": 10.0, "acc_mean": 0.50}
    assert _dominated(tie, pts), "平局 (flops=, acc=) 算压制 — 对潜空间从严"
    better = {"arm": "latent", "mflops": 9.9, "acc_mean": 0.51}
    assert not _dominated(better, pts)
    worse_both = {"arm": "latent", "mflops": 150.0, "acc_mean": 0.55}
    assert not _dominated(worse_both, pts), "贵且差于所有 CoT 点不叫被压制"


def test_resume_requires_matching_steps(tmp_path):
    """低预算旧行 (steps=30 冒烟) 不得让正式跑跳过。"""
    args = argparse.Namespace(out_dir=str(tmp_path), steps=30000)
    path = _row_path(args, "latent", 0)
    import json
    with open(path, "w") as f:
        json.dump({"arm": "latent", "steps": 30, "acc_by_depth": {}}, f)
    assert not _done(args, path), "steps 不匹配必须重跑"
    with open(path, "w") as f:
        json.dump({"arm": "latent", "steps": 30000, "acc_by_depth": {}}, f)
    assert _done(args, path)


def test_cot_marks_degenerate():
    assert _cot_marks(1, 1) == [1]
    assert _cot_marks(4, 1000) == [4]          # g ≥ k → 直接作答
    assert _cot_marks(7, 3) == [3, 6, 7]       # 链尾必须落在 k
    with pytest.raises(ValueError):
        _cot_marks(0, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
