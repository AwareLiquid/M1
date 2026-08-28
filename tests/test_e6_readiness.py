"""tests/test_e6_readiness.py — E6 文本收益重测的就绪度契约（P1-7）

DEVELOPMENT_PLAN E6 协议三要素，防止回退：
  1. 好配方可一键启用且行内记录（beta2=0.999 / clip=0）
  2. 默认配方保持 P0 历史口径（0.95 / 1.0）—— scaling_fp32/ 已归档结果
     不带这些 flag 复跑时必须可复现
  3. 2000 步 sanity 门存在（v4 的 PPL≈470 那种"没训起来"的跑法在 2000 步
     就该被拦下，不再烧掉剩余 6000 步 GPU）
"""

import argparse
from pathlib import Path

import pytest

pytest.importorskip("torch")

import benchmarks.scaling_comparison as sc  # noqa: E402


def _args(**kw):
    base = dict(beta2=0.95, grad_clip=1.0, good_recipe=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_default_recipe_is_p0_protocol():
    assert sc._resolve_recipe(_args()) == (0.95, 1.0)


def test_good_recipe_preset():
    assert sc._resolve_recipe(_args(good_recipe=True)) == (0.999, 0.0)


def test_good_recipe_overrides_explicit_flags():
    got = sc._resolve_recipe(_args(good_recipe=True, beta2=0.9, grad_clip=5.0))
    assert got == (0.999, 0.0)


def test_cli_exposes_recipe_and_sanity_flags():
    src = Path(sc.__file__).read_text(encoding="utf-8")
    for flag in ("--good_recipe", "--beta2", "--grad_clip",
                 "--sanity_steps", "--sanity_ppl"):
        assert flag in src, f"{flag} missing from CLI"
    # sanity 门阈值默认 = E6 协议的 150（v4 的 470 会在 2000 步被拦）
    assert "--sanity_ppl\", type=float, default=150.0" in src


def test_result_row_records_recipe():
    # 行内记录：row 必须带 beta2/grad_clip 字段（E0 纪律，防二手结论）
    src = Path(sc.__file__).read_text(encoding="utf-8")
    assert '"beta2": beta2, "grad_clip": grad_clip' in src
    assert 'row["sanity"] = {"steps": args.sanity_steps' in src
