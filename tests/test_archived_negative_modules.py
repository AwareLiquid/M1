"""tests/test_archived_negative_modules.py — 归档负结果模块守卫。

背景: BENCHMARKS.md §O1 module switch-matrix(2026-07-05, 48M)实测五个可选
模块全 PPL-neutral(predictive coding 甚至趋势为负);PRODUCT_LINES.md
「Biological-prior policy」记录该结论。这些模块的代码路径保留供更大规模复测,
但默认必须 OFF 且显式开启时要发出 [archived-negative] 警告 —— 防止"负结果
模块被当活跃旋钮误用"(本项目死旋钮债务治理的一部分)。
"""

from __future__ import annotations

import warnings

import pytest

from mt_lnn.config import _ARCHIVED_NEGATIVE_MODULES, MTLNNConfig

ARCHIVED_FIELDS = tuple(_ARCHIVED_NEGATIVE_MODULES.keys())


def test_default_config_no_archive_warnings() -> None:
    """默认配置(所有归档字段 False)不得触发任何 [archived-negative] 警告。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 默认配置触发归档警告 = 测试失败
        MTLNNConfig()


@pytest.mark.parametrize("field", ARCHIVED_FIELDS)
def test_enabling_archived_module_warns(field: str) -> None:
    """显式开启任一归档模块必须发出 [archived-negative] 警告。"""
    with pytest.warns(UserWarning, match=r"\[archived-negative\]") as record:
        MTLNNConfig(**{field: True})
    # 断言消息包含字段名 + 证据锚(48M O1 判负)
    archived = [r for r in record if "archived-negative" in str(r.message)]
    assert len(archived) == 1, f"{field}=True 应产生 1 条 archived-negative 警告"
    assert field in str(archived[0].message)
    assert "48M" in str(archived[0].message)


def test_archive_registry_matches_switch_matrix() -> None:
    """归档清单必须恰好覆盖 BENCHMARKS.md §O1 判负的五个模块(含 Hebbian refactor)。

    若开关矩阵复测推翻了某个判负(更大规模出现正效应),先更新 BENCHMARKS.md
    结论再从这里移除 —— 归档与证伪结论必须同源,不允许单边漂移。
    """
    expected = {
        "use_predictive_coding",
        "use_competitive_gwtb",
        "use_world_model",
        "use_rhythm",
        "use_hebbian",
        "use_hebbian_refactor",
    }
    assert set(ARCHIVED_FIELDS) == expected


def test_archived_fields_default_to_false() -> None:
    """归档模块的默认值必须是 False —— 默认配置跑的是核心主干。"""
    default = MTLNNConfig()
    for field in ARCHIVED_FIELDS:
        assert getattr(default, field) is False, f"{field} 默认应为 False"
