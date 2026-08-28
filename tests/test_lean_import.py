"""tests/test_lean_import.py — `import mt_lnn` 默认不加载研究脚手架（P2-13）

README 承诺 "shipped configs run the lean core"：已撤回/已证惰性的研究模块
（意识度量、物理/声学/空间研究栈、生物可塑性等）经 PEP 562 惰性导出，
不在默认 import 链上 —— 旧用法 `from mt_lnn import X` 仍 100% 可用。
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# 已撤回/惰性的研究模块 —— 默认 import 后不得出现在 sys.modules。
# （phi_iit 除外：__init__ 刻意 try-import 它的 PYPHI_AVAILABLE 哨兵）
FORBIDDEN_EAGER = {
    "mt_lnn.anesthesia", "mt_lnn.phi_hat", "mt_lnn.phi_spectral",
    "mt_lnn.hamiltonian_head", "mt_lnn.world_model",
    "mt_lnn.imagination", "mt_lnn.active_inference",
    "mt_lnn.spatial", "mt_lnn.spatial_ops", "mt_lnn.spatial_reasoning",
    "mt_lnn.physics_ops", "mt_lnn.acoustic_ops",
    "mt_lnn.salience_events", "mt_lnn.failsafe", "mt_lnn.ingest_ops",
    "mt_lnn.slow_layer", "mt_lnn.pipeline", "mt_lnn.plasticity",
    "mt_lnn.astrocyte", "mt_lnn.neuromodulation", "mt_lnn.sleep_consolidation",
}


def test_import_mtlnn_stays_lean():
    """子进程干净解释器里 import mt_lnn，研究模块不得被加载。"""
    code = ("import sys, json; import mt_lnn; "
            "print(json.dumps(sorted(m for m in sys.modules "
            "if m.startswith('mt_lnn.'))))")
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, cwd=str(REPO_ROOT),
                       timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    loaded = set(__import__("json").loads(r.stdout))
    leaked = loaded & FORBIDDEN_EAGER
    assert not leaked, f"研究模块被默认 import 拖入: {sorted(leaked)}"


def test_lazy_attr_still_resolves_and_caches():
    import mt_lnn

    cls = mt_lnn.AnesthesiaController          # 触发惰性导入
    from mt_lnn.anesthesia import AnesthesiaController as direct
    assert cls is direct
    assert mt_lnn.AnesthesiaController is cls  # 解析后缓存


def test_phi_iit_sentinel_always_present():
    import mt_lnn

    # PYPHI_AVAILABLE 是稳定哨兵（pyphi 缺失时恒 False），必须立刻可读
    assert isinstance(mt_lnn.PENNYLANE_AVAILABLE, bool)
    assert isinstance(mt_lnn.PYPHI_AVAILABLE, bool)
