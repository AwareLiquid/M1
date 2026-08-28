"""tests/test_lnn_broadcast.py — broadcast 训练提速开关（P1-8）

依据（2026-08-28 消融, scripts/ppl_diagnosis.py, 128M checkpoint）:
  scan(真循环) PPL 153.92 vs broadcast 154.61 (比值 0.996) — 语言质量无差;
  broadcast fwd+bwd 实测 ~12% 提速。推理/流式路径永远 scan (O(1) 主张载体)。

契约:
  • 模型 forward 两种模式都跑得通且数值接近（随机初始化小模型）
  • train.py / scaling_comparison.py 的 --lnn_broadcast 默认 off（P0 历史口径）
  • 开关确实改变 layer 收到的 use_scan（否则是死 flag）
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _tiny():
    torch.manual_seed(0)
    cfg = MTLNNConfig(vocab_size=64, d_model=104, n_layers=2, n_heads=13,
                      n_kv_heads=1, d_head=8, max_seq_len=64, gwtb_n_heads=1)
    return MTLNNModel(cfg)


def test_broadcast_close_to_scan():
    m = _tiny().eval()
    ids = torch.randint(0, 64, (2, 48))
    with torch.no_grad():
        a = m(ids, use_lnn_recurrence=True)["logits"]
        b = m(ids, use_lnn_recurrence=False)["logits"]
    # 128M 训练后消融比值 0.996 (PPL)；随机初始化小模型的 h 动力学会放大
    # 两种模式的分歧，这里只断言"方向一致"（cosine > 0.9），等价性主张
    # 以训练后模型的 PPL 消融为准（scripts/ppl_diagnosis.py）。
    cos = torch.nn.functional.cosine_similarity(
        a.flatten(0, 1), b.flatten(0, 1), dim=-1)
    assert float(cos.min()) > 0.9, f"min cosine {float(cos.min()):.4f}"


def test_flag_changes_use_scan_at_layer(monkeypatch):
    """死 flag 检测：翻 flag 必须真的翻 layer 的 use_scan。"""
    m = _tiny().eval()
    seen = []
    import mt_lnn.mt_lnn_layer as layer_mod
    orig = layer_mod.MTLNNLayer.forward

    def spy(self, *a, **kw):
        seen.append(kw.get("use_scan", "missing"))
        return orig(self, *a, **kw)

    monkeypatch.setattr(layer_mod.MTLNNLayer, "forward", spy)
    ids = torch.randint(0, 64, (1, 32))
    with torch.no_grad():
        m(ids, use_lnn_recurrence=True)
        m(ids, use_lnn_recurrence=False)
    assert seen and all(v is True for v in seen[:len(seen) // 2])
    assert all(v is False for v in seen[len(seen) // 2:])


def test_train_entrypoints_default_off():
    for path in (REPO / "train.py", REPO / "benchmarks" / "scaling_comparison.py"):
        src = path.read_text(encoding="utf-8")
        assert "--lnn_broadcast" in src, path
        assert "default=False" in src, f"{path}: 默认必须 off (P0 历史口径)"


def test_scaling_row_records_recurrence_mode():
    src = (REPO / "benchmarks" / "scaling_comparison.py").read_text(encoding="utf-8")
    assert '"lnn_recurrence": "scan"' in src
    assert "use_lnn_recurrence=not args.lnn_broadcast" in src
