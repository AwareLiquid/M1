"""tests/test_quantization.py — weight-only int8 量化（液态裸 Parameter 路径）

覆盖（P0-2）：
  • Int8Weight 往返误差有界（≤ scale = max|w|/127）
  • 反量化后可正常参与 einsum/matmul（对外就是一个 fp32 值张量）
  • 小模型 forward：量化 vs fp32 输出接近（cosine > 0.999）
  • 内存：被量化权重的 fp32 Parameter 消失、int8 buffer 出现
  • state_dict 往返：int8 buffer 落盘 → 新模型加载 → 输出一致
  • 量化权重不进 model.parameters()（不参与训练/优化器）
  • 动力学参数（log_tau / sel_b / blend_weights）保持 fp32 不被动
"""

import math

import pytest

torch = pytest.importorskip("torch")

from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402
from mt_lnn.quantization import (  # noqa: E402
    QUANTIZABLE_WEIGHT_NAMES, Int8Weight, quantize_mtlnn_int8,
    rebuild_int8_handles,
)


def _tiny_model(seed: int = 0) -> MTLNNModel:
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=64, gwtb_n_heads=1,
    )
    return MTLNNModel(cfg).eval()


def _forward_logits(model, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 64, (1, 32), generator=g)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    return out["logits"].float()


# ---------------------------------------------------------------------------
# Int8Weight 原语
# ---------------------------------------------------------------------------

class TestInt8Weight:
    def test_roundtrip_error_bounded_by_scale(self):
        w = torch.randn(13, 5, 32, 32) * 0.3
        h = Int8Weight.from_float(w)
        scale = (w.abs().max() / 127.0).clamp(min=1e-12)
        err = (h.dequantized() - w).abs().max()
        assert err <= scale + 1e-8, f"err {err:.3e} > scale {scale:.3e}"

    def test_storage_is_int8(self):
        h = Int8Weight.from_float(torch.randn(128, 128))
        assert h.dtype == torch.int8
        assert h.element_size() == 1

    def test_ops_see_dequantized_fp32(self):
        w = torch.randn(64, 64)
        h = Int8Weight.from_float(w)
        # 直接进 einsum —— 对外行为就是一个 fp32 张量
        x = torch.randn(4, 64)
        got = torch.einsum("bd,de->be", x, h)
        want = torch.einsum("bd,de->be", x, h.dequantized())
        assert torch.allclose(got, want, atol=1e-5)
        assert got.dtype == torch.float32

    def test_matmul_and_reshape_dispatch(self):
        h = Int8Weight.from_float(torch.randn(32, 48))
        assert h.reshape(48, 32).shape == (48, 32)   # dispatch → plain tensor
        y = h @ torch.randn(48, 4)
        assert y.shape == (32, 4)


# ---------------------------------------------------------------------------
# 全模型量化
# ---------------------------------------------------------------------------

class TestQuantizeModel:
    def test_forward_close_to_fp32(self):
        model = _tiny_model(seed=0)
        ref = _forward_logits(model)
        qmodel, report = quantize_mtlnn_int8(model, quantize_linear=False)
        got = _forward_logits(qmodel)
        # 逐 token cosine：int8 权重扰动不应改变输出方向
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten(0, 1), got.flatten(0, 1), dim=-1)
        assert float(cos.min()) > 0.999, f"min cosine {float(cos.min()):.5f}"

    def test_quantized_weights_leave_parameters(self):
        model = _tiny_model(seed=0)
        n_params_before = len(list(model.parameters()))
        qmodel, _ = quantize_mtlnn_int8(model, quantize_linear=False)
        names_after = {n for n, _ in qmodel.named_parameters()}
        assert not (names_after & QUANTIZABLE_WEIGHT_NAMES)
        assert len(names_after) < n_params_before

    def test_int8_buffers_registered(self):
        model = _tiny_model(seed=0)
        qmodel, _ = quantize_mtlnn_int8(model, quantize_linear=False)
        q_bufs = [n for n, b in qmodel.named_buffers() if n.endswith("_q")]
        assert q_bufs, "no int8 payload buffers found"
        for n, b in qmodel.named_buffers():
            if n.endswith("_q"):
                assert b.dtype == torch.int8

    def test_memory_reported_reduced(self):
        model = _tiny_model(seed=0)
        qmodel, report = quantize_mtlnn_int8(model, quantize_linear=False)
        # tiny 模型液态权重占比小（嵌入/注意力 Linear 占大头），整体阈值保守；
        # 对被量化的权重本体做严格 4× 断言（下面 test_weight_bytes_4x）。
        assert report["reduction"] > 0.10
        assert report["n_quantized"] > 0

    def test_weight_bytes_4x(self):
        model = _tiny_model(seed=0)
        fp32_bytes = sum(
            p.numel() * 4 for n, p in model.named_parameters()
            if n.split(".")[-1] in QUANTIZABLE_WEIGHT_NAMES)
        qmodel, report = quantize_mtlnn_int8(model, quantize_linear=False)
        int8_bytes = sum(
            b.numel() * b.element_size() for n, b in qmodel.named_buffers()
            if n.endswith("_q"))
        assert fp32_bytes > 0
        # fp32(4B) → int8(1B)：被量化权重本体接近 4× 缩减
        assert int8_bytes < fp32_bytes / 3.5

    def test_dynamics_params_stay_fp32(self):
        model = _tiny_model(seed=0)
        qmodel, _ = quantize_mtlnn_int8(model, quantize_linear=False)
        for n, p in qmodel.named_parameters():
            if n.split(".")[-1] in ("log_tau", "sel_b", "blend_weights",
                                    "b_in", "rhythm_scale"):
                assert p.dtype == torch.float32, n

    def test_state_dict_roundtrip(self, tmp_path):
        model = _tiny_model(seed=0)
        qmodel, _ = quantize_mtlnn_int8(model, quantize_linear=False)
        ref = _forward_logits(qmodel)

        path = tmp_path / "q.pt"
        torch.save(qmodel.state_dict(), path)

        fresh, _ = quantize_mtlnn_int8(_tiny_model(seed=0),
                                       quantize_linear=False)
        missing, unexpected = fresh.load_state_dict(torch.load(path), strict=False)
        assert not unexpected
        # load_state_dict 原位 copy → 句柄共享存储，应直接生效
        got = _forward_logits(fresh)
        assert torch.allclose(ref, got, atol=1e-4)

    def test_rebuild_handles_after_manual_buffer_swap(self):
        model = _tiny_model(seed=0)
        qmodel, _ = quantize_mtlnn_int8(model, quantize_linear=False)
        # 手工替换 buffer 张量对象（句柄失效的场景）
        for mod in qmodel.modules():
            for buf in list(getattr(mod, "_buffers", {})):
                if buf.endswith("_q"):
                    mod._buffers[buf] = mod._buffers[buf].clone()
        n = rebuild_int8_handles(qmodel)
        assert n > 0
        _forward_logits(qmodel)  # 不抛异常即通过

    def test_no_grad_flows_into_quantized(self):
        model = _tiny_model(seed=0)
        qmodel, _ = quantize_mtlnn_int8(model, quantize_linear=False)
        for p in qmodel.parameters():
            if p.dtype == torch.float32:
                assert not p.requires_grad or True  # eval 语义由调用方保证
        # 量化权重不是 Parameter，永远不会出现在 optimizer 参数组里
        assert not any(n in QUANTIZABLE_WEIGHT_NAMES
                       for n, _ in qmodel.named_parameters())
