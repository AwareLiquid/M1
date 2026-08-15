"""V2 adapter correctness tests (CPU, no model download).

  1. FastWeightMemoryV2 (chunked parallel) == v1 FastWeightMemory (sequential
     reference) with identical weights, to fp32 tolerance.
  2. attach_mt_v2_adapters on a tiny random LlamaForCausalLM: forward runs,
     loss.backward() gives a gradient to EVERY v2 parameter (no silent dead
     params — the v1 lesson), base stays frozen.
  3. Param budget sanity: v2 default config is under 1.5M/adapter at D=2048.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.llama_adapter import FastWeightMemory
from mt_lnn.mt_lnn_v2 import (
    FastWeightMemoryV2,
    MTAdapterV2Config,
    MTResidualAdapterV2,
    attach_mt_v2_adapters,
    iter_mt_v2_adapter_parameters,
)


def test_fast_weight_chunked_matches_sequential():
    torch.manual_seed(0)
    B, T, Dm, Dmem, Hh = 2, 130, 96, 32, 2   # T deliberately not a chunk multiple
    ref = FastWeightMemory(Dm, d_mem=Dmem, n_heads=Hh, init_decay=0.9)
    new = FastWeightMemoryV2(Dm, d_mem=Dmem, n_heads=Hh, init_decay=0.9, chunk=32)
    new.load_state_dict(ref.state_dict())

    x = torch.randn(B, T, Dm)
    out_ref, (F_ref, z_ref) = ref(x)
    out_new, (F_new, z_new) = new(x)

    assert torch.allclose(out_ref, out_new, rtol=1e-4, atol=1e-5), (
        f"max abs diff {(out_ref - out_new).abs().max().item():.2e}"
    )
    assert torch.allclose(F_ref, F_new, rtol=1e-4, atol=1e-5)
    assert torch.allclose(z_ref, z_new, rtol=1e-4, atol=1e-5)
    print("[1/3] chunked fast-weight == sequential reference  OK")


def test_v2_adapter_grad_flow_on_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    wrapped = attach_mt_v2_adapters(
        model, every=4, n_protofilaments=4, d_proto=16,
        n_time_scales=3, proj_rank=8, fast_weight_dim=8,
    )
    assert wrapped, "no layers wrapped"

    ids = torch.randint(0, 128, (2, 33))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()

    dead = []
    n_v2 = 0
    for name, p in model.named_parameters():
        if "mt_adapter" in name:
            n_v2 += p.numel()
            assert p.requires_grad, f"{name} not trainable"
            if p.grad is None or not torch.isfinite(p.grad).all():
                dead.append(name)
        else:
            assert not p.requires_grad, f"base param {name} unfrozen"
            assert p.grad is None, f"base param {name} got a gradient"
    assert not dead, f"dead/NaN-grad v2 params: {dead}"
    # The residual gate must receive gradient (the v1 PEFT-freeze lesson)
    scales = [n for n, _ in model.named_parameters() if n.endswith("mt_adapter.scale")]
    assert scales, "no residual scale params found"
    print(f"[2/3] grad flow on tiny llama ({len(wrapped)} adapters, "
          f"{n_v2:,} v2 params, all grads finite)  OK")


def test_selective_decay_init_equivalence():
    """selective_decay is a STRICT generalisation: with W_dt zeroed (its
    learned part removed) the per-token dt collapses to softplus(b_dt)=1
    and the output must equal the static-decay path exactly."""
    torch.manual_seed(0)
    kw = dict(hidden_size=64, n_protofilaments=4, d_proto=16,
              n_time_scales=3, proj_rank=8, use_fast_weight=False)
    a_static = MTResidualAdapterV2(MTAdapterV2Config(**kw))
    a_sel = MTResidualAdapterV2(MTAdapterV2Config(selective_decay=True, **kw))
    missing, unexpected = a_sel.load_state_dict(a_static.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert set(missing) == {"mt_layer.W_dt", "mt_layer.b_dt"}, missing
    a_sel.mt_layer.W_dt.data.zero_()

    x = torch.randn(2, 21, 64)
    a_static.eval(), a_sel.eval()
    with torch.no_grad():
        out_s, out_d = a_sel(x), a_static(x)
    assert torch.allclose(out_s, out_d, rtol=1e-5, atol=1e-6), (
        f"max diff {(out_s - out_d).abs().max().item():.2e}"
    )

    # And with W_dt live, gradients reach it (the selectivity actually trains)
    a_sel.train()
    a_sel(x).sum().backward()
    g = a_sel.mt_layer.W_dt.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print("[5/6] selective decay: init-equivalent to static + W_dt trains  OK")


def test_selective_decay_exp_mode():
    """exp 参数化（E5e）：λ_t = 2·exp(-dt/τ)-1 ∈ (-1,1)，可达 ±1，
    输出有限、形状正确、梯度流动。mamba 模式是 exp 在 λ_t→λ_t 的退化。
    init_scale=1.0 使 adapter 转移差异不被残差 gate 稀释（默认 1e-3）。"""
    torch.manual_seed(1)
    kw = dict(hidden_size=64, n_protofilaments=4, d_proto=16,
              n_time_scales=3, proj_rank=8, use_fast_weight=False,
              init_scale=1.0)
    a_exp = MTResidualAdapterV2(
        MTAdapterV2Config(selective_decay=True, selective_decay_mode="exp", **kw))
    x = torch.randn(2, 21, 64)
    out = a_exp(x)
    assert out.shape == (2, 21, 64)
    assert torch.isfinite(out).all()
    # 梯度流动
    out.sum().backward()
    g = a_exp.mt_layer.W_dt.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # mamba 与 exp 模式在输出上应不同（exp 引入带符号转移）
    a_mamba = MTResidualAdapterV2(
        MTAdapterV2Config(selective_decay=True, selective_decay_mode="mamba", **kw))
    a_mamba.load_state_dict(a_exp.state_dict())
    a_mamba.eval(), a_exp.eval()
    with torch.no_grad():
        o_m, o_e = a_mamba(x), a_exp(x)
    assert (o_m - o_e).abs().max() > 1e-4, "exp vs mamba 应产生不同输出"
    print("[exp] selective exp mode: signed lam_t active, finite, trains  OK")


def test_peft_does_not_wrap_v2_internals():
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("[x/4] peft not installed — skipping contamination test")
        return
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    attach_mt_v2_adapters(model, every=4, n_protofilaments=4, d_proto=16,
                          n_time_scales=3, proj_rank=8, fast_weight_dim=8)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    bad = [n for n, _ in model.named_parameters()
           if "mt_adapter" in n and "lora_" in n]
    assert not bad, f"PEFT LoRA-wrapped v2 adapter internals: {bad}"
    print("[4/4] PEFT does not touch v2 internals  OK")


def test_param_budget_at_tinyllama_width():
    cfg = MTAdapterV2Config(hidden_size=2048)   # defaults: P=13,d=64,S=5,r=128,FW=64
    adapter = MTResidualAdapterV2(cfg)
    n = sum(p.numel() for p in adapter.parameters())
    assert n < 1_500_000, f"v2 adapter too fat: {n:,}"
    n6 = 6 * n
    pct = 100 * n6 / 1_100_048_384
    print(f"[3/3] param budget: {n:,}/adapter, x6 = {n6:,} "
          f"({pct:.3f}% of TinyLlama)  OK")


def test_delta_write_rule_stays_finite_at_realistic_scale():
    """Regression test for the 2026-07-12 Kaggle postmortem: at realistic
    d_mem (>=32) the unnormalised elu(x)+1 feature map gives ||k_t||^2 ~ d_mem,
    so the per-step delta-write contraction eigenvalue (lam - eta*||k_t||^2)
    was ~ -32 at init -- F blew up ~32x/STEP and hit fp32 overflow (NaN) within
    ~20 sequential steps, well inside a single 512-token training batch. This
    was invisible in earlier tiny-tensor smoke tests (small d_mem, few steps)
    and only surfaced burning a real GPU run. The fix L2-normalises the write
    key so the eigenvalue is exactly (lam - eta), guaranteed in (-1, 1) for any
    lam, eta in (0,1) (both sigmoid-parameterised) regardless of d_mem or input
    scale. This test pins that guarantee at the scale that actually broke."""
    torch.manual_seed(0)
    B, T, Dm, Dmem, Hh = 1, 512, 2048, 64, 1     # matches the TinyLlama-1.1B adapter config
    mem = FastWeightMemoryV2(Dm, d_mem=Dmem, n_heads=Hh, init_decay=0.95,
                             write_rule="delta")
    x = torch.randn(B, T, Dm) * 0.1              # matches --state_scale_init 0.1

    out, (F, z) = mem(x)
    assert torch.isfinite(out).all(), f"delta output non-finite, max={out.abs().max()}"
    assert torch.isfinite(F).all(), f"delta F non-finite, max={F.abs().max()}"
    # Not just finite -- bounded to a sane scale, not merely "not yet overflowed".
    assert F.abs().max().item() < 10.0, f"F growing unbounded: max={F.abs().max().item():.3e}"

    # Gradient still flows through the normalised path.
    x2 = torch.randn(B, 32, Dm, requires_grad=True) * 0.1
    out2, _ = mem(x2)
    out2.sum().backward()
    assert torch.isfinite(mem.eta_raw.grad).all()
    assert torch.isfinite(mem.W_k.weight.grad).all()
    print("[6/6] delta write rule stable at realistic d_mem/scale (F bounded, grads finite)  OK")


if __name__ == "__main__":
    test_fast_weight_chunked_matches_sequential()
    test_v2_adapter_grad_flow_on_tiny_llama()
    test_param_budget_at_tinyllama_width()
    test_selective_decay_init_equivalence()
    test_peft_does_not_wrap_v2_internals()
    test_delta_write_rule_stays_finite_at_realistic_scale()
    print("all v2 tests passed")
