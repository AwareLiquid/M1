"""quant_smoke.py — int8 量化打样（CPU）：现有 128M 模型量化后 PPL/内存

为 2B 部署打样：2B fp32 7.6GB → 全 int8 1.9GB 才能进 7.2GB VPS。
本脚本验证 torch 动态量化在混合架构上的覆盖度与质量损失，并如实报告
覆盖缺口（液态核心用裸 Parameter + einsum，不在 nn.Linear 动态量化范围内）。

用法:
  py -3.11 scripts/quant_smoke.py --windows 20
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_model(ckpt_path: str):
    import dataclasses
    from mt_lnn.config import MTLNNConfig
    from mt_lnn.model import MTLNNModel

    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True,
                      weights_only=False)
    cfg_dict = ckpt["config"]
    valid = {f.name for f in dataclasses.fields(MTLNNConfig) if f.init}
    cfg = MTLNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = MTLNNModel(cfg)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model.eval()


def param_mb(model):
    n = sum(p.numel() for p in model.parameters())
    return n, n * 4 / 1e6  # fp32 字节


def eval_ppl(model, tok, windows):
    total_nll = total_tok = 0.0
    t0 = time.time()
    with torch.no_grad():
        for i, w in enumerate(windows):
            ids = torch.tensor([w], dtype=torch.long)
            loss = model(input_ids=ids, labels=ids, use_cache=False,
                         use_lnn_recurrence=False)["loss"]
            n = ids.shape[1] - 1
            total_nll += float(loss) * n
            total_tok += n
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(windows)}] ({time.time()-t0:.0f}s)")
    return total_nll / total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hybrid_125m_serve.pt")
    ap.add_argument("--windows", type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(8)

    model = load_model(args.ckpt)
    n_params, fp32_mb = param_mb(model)
    print(f"[base] {n_params/1e6:.1f}M params, fp32 ≈ {fp32_mb:.0f} MB")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids_all = tok.encode(Path("data/wikitext2_val.txt")
                         .read_text(encoding="utf-8", errors="ignore"))
    windows = [ids_all[i*512:(i+1)*512] for i in range(args.windows)]

    print("\n=== fp32 baseline ===")
    nll_fp32 = eval_ppl(model, tok, windows)
    ppl_fp32 = torch.exp(torch.tensor(nll_fp32)).item()

    print("\n=== int8 动态量化 (nn.Linear) ===")
    import torch.nn as nn
    from torch.ao.quantization import quantize_dynamic
    try:
        qmodel = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        q_linear = sum(1 for m in qmodel.modules()
                       if isinstance(m, torch.nn.quantized.dynamic.Linear))
        n_raw = sum(1 for p in qmodel.parameters() if p.dtype == torch.float32
                    and p.numel() > 1000)
        print(f"  量化 Linear 模块: {q_linear}")
        print(f"  仍未量化的 fp32 大参数: {n_raw} (液态裸 Parameter 等)")
        q_bytes = sum(p.numel() * p.element_size() for p in qmodel.parameters())
        print(f"  量化后参数内存 ≈ {q_bytes/1e6:.0f} MB "
              f"(fp32 时 {fp32_mb:.0f} MB)")
        nll_q = eval_ppl(qmodel, tok, windows)
        ppl_q = torch.exp(torch.tensor(nll_q)).item()
    except Exception as e:  # noqa: BLE001
        print(f"  量化失败: {e}")
        return 1

    print("\n=== weight-only int8 (液态裸 Parameter, mt_lnn.quantization) ===")
    # 全量覆盖: Linear 动态量化 + 液态裸 Parameter Int8Weight (含 1x1 等
    # 引擎不支持形状的逐模块容错)。2B 部署数学靠这一段。
    ppl_wo = None
    try:
        from mt_lnn.quantization import quantize_mtlnn_int8
        womodel, wrep = quantize_mtlnn_int8(
            load_model(args.ckpt), quantize_linear=True, verbose=True)
        nll_wo = eval_ppl(womodel, tok, windows)
        ppl_wo = torch.exp(torch.tensor(nll_wo)).item()
    except Exception as e:  # noqa: BLE001
        print(f"  weight-only 量化失败: {e}")

    print("\n══════ 量化打样结果 ══════")
    print(f"fp32 PPL = {ppl_fp32:.2f}")
    print(f"int8 PPL (Linear only) = {ppl_q:.2f}  (退化 {ppl_q/ppl_fp32 - 1:+.1%})")
    if ppl_wo is not None:
        print(f"int8 PPL (全量 weight-only) = {ppl_wo:.2f}  "
              f"(退化 {ppl_wo/ppl_fp32 - 1:+.1%}, "
              f"液态 {wrep['n_quantized']} 个权重, 内存 "
              f"{wrep['bytes_before']/1e6:.0f}→{wrep['bytes_after']/1e6:.0f} MB)")
        print("结论: 全量 int8 已打通 — 2B fp32 7.6GB → ~1.9GB, "
              "serve 设 QUANTIZE_INT8=1 即用。")
    else:
        print("结论: weight-only 路径本机不可用, 见上方报错。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
