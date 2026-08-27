"""ppl_diagnosis.py — 128M 混合模型 PPL 根因诊断（CPU 可跑）

回答路线核心问题：液态核心对语言建模质量是贡献、中性、还是损伤？

消融配置:
  full       - 完整模型 (基线)
  lnn_zeroed - 液态核心输出置零 (attention + GWTB 保持不变)
              → PPL 基本不变 = 液态对 LM 质量中性 (形态优势路线成立)
              → PPL 大幅变差   = 液态确实贡献语言能力
              → PPL 变好       = 液态在损伤 LM 质量 (架构必须改)

用法:
  py -3.11 scripts/ppl_diagnosis.py --ckpt checkpoints/hybrid_125m_serve.pt --windows 60
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.mt_lnn_layer import MTLNNLayer  # noqa: E402


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
    missing, unexpected = model.load_state_dict(ckpt["model_state"],
                                                strict=False)
    print(f"[load] step={ckpt.get('step')} missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    return model.eval()


def zero_lnn(model):
    """液态核心输出置零：跑原路径但 out 换 zeros（保持 h 状态形状一致）。"""
    orig = MTLNNLayer.forward

    def zeroed(self, x, h_prev=None, position_offset=0, use_scan=True,
               pad_mask=None):
        out, h_last = orig(self, x, h_prev, position_offset=position_offset,
                           use_scan=use_scan, pad_mask=pad_mask)
        return torch.zeros_like(out), h_last

    MTLNNLayer.forward = zeroed
    return orig


def zero_attn(model):
    """注意力输出置零：只留液态路径（测液态单独的语言能力上限）。"""
    from mt_lnn.mt_attention import MicrotubuleAttention
    orig = MicrotubuleAttention.forward

    def zeroed(self, x, pad_mask=None, past_kv=None, position_offset=0,
               use_cache=False, h_prev=None):
        out, new_kv = orig(self, x, pad_mask=pad_mask, past_kv=past_kv,
                           position_offset=position_offset, use_cache=use_cache,
                           h_prev=h_prev)
        return torch.zeros_like(out), new_kv

    MicrotubuleAttention.forward = zeroed
    return orig


def eval_ppl(model, tok, windows, device):
    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for i, w in enumerate(windows):
            ids = torch.tensor([w], dtype=torch.long, device=device)
            out = model(input_ids=ids, labels=ids, use_cache=False,
                        use_lnn_recurrence=False)
            loss = out["loss"]
            n_tokens = ids.shape[1] - 1
            total_nll += float(loss) * n_tokens
            total_tokens += n_tokens
            if (i + 1) % 10 == 0:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(windows)}] "
                      f"ppl={torch.exp(torch.tensor(total_nll/total_tokens)):.2f} "
                      f"({dt/(i+1):.1f}s/win)")
    return total_nll / total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hybrid_125m_serve.pt")
    ap.add_argument("--val", default="data/wikitext2_val.txt")
    ap.add_argument("--windows", type=int, default=60,
                    help="评测窗口数 (每窗口 512 tokens; CPU 每窗口 ~1-2s)")
    args = ap.parse_args()

    device = "cpu"
    torch.set_num_threads(8)

    model = load_model(args.ckpt)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    text = Path(args.val).read_text(encoding="utf-8", errors="ignore")
    ids_all = tok.encode(text)
    print(f"[data] {args.val}: {len(ids_all)} tokens → "
          f"{len(ids_all)//512} windows, 用前 {args.windows} 个")
    windows = [ids_all[i*512:(i+1)*512] for i in range(args.windows)
               if (i+1)*512 <= len(ids_all)]

    print("\n=== A: full ===")
    nll_full = eval_ppl(model, tok, windows, device)
    ppl_full = torch.exp(torch.tensor(nll_full)).item()

    print("\n=== B: lnn_zeroed ===")
    orig_lnn = zero_lnn(model)
    nll_zero = eval_ppl(model, tok, windows, device)
    ppl_zero = torch.exp(torch.tensor(nll_zero)).item()
    MTLNNLayer.forward = orig_lnn  # 恢复

    print("\n=== C: attn_zeroed (纯液态) ===")
    orig_attn = zero_attn(model)
    nll_attn0 = eval_ppl(model, tok, windows, device)
    ppl_attn0 = torch.exp(torch.tensor(nll_attn0)).item()
    from mt_lnn.mt_attention import MicrotubuleAttention
    MicrotubuleAttention.forward = orig_attn  # 恢复

    delta = ppl_zero / ppl_full
    delta_attn = ppl_attn0 / ppl_full
    print("\n═══════════ 诊断结果 ═══════════")
    print(f"full         PPL = {ppl_full:.2f}")
    print(f"lnn_zeroed   PPL = {ppl_zero:.2f}  (比值 {delta:.3f}) — 只剩注意力")
    print(f"attn_zeroed  PPL = {ppl_attn0:.2f}  (比值 {delta_attn:.3f}) — 只剩液态")
    print()
    if delta_attn < 2.0 and delta > 3.0:
        verdict = ("液态核心是语言质量的主引擎（纯液态接近完整模型），"
                   "注意力是辅助。混合架构值得 2B 投入，且可探索液态为主、"
                   "注意力稀疏化的架构")
    elif delta > 3.0:
        verdict = "液态核心显著贡献语言质量（但需注意力配合）——混合架构值得 2B 投入"
    elif delta < 1.05:
        verdict = ("液态核心对 LM 质量≈中性——其价值在记忆形态，不在语言表达。"
                   "路线: 液态作记忆旁路 + 强 Transformer 基座")
    else:
        verdict = "液态核心小幅贡献语言质量——保留混合架构但别指望它翻盘"
    print(f"结论: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
