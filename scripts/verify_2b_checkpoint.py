"""verify_2b_checkpoint.py — M1-2B 发布前校验（权重到本地后跑这个）

检查项:
  1. checkpoint 结构 (config / model_state / step)
  2. config 与预期一致 (2912d × 35L, ~1.9B 参数; 不匹配只警告, config 才是
     服务的真源)
  3. state_dict 与 MTLNNModel 的 missing/unexpected 计数 (服务端 strict=False,
     只关心关键层是否缺)
  4. 可选冒烟生成 (--tokens N, CPU 上 1.9B 很慢, 默认跳过)

用法:
  py -3.11 scripts/verify_2b_checkpoint.py E:\\M1\\checkpoints\\ckpt_120000.pt
  py -3.11 scripts/verify_2b_checkpoint.py ckpt.pt --tokens 8 --json
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 官网/卡片声明的 2B 规格 (release-m1-2b-checklist.md 同源)
EXPECT_D_MODEL = 2912
EXPECT_N_LAYERS = 35
EXPECT_PARAMS_RANGE = (1.7e9, 2.2e9)


def load_model(cfg: dict):
    from mt_lnn.config import MTLNNConfig
    from mt_lnn.model import MTLNNModel

    valid = {f.name for f in dataclasses.fields(MTLNNConfig) if f.init}
    cfg_clean = {k: v for k, v in cfg.items() if k in valid}
    return MTLNNModel(MTLNNConfig(**cfg_clean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint .pt 路径")
    ap.add_argument("--tokens", type=int, default=0,
                    help="冒烟生成 token 数 (0=跳过; CPU 上 1.9B 每 token 数十秒)")
    ap.add_argument("--json", action="store_true", help="附加输出 JSON 摘要")
    args = ap.parse_args()

    path = Path(args.ckpt)
    assert path.exists(), f"not found: {path}"

    report = {"path": str(path), "size_gb": round(path.stat().st_size / 1e9, 2)}

    print(f"[1/5] mmap 加载 (不整读入内存) ... {report['size_gb']} GB")
    ckpt = torch.load(str(path), map_location="cpu", mmap=True,
                      weights_only=False)  # 与服务端 server.py 同口径
    report["step"] = ckpt.get("step")
    cfg = ckpt.get("config")
    assert cfg is not None, "checkpoint 缺 config 字段, 服务端无法加载"

    print("[2/5] config:")
    print(json.dumps(cfg, ensure_ascii=False, indent=2, default=str))
    report["config"] = cfg

    d, L = cfg.get("d_model"), cfg.get("n_layers")
    if d != EXPECT_D_MODEL or L != EXPECT_N_LAYERS:
        print(f"  !! 规格警告: d_model={d} n_layers={L} (预期 "
              f"{EXPECT_D_MODEL}/{EXPECT_N_LAYERS}) — 确认没拿错 checkpoint")
    report["config_ok"] = (d == EXPECT_D_MODEL and L == EXPECT_N_LAYERS)

    state = ckpt.get("model_state")
    assert state is not None, "checkpoint 缺 model_state 字段"
    n_params = sum(t.numel() for t in state.values())
    lo, hi = EXPECT_PARAMS_RANGE
    report["n_params"] = n_params
    ok = lo <= n_params <= hi
    print(f"[3/5] state_dict 参数合计: {n_params/1e9:.2f}B "
          f"({'OK' if ok else '!! 超出 1.7-2.2B 预期区间'})")
    report["params_ok"] = ok

    print("[4/5] 与 MTLNNModel 结构对齐 (strict=False) ...")
    model = load_model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    report["missing_keys"] = len(missing)
    report["unexpected_keys"] = len(unexpected)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing 样例:", missing[:5])
    if unexpected:
        print("  unexpected 样例:", unexpected[:5])

    if args.tokens > 0:
        print(f"[5/5] 冒烟生成 {args.tokens} tokens (CPU) ...")
        model.eval()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        ids = torch.tensor([tok.encode("The AwareLiquid model is")],
                           dtype=torch.long)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.tokens,
                                 do_sample=False, eos_token_id=tok.eos_token_id)
        text = tok.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"  → {text!r}")
        report["smoke_text"] = text
    else:
        print("[5/5] 冒烟生成跳过 (--tokens N 开启)")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("\n校验完成:", "PASS" if (report["config_ok"] and report["params_ok"])
          else "CHECK (见上方警告)")
    return 0 if (report["config_ok"] and report["params_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
