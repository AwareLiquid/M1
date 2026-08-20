"""train_curriculum.py — 长度课程训练（S2: curriculum 长序列 + 外推 ×8 验证）。

M1 的 train.py 用固定 seq_len；本脚本按长度调度逐 stage 递增训练，并在每个
stage 末做外推探针（在 L × {1,2,4,8} 上测 PPL），把 M1 在 parity 上发现的
"外推极限 = 训练长度 ×8" 规律迁移到 LM 训练协议。

设计（docs/phase2-plan.md S2）：curriculum 到 32K，验证 128K-256K 外推。

用法：
  # 冒烟（CPU，无需数据，~1 分钟）
  python train_curriculum.py --dummy --seq_schedule 32,64,128 --steps_per_stage 20 --batch 2 --vocab_size 200
  # 正式（云端 GPU，需先 prepare_data.py）
  python train_curriculum.py --seq_schedule 512,1024,2048,4096,8192,16384,32768 --steps_per_stage 2000 --batch 8
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


# ---------------------------------------------------------------------------
# 数据（内联实现，避免依赖 train.py 的模块级副作用）
# ---------------------------------------------------------------------------

class MemmapBinDataset(Dataset):
    """memmap uint16 token 窗口（与 prepare_data.py 输出兼容）。"""

    def __init__(self, bin_path: str, seq_len: int, stride: int = None):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.stride = stride or seq_len

    def __len__(self):
        return max(1, (len(self.data) - self.seq_len - 1) // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


class DummyDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 64):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.randint(1, self.vocab_size, (self.seq_len,))
        return x, x.clone()


# ---------------------------------------------------------------------------
# 评估：外推探针
# ---------------------------------------------------------------------------

@torch.no_grad()
def ppl_at(model, loader, max_batches=4):
    """返回 (PPL, CE)。"""
    device = next(model.parameters()).device
    model.eval()
    total, n = 0.0, 0
    for inp, lbl in loader:
        if n >= max_batches:
            break
        inp = inp.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        out = model(inp, labels=lbl)
        total += float(out.get("lm_loss", out["loss"]))
        n += 1
    model.train()
    if n == 0:
        return float("inf"), float("inf")
    return math.exp(min(total / n, 20.0)), total / n


@torch.no_grad()
def extrapolation_probe(model, make_loader, base_len, device):
    """在 base_len × {1,2,4,8} 长度上测 PPL——×8 外推规律的直接读数。"""
    probes = {}
    for mult in (1, 2, 4, 8):
        L = base_len * mult
        loader = make_loader(L)
        ppl, ce = ppl_at(model, loader)
        probes[f"L{mult}x"] = {"length": L, "ppl": round(ppl, 2)}
        print(f"    probe {mult}x (L={L}): PPL {ppl:.2f}", flush=True)
    return probes


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_schedule", default="512,1024,2048,4096,8192,16384,32768",
                    help="逗号分隔的长度调度（最后一个为 max_seq_len）")
    ap.add_argument("--steps_per_stage", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=832)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=13)
    ap.add_argument("--n_kv_heads", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup_steps", type=int, default=120)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--vocab_size", type=int, default=1000)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="curriculum_results.jsonl")
    ap.add_argument("--selective_decay", action="store_true")
    ap.add_argument("--sel_mode", default="exp")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = [int(x) for x in args.seq_schedule.split(",")]
    max_len = max(schedule)
    print(f"curriculum {schedule} · steps/stage {args.steps_per_stage} · device {device}")

    # 数据源（各 stage 按长度重建 loader）
    if args.dummy:
        def make_loader(L, train=True):
            ds = DummyDataset(args.vocab_size, L,
                              n_samples=32 if train else 8)
            return DataLoader(ds, batch_size=args.batch, shuffle=train, drop_last=True)
    else:
        meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
        args.vocab_size = meta["vocab_size"]

        def make_loader(L, train=True):
            split = "train" if train else "validation"
            path = os.path.join(args.data_dir, f"{split}.bin")
            if not os.path.exists(path):
                path = os.path.join(args.data_dir, "test.bin")
            ds = MemmapBinDataset(path, L)
            return DataLoader(ds, batch_size=args.batch, shuffle=train, drop_last=True)

    # 模型：max_seq_len = 调度最大值（RoPE/注意力窗口上限）
    config = MTLNNConfig(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
        d_head=args.d_model // args.n_heads,
        max_seq_len=max_len, vocab_size=args.vocab_size,
        dropout=0.0,
        selective_decay=args.selective_decay,
        selective_decay_mode=args.sel_mode,
    )
    model = MTLNNModel(config).to(device)
    print(f"params: {model.get_num_params()/1e6:.1f}M  max_seq_len={max_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8)
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("step", 0))
        print(f"[resume] step {global_step}")

    results = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for stage, L in enumerate(schedule):
        loader = make_loader(L, train=True)
        t0 = time.time()
        stage_loss, n = 0.0, 0
        stage_iter = iter(loader)
        for step in range(args.steps_per_stage):
            global_step += 1
            try:
                inp, lbl = next(stage_iter)
            except StopIteration:
                stage_iter = iter(loader)
                inp, lbl = next(stage_iter)
            inp = inp.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            out = model(inp, labels=lbl)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            stage_loss += float(loss)
            n += 1
        avg = stage_loss / max(n, 1)
        print(f"[stage {stage}] L={L} · avg loss {avg:.4f} · "
              f"{time.time()-t0:.0f}s · global_step {global_step}", flush=True)

        # 外推探针（stage 末）
        print(f"  extrapolation probe (base L={L}):")
        probes = extrapolation_probe(model, lambda ln: make_loader(ln, train=False),
                                     L, device)
        rec = {"stage": stage, "length": L, "global_step": global_step,
               "avg_loss": round(avg, 4), "probes": probes}
        results.append(rec)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # 保存最终 checkpoint（含 optimizer，支持续训）
    ckpt_path = args.out.rsplit(".", 1)[0] + "_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": global_step}, ckpt_path)
    print(f"saved {ckpt_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
