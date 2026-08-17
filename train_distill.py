"""P2 蒸馏：强基座教师 → O 系列（attention-free）学生。

BEAT_TRANSFORMER_PLAN.md P2（体积维度）——把类脑机制蒸馏到端侧小模型。
教师（Qwen2.5-0.5B，151936 vocab）的 logits 蒸馏到 O 系列学生（纯 LNN，
attention_layers=()），学生用教师同一 tokenizer 保证 KL 对齐。

蒸馏损失 = T²·KL(softmax(s_logits/T) || softmax(t_logits/T)) + α·CE(s_logits, labels)
（Hinton 2015，温度 T 软化教师分布）。

运行（云 GPU）:
  py -3.11 train_distill.py --teacher Qwen/Qwen2.5-0.5B-Instruct \
      --dataset Salesforce/wikitext --dataset_config wikitext-2-raw-v1 \
      --d_model 384 --n_layers 8 --steps 5000 --temperature 4.0 --alpha 0.5

注意：学生 vocab = 教师 vocab（151936），embedding 层占大头（约 58M @ d_model 384）。
端侧 <5MB 需要后续词表裁剪 + int8 量化（第二段，见 deploy 注释）。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def tokenize_wikitext(dataset_name, dataset_config, tokenizer, seq_len):
    """复用 train_llama_mt_adapter 的 wikitext tokenize 逻辑（简化版）。"""
    ds = load_dataset(dataset_name, dataset_config)
    train = ds["train"]["text"]
    if isinstance(train, str):
        train = [train]
    # 简单分块（生产可优化为 batch tokenize）
    all_ids = []
    buf = []
    for text in train:
        if not text or not text.strip():
            continue
        ids = tokenizer(text)["input_ids"]
        buf.extend(ids)
        while len(buf) >= seq_len + 1:
            all_ids.append(buf[:seq_len + 1])
            buf = buf[seq_len:]
    return all_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--n_protofilaments", type=int, default=13)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="CE loss weight vs KL (1.0 = pure CE, 0.0 = pure KL)")
    p.add_argument("--sel_mode", choices=["tanh", "exp"], default="exp")
    p.add_argument("--out_dir", default="checkpoints/distill")
    p.add_argument("--eval_every", type=int, default=500)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  teacher={args.teacher}")

    # 教师 + tokenizer（同一 vocab 保证 KL 对齐）
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16).to(device).eval()
    teacher_vocab = teacher.config.vocab_size

    # 学生：O 系列（attention_layers=() 纯 LNN），教师同一 vocab
    cfg = MTLNNConfig(
        vocab_size=teacher_vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=1,
        d_head=args.d_model // args.n_heads,
        n_protofilaments=args.n_protofilaments,
        max_seq_len=args.seq_len,
        dropout=0.0,
        attention_dropout=0.0,
        attention_layers=(),           # O 系列：无注意力，纯 LNN
        gwtb_n_heads=1,                # d_gw = d_model//8 需被整除（1 恒满足）
        selective_decay=True,
        selective_decay_mode=args.sel_mode,
        tau_max=200.0,                 # exp 参数化需要大 tau_max 逼近 ±1
    )
    student = MTLNNModel(cfg).to(device)
    n_params = student.get_num_params()
    print(f"student params: {n_params/1e6:.1f}M (O-series, attention-free, "
          f"{args.d_model}d x {args.n_layers}L)")

    # 数据
    data = tokenize_wikitext(args.dataset, args.dataset_config, tokenizer,
                             args.seq_len)
    print(f"train chunks: {len(data)}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.999))
    student.train()
    step = 0
    t0 = time.time()
    while step < args.steps:
        for chunk in data:
            if step >= args.steps:
                break
            ids = torch.tensor(chunk, device=device).unsqueeze(0)   # (1, T+1)
            inp, lbl = ids[:, :-1], ids[:, 1:]                      # (1, T)

            with torch.no_grad():
                t_logits = teacher(inp).logits.float()              # (1, T, V)

            s_out = student(inp)                                    # 无 labels → logits
            s_logits = s_out["logits"]                              # (1, T, V)

            # 蒸馏损失：KL(softmax(s/T) || softmax(t/T)) * T^2 + alpha * CE
            T = args.temperature
            kl = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            ce = F.cross_entropy(s_logits.reshape(-1, teacher_vocab),
                                 lbl.reshape(-1))
            loss = (1.0 - args.alpha) * kl + args.alpha * ce
            loss = loss / args.grad_accum

            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            if step % args.eval_every == 0:
                print(f"step {step:5d}  kl={kl.item():.3f}  ce={ce.item():.3f}  "
                      f"loss={loss.item() * args.grad_accum:.3f}  "
                      f"({(time.time() - t0):.0f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"distill_{args.d_model}d.pt")
    torch.save({"config": cfg, "model": student.state_dict()}, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
