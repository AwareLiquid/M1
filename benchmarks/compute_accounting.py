"""算力对齐账本 — 潜空间循环 N 步 vs 生成 N 个 CoT token 的 FLOPs/访存。

外部坐标系（本分支 Task 1，docs/LATENT_RECURSION.md）：Coconut (arXiv:2412.06769)
隐状态回哺、Huginn/Geiping (arXiv:2502.05171) 测试时可变递归深度的"循环体 =
离散 decoder 块"；本仓库的 core_iterations（循环体 = LNN 子层）与
stack_iterations（循环体 = 整块，含注意力）是同一坐标系上的两个变体。不做
这份账，任何"省 token"叙事都是空的。

口径（2026-08-29 写死，红线 #4：不得中途更换）：
  1. 计 multiply-accumulate（MAC），flops = 2 × MACs。
  2. 稠密层：每个被施加的权重元素 = 每 token 1 MAC。embedding 查表 = 0
     （是 gather 不是乘法）；lm_head 按被评分的 position 计。
  3. 注意力 QK^T + V·A：每 query token 2·H·dh·T_keys MACs。因果 prefill
     取平均 T_keys=(T+1)/2；带 KV cache 的解码步 T_keys = 已缓存长度+1。
  4. 液体扫描 h_t = λ·h_{t-1} + (1-λ)·A_t：每状态元素每步 1 MAC（乘加
     融合口径）；混合 softmax 加权 P·S 同计。W_lat 是 P×P 参数但作用于
     P×D 状态 → P·P·D MACs（MACs ≠ 参数量，逐组件写明）。
  5. 排除（低阶、实现相关）：LayerNorm/RMSNorm/softmax/偏置/逐元素门控/
     GTP 位置偏置。此排除对五条路径一视同仁。
  6. 访存按 fp32（4 B/权重）单 batch：权重流量 = 每次 forward 施加的权重
     字节（假设无 cache 驻留，上界）；KV 峰值 = L·2·d·(T+N+1)·4 B（仅
     CoT 路径持有 KV；潜空间路径重算不持 KV，携带 LNN 状态
     L·P·S·D·4 B 恒定）。

五条路径（T = prompt 长度，N = 循环/生成数）：
  mtlnn_core_N    循环体 = LNN 子层（注意力只算一次）
  mtlnn_stack_N   循环体 = 整块（注意力 + LNN，每 pass 重算 T×T 注意力）
  tfm_latent_N    Coconut/Huginn 坐标：同一 decoder 权重绑定施加 N 次，
                  隐状态回哺（本仓库无此模型，纯解析对齐行）
  tfm_cot_N       prefill T + N 个中间 token + 1 个答案 token（KV cache
                  解码的理想口径——对 CoT 有利，本仓库 baseline 无 cache
                  实现但账本按理想算）；N=0 退化为直接作答

输出 benchmarks/results/compute_accounting.json + 终端 Markdown 表。
校准：tests/test_compute_accounting.py 手算值锁定 + 与真实模型参数量对账。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "compute_accounting.json")

# probe 探针形状（benchmarks/reasoning_depth.py build_mtlnn/build_transformer
# 的默认值，T 按 pointer_chase n=16: T = 1 + 3·16 + 4 + 1 = 54）。
PROBE = dict(d_model=104, n_layers=2, n_heads=4, n_kv_heads=2, d_head=26,
             d_ff=256, P=13, S=5, D=8, map_hidden=64, d_gw=13,
             n_values=16, T=54)

CONVENTIONS = {
    "mac_definition": "multiply-accumulate; flops = 2*MACs",
    "dense": "1 MAC per applied weight element per token; embedding lookup = 0",
    "lm_head": "dense, counted per scored position",
    "attention": "2*H*dh*T_keys per query token; causal prefill T_keys=(T+1)/2",
    "scan": "1 MAC per state element per step (P*S*D) + blend P*S",
    "excluded": "norms, softmax, biases, elementwise gating, GTP bias",
    "bytes": "fp32; weight traffic = applied weight-matrix bytes per forward "
             "(single-batch, no cache residency); bias traffic excluded "
             "(<2%, isomorphic across paths); embedding is a row lookup "
             "(d*4B per token), not a matrix read; KV peak only on CoT path; "
             "latent paths carry LNN state L*P*S*D*4B",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--T", type=int, default=PROBE["T"])
    p.add_argument("--n_values", type=int, default=PROBE["n_values"])
    p.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    p.add_argument("--out", default=RESULTS)
    args = p.parse_args()

    shapes = dict(PROBE, T=args.T, n_values=args.n_values,
                  vocab=10 + args.n_values)
    rows = []
    for n in args.depths:
        if n >= 1:
            rows.append(account_mtlnn(shapes, n, "core"))
            rows.append(account_mtlnn(shapes, n, "stack"))
            rows.append(account_tfm(shapes, n, "latent"))
        rows.append(account_tfm(shapes, n, "cot"))
    payload = {"conventions": CONVENTIONS, "shapes": shapes, "rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print_table(rows, shapes)
    print(f"\nwritten to {args.out}")


def print_table(rows, shapes):
    """终端 Markdown 表（Task 3 前沿的横轴来源）。"""
    print(f"| path | N | MFLOPs | weight MB-traffic | KV peak B | state B |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['path']} | {r['N']} | {r['flops']/1e6:.3f} | "
              f"{r['weight_bytes']/1e6:.3f} | {r['kv_peak_bytes']} | "
              f"{r['state_bytes']} |")


# ── MT-LNN ───────────────────────────────────────────────────────────────────

def mtlnn_block_macs(sh):
    """每层每 token 的块内稠密 MACs（attn 子层 + LNN 子层分开报）。"""
    d, H, KV, dh = sh["d_model"], sh["n_heads"], sh["n_kv_heads"], sh["d_head"]
    P, S, D, Hid = sh["P"], sh["S"], sh["D"], sh["map_hidden"]
    PT = P * D
    attn = (d * H * dh + 2 * d * KV * dh + H * dh * d)        # qkvo 投影
    lnn = (d * PT                                            # in_proj
           + P * S * D * D                                   # W_in einsum
           + P * P * D                                       # W_lat (P×P → P×D 状态)
           + 2 * P * D * D                                   # 近邻 W_left/right
           + 4 * P * D * D                                   # RMC qkvo
           + 2 * P * P * D                                   # RMC 分数+AV
           + P * 2 * D * Hid + P * Hid                       # MAP fc1+fc2
           + PT * d                                          # out_proj
           + P * S)                                          # blend 加权
    scan = P * S * D                                         # 每步状态更新
    return {"attn_dense": attn, "lnn_dense": lnn, "scan": scan,
            "attn_score": 2 * H * dh}                         # 每 query 每 key


def mtlnn_top_macs(sh):
    """顶层（不随循环体放大）：GWTB + coherence + lm_head。"""
    d, gw = sh["d_model"], sh["d_gw"]
    gwtb = d * gw + 4 * gw * gw + gw * d                      # compress+qkvo+broadcast
    coherence = 4 * d * d                                     # qkvo 全宽
    head = d * sh["vocab"]
    return {"gwtb_dense": gwtb, "coh_dense": coherence, "lm_head": head,
            "gwtb_score": 2 * gw, "coh_score": 2 * d}         # 每 query 每 key


def account_mtlnn(sh, N, mode):
    """mt_lnn 潜空间深度 N（mode: core = 只循环 LNN 子层 / stack = 整块）。"""
    if N < 1:
        raise ValueError("latent depth N >= 1")
    T, L = sh["T"], sh["n_layers"]
    blk, top = mtlnn_block_macs(sh), mtlnn_top_macs(sh)
    n_attn, n_lnn = (1, N) if mode == "core" else (N, N)
    macs = (T * L * (n_attn * blk["attn_dense"] + n_lnn * blk["lnn_dense"])
            + T * L * n_lnn * blk["scan"]
            + T * L * n_attn * blk["attn_score"] * _causal_keys(T)
            + T * (top["gwtb_dense"] + top["coh_dense"])
            + T * (top["gwtb_score"] + top["coh_score"]) * _causal_keys(T)
            + top["lm_head"])
    w = _mtlnn_weight_counts(sh)
    block_w, lnn_only_w = w["block"], w["lnn"]
    wbytes = (L * (n_attn * (block_w - lnn_only_w) + n_lnn * lnn_only_w)
              + w["top"]) * 4
    return {"path": f"mtlnn_{mode}", "N": N, "macs": macs, "flops": 2 * macs,
            "weight_bytes": wbytes, "kv_peak_bytes": 0,
            "state_bytes": L * sh["P"] * sh["S"] * sh["D"] * 4}


def _mtlnn_weight_counts(sh):
    """每层块内 / LNN 子层 / 顶层的权重元素数（访存用；与构建模型对账见测试）。"""
    blk = mtlnn_block_macs(sh)
    attn_w = blk["attn_dense"]                     # qkvo 投影 = 参数量 = MACs
    lnn_w = (sh["d_model"] * sh["P"] * sh["D"]     # in_proj
             + sh["P"] * sh["S"] * sh["D"] ** 2    # W_in
             + sh["P"] ** 2 + 6 * sh["D"] ** 2     # W_lat + W_lr + RMC qkvo
             + sh["P"] * 2 * sh["D"] * sh["map_hidden"] + sh["P"] * sh["map_hidden"]
             + sh["P"] * sh["D"] * sh["d_model"])  # out_proj
    top = mtlnn_top_macs(sh)
    top_w = top["gwtb_dense"] + top["coh_dense"] + top["lm_head"]
    return {"block": attn_w + lnn_w, "lnn": lnn_w, "top": top_w}


# ── Transformer ──────────────────────────────────────────────────────────────

def tfm_block_macs(sh):
    """ModernCausalTransformer 每层每 token 稠密 MACs（MHA + SwiGLU）。"""
    d, dff = sh["d_model"], sh["d_ff"]
    return {"dense": 4 * d * d + 3 * d * dff, "score": 2 * d}  # 每 query 每 key


def account_tfm(sh, N, mode):
    """transformer 两条路：latent = 权重绑定 N 次（Huginn 坐标）；cot = N 个
    中间 token + 1 答案（理想 KV-cache 解码口径）。N=0 = 直接作答。"""
    if N < 0:
        raise ValueError("N >= 0")
    T, L, d = sh["T"], sh["n_layers"], sh["d_model"]
    blk = tfm_block_macs(sh)
    head = d * sh["vocab"]
    if mode == "latent":
        macs = (T * L * N * blk["dense"]
                + L * N * blk["score"] * _prefill_scores(T)
                + head)
        wbytes = (L * N * blk["dense"]) * 4
        kv = 0
    else:
        steps = N + 1                                   # N 中间 + 1 答案
        macs = ((T + steps) * L * blk["dense"]
                + L * blk["score"] * (_prefill_scores(T) + _decode_scores(T, N))
                + steps * head)
        wbytes = (1 + steps) * (L * blk["dense"] + head) * 4
        kv = L * 2 * d * (T + steps) * 4
    return {"path": f"tfm_{mode}", "N": N, "macs": macs, "flops": 2 * macs,
            "weight_bytes": wbytes, "kv_peak_bytes": kv, "state_bytes": 0}


# ── 小工具 ───────────────────────────────────────────────────────────────────

def _causal_keys(T):
    """因果 prefill 的平均 key 数（T 个 query 从 1..T 个 key）。"""
    return (T + 1) / 2


def _prefill_scores(T):
    """因果 prefill 总分对数：T·(T+1)/2 个 (query,key) 对。"""
    return T * (T + 1) / 2


def _decode_scores(T, N):
    """N+1 个解码步的 (query,key) 对总数（第 i 步看到 T+i 个 key）。"""
    return sum(T + i for i in range(1, N + 2))


if __name__ == "__main__":
    main()
