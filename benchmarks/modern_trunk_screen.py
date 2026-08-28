"""modern_trunk_screen.py — modern-trunk 三缺失件的 2K 步配对方向筛选.

⚠️⚠️ 筛选纪律 (SCREENING DISCIPLINE — 写死, 不可协商):
2K 步数字只用于方向筛选 (GO/NO-GO for 20K confirm run), 禁止进入
RESULTS.md / README.md / 任何 headline。2026-07-19 收回 31% 主张的教训
正是 2K 步外推。任何值得记录的结论必须来自 20K 确认跑 (runbook 见
docs/MODERN_TRUNK.md), 本脚本输出的 JSON 一律携带 screening_only 标记。

预注册判优 (写死于代码, 跑之前就已定):
  all_on 配置的配对 val-PPL 差值 (vs base, 同 seed) 需在 ≥2/3 seeds 上
  为负 (更优) 才值得排 20K 确认跑; 否则 screen 记为 NULL, 不排确认跑。

四配置 (同 seed 配对, 每 config×seed 一个 row JSON, 断点续跑 = 重跑同命令):
  base      三旋钮全 off (历史位等价路径)
  +ffn      ffn_swiglu=True
  +qk_norm  qk_norm=True
  all_on    ffn_swiglu + qk_norm + scaled_residual_init

P0 固定口径: WikiText-103, gpt2 tokenizer, seq 512, batch 4, lr 3e-4,
AdamW beta2 0.95, grad clip 1.0, fp32 (与 scaling_comparison.py P0 历史口径
一致; tokenization 直接复用其 build_chunks, 保证对照可比)。

用法:
  python benchmarks/modern_trunk_screen.py --smoke          # 管道验证 (<10min)
  python benchmarks/modern_trunk_screen.py                  # 全量 2K 筛选
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

CONFIGS = ["base", "+ffn", "+qk_norm", "all_on"]
CONFIG_KNOBS = {
    "base": {},
    "+ffn": {"ffn_swiglu": True},
    "+qk_norm": {"qk_norm": True},
    "all_on": {"ffn_swiglu": True, "qk_norm": True,
               "scaled_residual_init": True},
}
VOCAB = 50257                                   # gpt2 BPE
BETA2 = 0.95                                    # P0 历史口径
GRAD_CLIP = 1.0
CONFIRM_MIN_SEEDS = 2                           # 预注册判优阈值
P0_RECIPE = {"dataset": "wikitext-103-raw-v1", "tokenizer": "gpt2",
             "seq_len": 512, "batch": 4, "lr": 3e-4, "beta2": BETA2,
             "grad_clip": GRAD_CLIP, "dtype": "fp32"}
DISCIPLINE = ("SCREENING ONLY: 2K-step numbers decide direction, nothing "
              "else. They must NOT enter RESULTS.md/README (the 2026-07-19 "
              "31% retraction was a 2K-step extrapolation). Confirm at 20K "
              "via the docs/MODERN_TRUNK.md runbook before any headline.")


def main():
    args = parse_args()
    if args.smoke:
        apply_smoke(args)
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    args.tag = "smoke" if args.smoke else f"{args.steps}step"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} steps={args.steps} seeds={args.seeds} "
          f"configs={CONFIGS} smoke={args.smoke}\n⛔ {DISCIPLINE}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = run_screen(args, device)
    table = paired_table(rows, args.seeds)
    print_report(rows, table, args.seeds)
    save_report(args, rows, table)


def train_model(model, chunks, args, device):
    """2K 步训练循环 (fp32, P0 口径); 返回 {final_loss, stable, steps}。"""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, BETA2))
    model.train()
    stable, last, step, t0 = True, float("nan"), 0, time.time()
    while step < args.steps and stable:
        order = torch.randperm(len(chunks))
        for start in range(0, len(order) - args.batch + 1, args.batch):
            ids = chunks[order[start:start + args.batch]].to(device)
            loss, finite = train_step(model, ids, opt)
            if not finite:
                stable = False
                print(f"  NON-FINITE loss at step {step} — UNSTABLE", flush=True)
                break
            last, step = loss, step + 1
            if step % args.log_every == 0:
                toks = args.batch * args.log_every * args.seq_len
                print(f"  {step}/{args.steps} loss {last:.4f} "
                      f"| {toks / max(time.time() - t0, 1e-3):.0f} tok/s",
                      flush=True)
                t0 = time.time()
            if step >= args.steps:
                break
    return {"final_loss": last, "stable": stable, "steps": step}


def build_model(name, args, device):
    """按配置名构建 MT-LNN (lean core + modern-trunk 旋钮), fp32。"""
    from mt_lnn.config import MTLNNConfig
    from mt_lnn.model import MTLNNModel
    cfg = MTLNNConfig(
        vocab_size=VOCAB, max_seq_len=args.seq_len,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=13, n_kv_heads=1, d_head=args.d_model // 13,
        dropout=0.0, attention_dropout=0.0, tie_embeddings=True,
        # GWTB 头数必须整除 d_gw=d_model//8; 832→104 整除 4 (全量口径与
        # scaling_comparison 一致), smoke 的 104→13 只能整除 1。
        gwtb_n_heads=4 if (args.d_model // 8) % 4 == 0 else 1,
        # Lean core trunk (与 scaling_comparison 的 mt_lnn arch 同口径):
        use_predictive_coding=False, use_competitive_gwtb=False,
        use_world_model=False, use_hebbian=False, use_rhythm=False,
        **CONFIG_KNOBS[name],
    )
    model = MTLNNModel(cfg).to(device)
    return model, sum(p.numel() for p in model.parameters())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=832)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--wikitext", default="wikitext-103-raw-v1")
    ap.add_argument("--train_token_cap", type=int, default=50_000_000,
                    help="train split tokenization 上限 (与对照口径一致性"
                         ">全量多样性; None = 全 split)")
    ap.add_argument("--eval_chunks", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--out_dir", default="benchmarks/results")
    ap.add_argument("--smoke", action="store_true",
                    help="管道验证档: wikitext-2 + 小模型 + 30 步")
    return ap.parse_args()


def paired_table(rows, seeds):
    """配对差值表 + 预注册判优 (CONFIRM / NULL)。"""
    by = {(r["config"], r["seed"]): r for r in rows}
    table = {}
    for name in CONFIGS[1:]:
        diffs = []
        for seed in seeds:
            b, v = by.get(("base", seed)), by.get((name, seed))
            if b and v and b["stable"] and v["stable"]:
                diffs.append(round(v["val_ppl"] - b["val_ppl"], 4))
        table[name] = {"ppl_minus_base_per_seed": diffs,
                       "n_seeds_better": sum(d < 0 for d in diffs)}
    better = table["all_on"]["n_seeds_better"]
    return {"paired_diff": table,
            "pre_registered_rule": (f"all_on beats base on >={CONFIRM_MIN_SEEDS}"
                                    f"/{len(seeds)} seeds → CONFIRM, else NULL"),
            "all_on_n_seeds_better": better,
            "verdict": "CONFIRM" if better >= CONFIRM_MIN_SEEDS else "NULL"}


def print_report(rows, table, seeds):
    by = {(r["config"], r["seed"]): r for r in rows}
    print("\n" + "=" * 74, flush=True)
    print(f"MODERN TRUNK SCREEN | {seeds} seeds | SCREENING ONLY (2K 步)",
          flush=True)
    print(f"{'config':<10} {'params':>11} {'stable':>7}  val_ppl per seed",
          flush=True)
    for name in CONFIGS:
        rs = [by.get((name, s)) for s in seeds]
        ppls = " ".join(f"{r['val_ppl']:.2f}" if r else "-" for r in rs)
        params = f"{rs[0]['params']:,}" if rs and rs[0] else "-"
        stable = all(bool(r and r["stable"]) for r in rs)
        print(f"{name:<10} {params:>11} {str(stable):>7}  {ppls}", flush=True)
    print(f"\npaired ΔPPL vs base: "
          f"{json.dumps(table['paired_diff'], ensure_ascii=False)}", flush=True)
    print(f"verdict: {table['verdict']}  ({table['pre_registered_rule']})",
          flush=True)
    print(f"⛔ {DISCIPLINE}", flush=True)


def run_screen(args, device):
    """4 配置 × N seeds; 已有 row JSON 即跳过 (断点续跑)。"""
    corpus = build_corpus(args)
    rows = []
    for name in CONFIGS:
        for seed in args.seeds:
            path = row_path(args, name, seed)
            if os.path.exists(path):
                rows.append(json.load(open(path)))
                print(f"[skip] {name} seed {seed}", flush=True)
                continue
            print(f"\n=== {name} (seed {seed}) ===", flush=True)
            row = run_cell(name, seed, corpus, args, device)
            json.dump(row, open(path, "w"), indent=2)
            rows.append(row)
    return rows


def run_cell(name, seed, corpus, args, device):
    """一个 (config, seed) 格子: 同 seed 构建 → 训练 → held-out PPL。"""
    torch.manual_seed(seed)
    model, n_params = build_model(name, args, device)
    train_c, test_c = corpus
    info = train_model(model, train_c, args, device)
    ppl = eval_ppl(model, test_c, args.eval_chunks, args.batch, device)
    print(f"  → val_ppl {ppl:.2f} (stable={info['stable']}, "
          f"{n_params/1e6:.1f}M params)", flush=True)
    del model
    return {"config": name, "seed": seed, "params": n_params,
            "stable": info["stable"], "steps": info["steps"],
            "final_loss": info["final_loss"], "val_ppl": ppl,
            "screening_only": True}


def eval_ppl(model, chunks, eval_chunks, batch, device):
    """Held-out PPL, 只用纯 next-token CE (lm_loss), fp32 无 autocast。"""
    model.eval()
    nll, ntok = 0.0, 0
    with torch.no_grad():
        stop = min(len(chunks), eval_chunks or len(chunks))
        for i in range(0, stop, batch):
            ids = chunks[i:i + batch].to(device)
            n = ids.shape[0] * (ids.shape[1] - 1)
            nll += model(ids, labels=ids)["lm_loss"].item() * n
            ntok += n
    mean = nll / ntok if ntok else float("nan")
    return math.exp(mean) if math.isfinite(mean) and mean < 709 else float("inf")


def train_step(model, ids, opt):
    """单个 optimizer step (fwd/bwd/clip/step); 返回 (loss, finite)。"""
    opt.zero_grad(set_to_none=True)
    loss = model(ids, labels=ids)["loss"]
    if not torch.isfinite(loss):
        return float("nan"), False
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    return loss.item(), True


def save_report(args, rows, table):
    out = {"schema": "modern_trunk_screen/1", "screening_only": True,
           "discipline": DISCIPLINE, "p0_recipe": P0_RECIPE,
           "steps": args.steps, "seeds": args.seeds, "smoke": args.smoke,
           "rows": rows, "verdict": table}
    path = os.path.join(args.out_dir, f"modern_trunk_screen_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n[saved] {path}", flush=True)


def build_corpus(args):
    """P0 口径语料 — 直接复用 scaling_comparison.build_chunks。"""
    from transformers import AutoTokenizer
    from benchmarks.scaling_comparison import build_chunks
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    train_c = build_chunks(tok, "train", args.seq_len, args.wikitext,
                           max_tokens=args.train_token_cap)
    test_c = build_chunks(tok, "test", args.seq_len, args.wikitext)
    return train_c, test_c


def apply_smoke(args):
    args.steps, args.seq_len, args.batch = 30, 128, 2
    args.d_model, args.n_layers = 104, 2
    args.eval_chunks, args.log_every = 4, 10
    args.wikitext, args.train_token_cap = "wikitext-2-raw-v1", 200_000


def row_path(args, name, seed):
    safe = name.replace("+", "plus")
    return os.path.join(args.out_dir,
                        f"modern_trunk_screen_{safe}_{args.tag}_s{seed}.json")


if __name__ == "__main__":
    main()
