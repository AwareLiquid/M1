"""Small-scale text A/B: does selective_decay improve LM convergence?

WikiText-2, matched tiny model (d_model=104, 2 layers, ~400K params),
2000 steps, 3 seeds, fp32 CPU. Compare:
  - stock (input-independent decay)
  - selective_decay (input-dependent transition)
Same protocol as the parity probe: fresh model per seed, report val PPL.
This is a DIRECTION probe, not a paper table — if selective helps here,
the 125M run on Kaggle is worth the GPU time; if not, we save it.

Windows note: torch 2.5.1 + transformers 5.x + pyarrow have a DLL/OpenMP
conflict — pyarrow read_table crashes (0xC0000005) once torch is loaded.
The parquet was converted to plain .txt by a standalone pyarrow pass
(data/wikitext2_{train,val}.txt); this script reads text only.
"""
import sys, time, json, os
sys.path.insert(0, 'E:/M1')
import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from transformers import AutoTokenizer


DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WT2_TRAIN_TXT = os.path.join(DATA, "wikitext2_train.txt")
WT2_VAL_TXT = os.path.join(DATA, "wikitext2_val.txt")


def build(selective_decay, seed):
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=50257, max_seq_len=128, d_model=104, n_layers=2,
        n_heads=4, n_kv_heads=2, d_head=26, dropout=0.0,
        attention_dropout=0.0, gwtb_n_heads=1, tie_embeddings=True,
        selective_decay=selective_decay,
    )
    return MTLNNModel(cfg)


def chunks(tok, text, seq_len=128, max_tokens=2_000_000):
    ids = tok(text, truncation=False)["input_ids"][:max_tokens]
    n = len(ids) // seq_len
    return [ids[i*seq_len:(i+1)*seq_len] for i in range(n)]


def load_text(txt_path, n_lines=None):
    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()
    if n_lines is not None:
        lines = lines[:n_lines]
    return "".join(lines)


def train_eval(sel, seed, steps=2000, batch=16, lr=3e-4):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("gpt2")
    full_train = load_text(WT2_TRAIN_TXT)
    lines = full_train.splitlines()
    n_train = len(lines) - 3000
    train_c = chunks(tok, "\n".join(lines[:n_train]), max_tokens=3_000_000)
    val_c = chunks(tok, "\n".join(lines[n_train:]), max_tokens=400_000)
    print(f"  [seed {seed}] sel={sel} train_chunks={len(train_c)} val_chunks={len(val_c)}", flush=True)

    m = build(sel, seed)
    n_params = m.get_num_params()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed)
    ckpt_dir = os.path.join(DATA, "text_ab_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"sel{int(sel)}_s{seed}.pt")
    start_step = 0
    if os.path.exists(ckpt_path) and "--resume" in sys.argv:
        st = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step = st["step"] + 1
        # load_state_dict restored the OLD lr (≈0 at end of the original
        # 2000-step cosine). Reset to the configured lr BEFORE creating the
        # new scheduler — CosineAnnealingLR captures base_lrs at __init__.
        for g in opt.param_groups:
            g["lr"] = lr
        # Rebuild scheduler for the NEW total budget, resuming mid-curve:
        # cosine LR continues from the current point and anneals to 0 at the
        # new `steps`, so a longer-budget continuation is meaningful (the old
        # T_max=2000 would have LR ≈ 0 already).
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        sched.last_epoch = start_step - 1
        print(f"  [seed {seed}] sel={sel} resumed from step {start_step}, "
              f"lr now {opt.param_groups[0]['lr']:.2e}", flush=True)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for step in range(steps):
        if step < start_step:
            continue
        idx = torch.randint(0, len(train_c), (batch,), generator=rng)
        ids = torch.tensor([train_c[i] for i in idx])
        out = m(ids, labels=ids)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 400 == 0:
            print(f"    [seed {seed}] sel={sel} step {step} loss {loss.item():.3f}", flush=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": steps - 1}, ckpt_path)

    # val PPL
    m.eval()
    total_nll = 0.0; total_tok = 0
    with torch.no_grad():
        for i in range(0, len(val_c) - batch, batch):
            ids = torch.tensor(val_c[i:i+batch])
            out = m(ids, labels=ids)
            total_nll += out["loss"].item() * ids.numel()
            total_tok += ids.numel()
    ppl = float(torch.tensor(total_nll / total_tok).exp().item())
    print(f"  [seed {seed}] selective={sel} params={n_params} val_ppl={ppl:.3f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    return {"seed": seed, "selective": sel, "params": n_params,
            "val_ppl": ppl, "steps": steps}


if __name__ == "__main__":
    OUT = "benchmarks/results/text_selective_ab.jsonl"
    # CLI: --steps N overrides the budget (default 2000); --resume continues
    # from saved checkpoints with the scheduler rebuilt for the new budget.
    steps = 2000
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
    print(f"steps={steps} resume={('--resume' in sys.argv)}", flush=True)
    rows = []
    for sel in [False, True]:
        for seed in [0, 1, 2]:
            rows.append(train_eval(sel, seed, steps=steps))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for sel in [False, True]:
        vals = [r["val_ppl"] for r in rows if r["selective"] == sel]
        print(f"selective={sel}: val_ppl {sum(vals)/len(vals):.3f} ± {torch.tensor(vals).std().item():.3f}")
    print(f"appended to {OUT}")
