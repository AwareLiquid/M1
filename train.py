"""
train.py — Training script for MT-LNN.

Key features:
  - Memory-mapped binary token streams (low-RAM, supports massive datasets)
  - SDPA / Flash-Attention via MicrotubuleAttention
  - Optional torch.compile (--compile)
  - Optional Weights & Biases logging (--wandb)
  - Separate LR groups for τ, γ, polarity, lateral coupling
  - MT diagnostics streamed at every eval

Pipeline:
    1) python prepare_data.py    (one-time tokenisation)
    2) python train.py           (uses data/{train,validation}.bin)
    3) python train.py --dummy   (no dataset needed for smoke tests)
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import make_param_groups, WarmupCosineScheduler, save_checkpoint


# ---------------------------------------------------------------------------
# Memory-mapped token dataset
# ---------------------------------------------------------------------------

class BinDataset(Dataset):
    """
    Reads a flat uint16 token stream from disk via numpy.memmap.
    Each __getitem__ returns a (seq_len+1) window starting at a random offset
    so we don't waste tokens on a fixed grid.
    """

    def __init__(self, bin_path: str, seq_len: int, stride: int = None):
        self.path = bin_path
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.stride = stride or seq_len   # non-overlapping windows by default

    def __len__(self):
        return max(1, (len(self.data) - self.seq_len - 1) // self.stride)

    def __getitem__(self, idx):
        # Random offset within the stride bucket → mild data augmentation
        base = idx * self.stride
        # Clip so we don't run off the end
        max_start = len(self.data) - self.seq_len - 1
        start = min(base + np.random.randint(0, self.stride), max_start)
        chunk = self.data[start: start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class DummyDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 200):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return ids[:-1], ids[1:]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for i, (inp, lbl) in enumerate(val_loader):
        if i >= max_batches:
            break
        inp, lbl = inp.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
        out = model(inp, labels=lbl)
        total_loss += out["loss"].item()
        n += 1
    model.train()
    return math.exp(min(total_loss / max(n, 1), 20.0))   # PPL, clipped


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.dummy:
        cfg_kwargs = dict(vocab_size=args.vocab_size or 1000)
        train_ds = DummyDataset(cfg_kwargs["vocab_size"], args.seq_len, n_samples=200)
        val_ds   = DummyDataset(cfg_kwargs["vocab_size"], args.seq_len, n_samples=20)
    else:
        meta_path = os.path.join(args.data_dir, "meta.json")
        assert os.path.exists(meta_path), \
            f"No meta.json at {meta_path}. Run `python prepare_data.py` first."
        meta = json.load(open(meta_path))
        cfg_kwargs = dict(vocab_size=meta["vocab_size"])
        train_ds = BinDataset(os.path.join(args.data_dir, "train.bin"), args.seq_len)
        val_path = os.path.join(args.data_dir, "validation.bin")
        if not os.path.exists(val_path):
            val_path = os.path.join(args.data_dir, "test.bin")
        val_ds = BinDataset(val_path, args.seq_len)
        print(f"Train tokens: {len(train_ds.data):,}  Val tokens: {len(val_ds.data):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    config = MTLNNConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_head=args.d_model // args.n_heads,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        # v2.0 modules — all default False to preserve existing behaviour
        gwtb_n_heads=args.gwtb_n_heads,
        use_competitive_gwtb=args.competitive_gwtb,
        n_competitive_bids=args.n_bids,
        use_world_model=args.world_model,
        world_model_loss_weight=args.world_model_weight,
        use_hebbian=args.hebbian,
        hebbian_lr=args.hebbian_lr,
        **cfg_kwargs,
    )
    model = MTLNNModel(config).to(device)
    n_params = model.get_num_params()
    if args.train_target_head:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.startswith("target_"):
                param.requires_grad = True
        print("Direct target mode: backbone frozen; training target_queries/target_norm/target_head only.")
    print(f"Parameters: {n_params/1e6:.1f}M  (config: {config.d_model}d × {config.n_layers}L × {config.n_heads}H, GQA={config.n_kv_heads})")

    # torch.compile for speed (skip on CPU since the gain isn't there)
    if args.compile and device == "cuda":
        print("Compiling model with torch.compile …")
        model = torch.compile(model, mode="default")

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    param_groups = make_param_groups(model, base_lr=args.lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, args.steps,
                                       min_lr=args.lr * 0.1)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={**vars(args), "n_params_M": n_params / 1e6},
            )
        except Exception as e:
            print(f"[W&B] init failed: {e}; falling back to console logging.")
            wandb_run = None

    def log(metrics: dict, step: int, histograms: dict = None):
        if wandb_run is not None:
            payload = dict(metrics)
            if histograms is not None:
                import wandb
                for k, v in histograms.items():
                    payload[f"hist/{k}"] = wandb.Histogram(v.numpy())
            wandb_run.log(payload, step=step)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    step = 0
    accum_loss_sum = 0.0
    accum_count = 0
    t0 = time.time()
    model.train()

    while step < args.steps:
        for inp, lbl in train_loader:
            if step >= args.steps:
                break
            inp = inp.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                if args.train_target_head or args.target_loss_weight > 0.0:
                    direct_len = args.direct_target_len
                    direct_labels = lbl[:, -direct_len:].contiguous()
                    out = model(
                        inp,
                        labels=None if args.train_target_head else lbl,
                        direct_target_labels=direct_labels,
                        target_len=direct_len,
                        return_target_logits=True,
                    )
                    raw_loss = out["target_loss"] if args.train_target_head else (
                        out["loss"] + args.target_loss_weight * out["target_loss"]
                    )
                else:
                    out = model(inp, labels=lbl)
                    raw_loss = out["loss"]
                loss = raw_loss / args.grad_accum

            # Track auxiliary losses for logging (detached from graph)
            _wm_loss   = out.get("world_model_loss")
            _hebb_loss = out.get("hebbian_loss")

            scaler.scale(loss).backward()
            accum_loss_sum += loss.item() * args.grad_accum
            accum_count += 1

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                # Targeted clip on the world-model head first: its auxiliary
                # self-supervised loss must never destabilise the LM objective.
                if getattr(model, "world_model_head", None) is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.world_model_head.parameters(), args.world_model_grad_clip)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1

            # ---- Periodic logging ----
            if step % args.log_every == 0:
                avg_loss = accum_loss_sum / max(accum_count, 1)
                ppl = math.exp(min(avg_loss, 20.0))
                tps = (args.log_every * args.batch * args.seq_len) / max(time.time() - t0, 1e-3)
                # Build aux loss suffix for display
                aux_parts = []
                if _wm_loss is not None:
                    aux_parts.append(f"wm={_wm_loss.item():.4f}")
                if _hebb_loss is not None:
                    aux_parts.append(f"hebb={_hebb_loss.item():.4f}")
                aux_str = " | " + " ".join(aux_parts) if aux_parts else ""
                msg = (f"step {step:6d} | loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                       f"lr {scheduler.current_lr:.2e} | {tps:.0f} tok/s{aux_str}")
                print(msg)
                log_dict = {"train/loss": avg_loss, "train/ppl": ppl,
                            "train/lr": scheduler.current_lr, "train/tokens_per_sec": tps}
                if _wm_loss is not None:
                    log_dict["train/world_model_loss"] = _wm_loss.item()
                if _hebb_loss is not None:
                    log_dict["train/hebbian_loss"] = _hebb_loss.item()
                log(log_dict,
                    step=step)
                accum_loss_sum, accum_count = 0.0, 0
                t0 = time.time()

            # ---- Eval + diagnostics ----
            if step % args.eval_every == 0:
                val_ppl = evaluate(model, val_loader, device, max_batches=args.eval_batches)
                # MT diagnostics (peel torch.compile if needed)
                base_model = getattr(model, "_orig_mod", model)
                diag = base_model.get_mt_diagnostics()
                hist = base_model.get_mt_histograms()
                print(f"  val PPL: {val_ppl:.2f} | "
                      f"τ={diag['tau_mean']:.2f}±{diag['tau_std']:.2f} "
                      f"[{diag['tau_min']:.2f}, {diag['tau_max']:.2f}] | "
                      f"γ={diag['gamma_mean']:.3f} | "
                      f"polarity_std={diag['polarity_std']:.3f} | "
                      f"rmc_gate={diag['rmc_gate_mean']:.3f} | "
                      f"collapse_gate={diag['collapse_gate_last']:.3f} | "
                      f"coherence_scale={diag['coherence_scale']:.3f}")
                log({"val/ppl": val_ppl, **{f"mt/{k}": v for k, v in diag.items()}},
                    step=step, histograms=hist)
                t0 = time.time()  # don't penalise tok/s for eval time

            # ---- Checkpointing ----
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:06d}.pt")
                base_model = getattr(model, "_orig_mod", model)
                save_checkpoint(base_model, optimizer, step, 0.0, ckpt_path, config)
                print(f"  saved {ckpt_path}")

    # Final save
    base_model = getattr(model, "_orig_mod", model)
    save_checkpoint(base_model, optimizer, step, 0.0,
                    os.path.join(args.ckpt_dir, "final.pt"), config)
    print(f"Training complete. Final checkpoint: {args.ckpt_dir}/final.pt")
    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Model
    # 125M defaults — Tensor-Core aligned:
    # d_model=832 = 13 × 64 means d_proto=d_head=64 (8-multiple). n_heads=13
    # makes each attention head correspond to one protofilament.
    p.add_argument("--d_model",       type=int,   default=832)
    p.add_argument("--n_layers",      type=int,   default=12)
    p.add_argument("--n_heads",       type=int,   default=13)
    p.add_argument("--n_kv_heads",    type=int,   default=1)
    # Start with 512; once converged, fine-tune at 2048+ — RoPE + MT bias
    # generalise well past the training length.
    p.add_argument("--seq_len",       type=int,   default=512)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--gwtb_n_heads",  type=int,   default=4,
                   help="Number of GWTB workspace attention heads (must divide d_model//gwtb_ratio)")
    # Training — defaults chosen for a 125M model on a single A100/3090.
    # Global batch = batch * grad_accum * #GPUs. With batch=8 and grad_accum=64
    # we hit the recommended global batch of 512 (critical for stable τ
    # learning on the LNN side).
    p.add_argument("--batch",         type=int,   default=8)
    p.add_argument("--grad_accum",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=6e-4)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--warmup_steps",  type=int,   default=2000)
    p.add_argument("--steps",         type=int,   default=50000)
    # Logging / IO
    p.add_argument("--log_every",     type=int,   default=100)
    p.add_argument("--eval_every",    type=int,   default=500)
    p.add_argument("--eval_batches",  type=int,   default=50)
    p.add_argument("--save_every",    type=int,   default=2000)
    p.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    p.add_argument("--data_dir",      type=str,   default="data")
    p.add_argument("--num_workers",   type=int,   default=2)
    # Switches
    p.add_argument("--compile",       action="store_true", help="Enable torch.compile")
    p.add_argument("--wandb",         action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb_project", type=str,   default="mt-lnn")
    p.add_argument("--wandb_run_name", type=str,  default=None)
    p.add_argument("--dummy",         action="store_true", help="Use random data")
    p.add_argument("--vocab_size",    type=int,   default=None,
                                       help="Override vocab_size (only with --dummy)")
    p.add_argument("--train_target_head", action="store_true",
                   help="Freeze the backbone and train only the direct target extraction head")
    p.add_argument("--direct_target_len", type=int, default=4,
                   help="Number of target slots supervised by the direct extraction head")
    p.add_argument("--target_loss_weight", type=float, default=0.0,
                   help="Optional auxiliary direct-target loss weight during normal LM training")
    # ---- v2.0 modules (all off by default) ----
    p.add_argument("--competitive_gwtb", action="store_true",
                   help="[Phase A] Enable CompetitiveGWTBLayer: K-bid workspace competition")
    p.add_argument("--n_bids", type=int, default=3,
                   help="[Phase A] Number of specialist bids competing for workspace (default 3)")
    p.add_argument("--world_model", action="store_true",
                   help="[Phase C] Enable PredictiveStateHead: next-state self-supervised loss")
    p.add_argument("--world_model_weight", type=float, default=0.01,
                   help="[Phase C] Weight of world-model MSE loss (default 0.01)")
    p.add_argument("--world_model_grad_clip", type=float, default=1.0,
                   help="[Phase C] Separate grad-norm clip for the world-model head (default 1.0)")
    p.add_argument("--hebbian", action="store_true",
                   help="[Phase D] Enable HebbianRegularizer: co-activation loss term")
    p.add_argument("--hebbian_lr", type=float, default=1e-4,
                   help="[Phase D] Base Hebbian learning rate α (default 1e-4)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
