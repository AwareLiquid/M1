"""
modal_train_m2.py -- Run M2 v2.0 pretraining on Modal T4 GPU.

Usage:
    modal run modal_train_m2.py              # 5000 steps ~7h
    modal run modal_train_m2.py --steps 500  # quick smoke test
"""
import os
import modal

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------
app = modal.App("awareliquid-m2")

# Build container image with all deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "datasets",
        "transformers",
        "tokenizers",
        "tqdm",
        "einops",
        "wandb",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/AwareLiquid/M1.git /opt/M1"
    )
)

# Persistent volume to save checkpoints across runs
volume = modal.Volume.from_name("awareliquid-m2-vol", create_if_missing=True)
CKPT_MOUNT = "/checkpoints"
DATA_MOUNT  = "/data"


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 10,        # 10h hard limit
    volumes={
        CKPT_MOUNT: volume,
        DATA_MOUNT: volume,       # data & checkpoints share the same volume
    },
    _allow_background_volume_commits=True,
)
def train(steps: int = 5000, batch: int = 4, grad_accum: int = 4):
    import os, subprocess, sys, shutil, json, time
    sys.path.insert(0, "/opt/M1")
    os.chdir("/opt/M1")

    print("=" * 70)
    print(f"AwareLiquid M2 -- Modal T4 | steps={steps} batch={batch} "
          f"grad_accum={grad_accum}")
    print("=" * 70)

    # Verify GPU
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    mem = (torch.cuda.get_device_properties(0).total_memory / 1e9
           if torch.cuda.is_available() else 0)
    print(f"[gpu] {dev}  sm_{cap[0]}{cap[1]}  {mem:.1f} GB")

    # Data directory inside the volume
    data_dir = os.path.join(DATA_MOUNT, "wikitext103")
    ckpt_dir = os.path.join(CKPT_MOUNT, "m2")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Tokenise WikiText-103 (skips if already done via volume)
    if not os.path.exists(os.path.join(data_dir, "meta.json")):
        print("\n[data] tokenising WikiText-103 ...")
        subprocess.run(
            [sys.executable, "prepare_data.py",
             "--dataset", "wikitext",
             "--config",  "wikitext-103-raw-v1",
             "--tokenizer", "gpt2",
             "--out_dir", data_dir],
            check=True,
        )
    else:
        meta = json.load(open(os.path.join(data_dir, "meta.json")))
        print(f"\n[data] reusing tokenised data  "
              f"({meta.get('train_tokens', 0):,} train tokens)")

    # Resume detection
    resume_args = []
    for candidate in [os.path.join(ckpt_dir, "last.pt"),
                      os.path.join(ckpt_dir, "final.pt")]:
        if os.path.exists(candidate):
            ck = torch.load(candidate, map_location="cpu", weights_only=False)
            resume_args = ["--resume", candidate]
            print(f"[resume] {candidate}  step={ck.get('step','?')}  "
                  f"loss={ck.get('loss','?')}")
            break
    else:
        print("[resume] fresh start")

    # Train
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    metrics_path = os.path.join(CKPT_MOUNT, "metrics.jsonl")

    cmd = [
        sys.executable, "train.py",
        "--data_dir",       data_dir,
        "--ckpt_dir",       ckpt_dir,
        "--metrics_jsonl",  metrics_path,
        "--metrics_every",  "50",
        "--wandb_run_name", "awareliquid-m2-modal",
        # 125M architecture
        "--d_model",    "832",
        "--n_layers",   "12",
        "--n_heads",    "13",
        "--n_kv_heads", "1",
        "--seq_len",    "512",
        # schedule
        "--batch",          str(batch),
        "--grad_accum",     str(grad_accum),
        "--lr",             "6e-4",
        "--warmup_steps",   "120",
        "--steps",          str(steps),
        "--log_every",      "50",
        "--eval_every",     "400",
        "--eval_batches",   "40",
        "--save_every",     "400",
        # v2.0 modules
        "--competitive_gwtb", "--n_bids", "3",
        "--world_model", "--world_model_weight", "0.01",
        "--world_model_grad_clip", "1.0",
        "--hebbian", "--hebbian_lr", "1e-4",
    ] + resume_args

    print("\n[train] " + " ".join(cmd[:12]) + " ...")
    subprocess.run(cmd, check=True)

    # Slim serve.pt
    final = os.path.join(ckpt_dir, "final.pt")
    serve  = os.path.join(ckpt_dir, "serve.pt")
    last   = os.path.join(ckpt_dir, "last.pt")
    if os.path.exists(final):
        shutil.copyfile(final, last)
        ck = torch.load(final, map_location="cpu", weights_only=False)
        slim = {"config": ck["config"], "model_state": ck["model_state"],
                "step": ck.get("step"), "loss": ck.get("loss")}
        torch.save(slim, serve)
        sz = os.path.getsize(serve) / 1e6
        print(f"\n[done] serve.pt  {sz:.0f} MB | "
              f"step {slim['step']} | loss {slim['loss']:.4f}")
        print(f"[done] Volume path: {serve}")
    volume.commit()
    print("[done] Volume committed -- run download_checkpoint() to get files")


# ---------------------------------------------------------------------------
# Download helper -- run locally to pull serve.pt from the volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={CKPT_MOUNT: volume},
)
def list_checkpoints():
    """List checkpoint files in the Modal volume."""
    import os
    for root, dirs, files in os.walk(CKPT_MOUNT):
        for f in files:
            path = os.path.join(root, f)
            sz = os.path.getsize(path) / 1e6
            print(f"  {path}  ({sz:.1f} MB)")


@app.local_entrypoint()
def main(steps: int = 5000, smoke: bool = False):
    """
    Entry point.
      modal run modal_train_m2.py              # full 5000-step run
      modal run modal_train_m2.py --smoke      # 50-step smoke test
      modal run modal_train_m2.py --steps 500  # custom steps
    """
    if smoke:
        steps = 50
    print(f"Submitting M2 training: {steps} steps on T4 GPU")
    train.remote(steps=steps)
    print("\nTraining complete. Listing checkpoints:")
    list_checkpoints.remote()
