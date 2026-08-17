# Colab Run — AwareLiquid-Tiny (coherent small-model demo)

Notebook: [`colab/train_tiny_colab.ipynb`](../../colab/train_tiny_colab.ipynb)

Trains a **~49M MT-LNN** (d_model=416, 6 layers) on **TinyStories** (gpt2 BPE) for a
genuinely coherent public demo. Free Colab **T4** is sufficient. Everything heavy
(tokenised `*.bin`, checkpoints) lives on **Google Drive**, so a disconnect never
loses more than ~1000 steps — just re-run and it resumes.

> Why this exists: a Kaggle re-train was blocked because every available API token
> resolves to the same exhausted `muningan` account (30 h/week GPU quota used up).
> Colab only needs **one** Google account and gives a fresh, independent GPU budget.

## 1. Open in Colab

1. <https://colab.research.google.com> → **File → Upload notebook** → pick
   `colab/train_tiny_colab.ipynb` from this repo (or open it from GitHub:
   File → Open notebook → GitHub → `everest-an/M1` → `colab/train_tiny_colab.ipynb`).
2. **Runtime → Change runtime type → Hardware accelerator → GPU** (T4 is fine).

## 2. Run

Click **Runtime → Run all**. Cells in order:

| Cell | Purpose | Time |
|---|---|---|
| 1 | GPU sanity (asserts CUDA, prints device/capability) | <5 s |
| 2 | mount Google Drive → `MyDrive/awareliquid_tiny/` | ~10 s (one OAuth click) |
| 3 | `git clone` GitHub `main` into `/content/M1` (always has the label fix) | ~10 s |
| 4 | pip install `datasets transformers tokenizers tqdm einops` (torch pinned) | ~2 min |
| 5 | tokenise TinyStories → `data/*.bin` on Drive (**skipped on re-runs**) | ~10–20 min first time |
| 6 | detect newest checkpoint on Drive for resume | <1 s |
| 7 | **train** (~49M, v2 signature modules ON; saves to Drive every 1000 steps) | several hours |
| 8 | write slim **`serve.pt`** + trigger browser download | ~30 s |

> The free T4 may disconnect before 20000 steps finish. That's expected — just
> **Run all again**. Cells 2/5 are cached on Drive and cell 6 resumes training
> from the newest `ckpt_*.pt`.

## 3. Get the trained model

After cell 8, the file is at:

```
MyDrive/awareliquid_tiny/checkpoints/serve.pt
```

This is a **slim, server-loadable** checkpoint (config + model_state, no optimizer
— ~1/3 the size of `final.pt`). It's the drop-in for `serve/server.py` and the demo.
Cell 8 also tries a direct browser download. Download it and hand it over for the
hot-swap.

## 4. Tunables (env vars, set before cell 7)

| Var | Default | Notes |
|---|---|---|
| `T_STEPS` | `20000` | total optimizer steps |
| `T_BATCH` | `12` | per-step batch; T4 15GB ≈ P100 16GB |
| `T_GRAD_ACCUM` | `4` | → global batch 48 |

Set e.g. `os.environ['T_STEPS'] = '10000'` in a cell above cell 7 for a shorter run.

## 5. Notes & gotchas

- **The label fix is baked in via `git clone`** — the notebook pulls GitHub `main`
  (commit `b027bbf` onward), so `BinDataset`/`DummyDataset` return aligned labels
  and the model trains the true next-token objective (not the old skip-one bug).
- **Drive checkpoints are ~416 MB each** (they include optimizer state for resume).
  Only `last.pt` is needed to resume; you can delete older `ckpt_*.pt` to save Drive
  space once a run is healthy.
- **Do not change the runtime to TPU** — the code path is CUDA-only.
- T4 (sm_75) is supported by stock Colab torch, so — unlike the Kaggle P100 (Pascal,
  sm_60) kernel — **no torch reinstall** is needed.
