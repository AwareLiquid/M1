# Grid-cell emergence — GPU run launch kit

Run the path-integration experiment on a real GPU and score whether the MT-LNN
recurrent core (vs. a GRU baseline) develops **hexagonal grid cells** — the
Banino/Sorscher "brain-like spatial code from motion" result.

Local CPU only validates the plumbing; **grid emergence needs thousands of
training steps on a GPU**. This kit wires the recommended hyperparameters so you
can launch in one command on a free/cheap GPU session.

## What you get

| File | Purpose |
|------|---------|
| `run_grid_cell.py` | The experiment (self-clones the repo for the MT-LNN core). |
| `run_gpu.sh` | One-command launcher with emergence-tuned `GC_*` defaults. |
| `requirements-gpu.txt` | Minimal deps: `torch`, `numpy`, `matplotlib`. |
| `grid_cell_colab.ipynb` | One-click **web Colab** notebook (free T4, no CLI/WSL needed). |

## Zero-setup option: web Colab (recommended on Windows)

Open `grid_cell_colab.ipynb` in Colab (the badge at the top of the notebook, or
`https://colab.research.google.com/github/everest-an/M1/blob/main/kaggle_kernels/grid_cell_emergence/grid_cell_colab.ipynb`),
set `Runtime → Change runtime type → GPU (T4)`, then `Run all`. It clones, trains
both models, scores grid emergence, and renders the figures inline. No terminal,
no WSL — ideal if you are on Windows.

Outputs land in `WORK_DIR` (default `artifacts/gridcell_gpu/`):
`grid_cell_metrics.json`, `gridcells_{GRU,MTLNN}.png`,
`gridscore_hist_{GRU,MTLNN}.png`.

## Recommended hyperparameters (already baked into `run_gpu.sh`)

| Env var | Default | Why |
|---------|---------|-----|
| `GC_PLACE_DOG` | `1` | Difference-of-Gaussians place target — **the** emergence trigger. Off => grids rarely form. |
| `GC_READOUT_DROPOUT` | `0.5` | Banino-style readout regulariser pushing the code toward periodic tuning. |
| `GC_STEPS` | `8000` | Training iterations. `3000` = quick first pass; higher = clearer grids. |
| `GC_N_PLACE` | `512` | Place-cell target richness. |
| `GC_EVAL_TRAJ` | `2000` | Trajectories for rate maps (more = less noisy grid score). |
| `GC_BATCH` / `GC_SEQ_LEN` | `256` / `180` | Throughput vs. memory. |

Read a positive result as: `grid_score_max > 0.3` and `n_units_score_gt_0.3 > 0`
for **MTLNN** (the GRU row is the sanity baseline that the pipeline works).

---

## Lightning AI (Studio)

1. New Studio → attach a GPU (T4/L4 on free credits is plenty).
2. In the Studio terminal:

   ```bash
   git clone --depth 1 https://github.com/everest-an/M1.git
   cd M1
   bash kaggle_kernels/grid_cell_emergence/run_gpu.sh --install
   ```

   (`--install` only needed the first time; the base image usually already has a
   CUDA torch, so the install is fast.)

3. When it finishes, grab the figures + JSON from `artifacts/gridcell_gpu/`.

## SageMaker Studio Lab

Free tier gives ~4h GPU sessions — enough for an `8000`-step run.

1. Start a **GPU** runtime → open a terminal.
2. Same three commands:

   ```bash
   git clone --depth 1 https://github.com/everest-an/M1.git
   cd M1
   bash kaggle_kernels/grid_cell_emergence/run_gpu.sh --install
   ```

3. If the session is tight on time, start smaller and scale up:

   ```bash
   GC_STEPS=3000 bash kaggle_kernels/grid_cell_emergence/run_gpu.sh
   ```

## Quick ablations

```bash
# Is DoG really the trigger? Turn it off and watch grids vanish.
GC_PLACE_DOG=0 bash kaggle_kernels/grid_cell_emergence/run_gpu.sh

# Longer run for publication-quality grids.
GC_STEPS=15000 GC_EVAL_TRAJ=4000 bash kaggle_kernels/grid_cell_emergence/run_gpu.sh
```

## Notes

- `run_grid_cell.py` clones the repo into `WORK_DIR/M1` for the MT-LNN core, so
  it is self-contained; the clone uses `--depth 1` of `main` (which now carries
  the `inputs_embeds` entry point the MT-LNN integrator needs).
- On Pascal GPUs (P100, sm_60) the script auto-reinstalls a Pascal-compatible
  `torch==2.6.0+cu124` and re-execs — no action needed.
- The MT-LNN core (model B) is wrapped in try/except: any model-API hiccup on a
  node still leaves you the GRU baseline + a recorded error in the JSON.
