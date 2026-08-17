#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_gpu.sh — one-command launcher for the grid-cell emergence GPU run.
#
# Target platforms: Lightning AI Studio, SageMaker Studio Lab, or any single-GPU
# Linux box with a CUDA-capable torch. Validated CPU-smoke locally; this script
# just wires the recommended hyperparameters for a *real* emergence run on GPU.
#
# What it does:
#   1. (optional) installs the minimal deps (torch/numpy/matplotlib).
#   2. runs run_grid_cell.py, which self-clones the repo for the MT-LNN core and
#      trains BOTH a GRU baseline and the MT-LNN recurrent core as path
#      integrators, then scores hexagonal grid emergence.
#
# Every GC_* hyperparameter is overridable from the environment, e.g.:
#   GC_STEPS=3000 ./run_gpu.sh        # quicker first pass
#   GC_PLACE_DOG=0 ./run_gpu.sh       # ablate the DoG emergence trigger
#
# Usage:
#   bash kaggle_kernels/grid_cell_emergence/run_gpu.sh [--install]
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Optional dependency install (pass --install on a fresh environment) ---
if [[ "${1:-}" == "--install" ]]; then
  echo "[run_gpu] installing minimal deps ..."
  pip install -q -r "${HERE}/requirements-gpu.txt"
fi

# --- 2. Recommended hyperparameters for grid-cell EMERGENCE -------------------
# These are the levers that actually decide whether hexagonal grids appear.
# Defaults below are tuned for a clear emergence run on a single mid-range GPU
# (T4 / L4 / P100). Override any of them from the environment.
#
#   GC_PLACE_DOG=1        difference-of-Gaussians place target — THE emergence
#                         trigger (Sorscher 2019). Without it grids rarely form.
#   GC_READOUT_DROPOUT    Banino-style readout regulariser that pushes the
#                         hidden code toward periodic (grid) tuning.
#   GC_STEPS              training iterations. 3000 = quick pass; 8000 = clearer
#                         grids. Bump higher if the GPU session allows.
#   GC_N_PLACE            number of place-cell targets (spatial-code richness).
#   GC_EVAL_TRAJ          trajectories used to build the rate maps (more = less
#                         noisy grid scores, slower eval).
export GC_PLACE_DOG="${GC_PLACE_DOG:-1}"
export GC_READOUT_DROPOUT="${GC_READOUT_DROPOUT:-0.5}"
export GC_STEPS="${GC_STEPS:-8000}"
export GC_BATCH="${GC_BATCH:-256}"
export GC_SEQ_LEN="${GC_SEQ_LEN:-180}"
export GC_N_PLACE="${GC_N_PLACE:-512}"
export GC_PLACE_SIGMA="${GC_PLACE_SIGMA:-0.12}"
export GC_HIDDEN="${GC_HIDDEN:-256}"
export GC_N_BINS="${GC_N_BINS:-32}"
export GC_EVAL_TRAJ="${GC_EVAL_TRAJ:-2000}"
export GC_LR="${GC_LR:-1e-3}"
export GC_WD="${GC_WD:-1e-4}"
export GC_SEED="${GC_SEED:-0}"

# Output directory (figures + grid_cell_metrics.json land here).
export WORK_DIR="${WORK_DIR:-${HERE}/../../artifacts/gridcell_gpu}"
mkdir -p "${WORK_DIR}"

echo "[run_gpu] WORK_DIR=${WORK_DIR}"
echo "[run_gpu] DoG=${GC_PLACE_DOG} dropout=${GC_READOUT_DROPOUT} steps=${GC_STEPS} batch=${GC_BATCH} n_place=${GC_N_PLACE} eval_traj=${GC_EVAL_TRAJ}"

# --- 3. Launch ----------------------------------------------------------------
# The script self-clones the repo into ${WORK_DIR}/M1 for the MT-LNN core, so it
# is self-contained; PYTHONPATH is set as a belt-and-suspenders fallback.
PYTHONPATH="${HERE}/../..:${PYTHONPATH:-}" python "${HERE}/run_grid_cell.py"

echo "[run_gpu] done -> ${WORK_DIR}/grid_cell_metrics.json"
