#!/usr/bin/env python3
"""Grid-cell emergence -- minimal path-integration experiment (Kaggle GPU).

Scientific question
-------------------
When a recurrent network is trained to PATH-INTEGRATE (turn a stream of
self-motion / velocity inputs into an estimate of its allocentric position,
read out as place-cell activity), do its hidden units spontaneously develop
HEXAGONAL grid firing fields -- the signature of biological medial-entorhinal
grid cells? This is the Cueva & Wei (2018) / Banino et al. (2018, Nature)
result. Grid-cell emergence is a clean, quantitative probe of whether an
architecture learns a *metric, allocentric spatial code* purely from motion --
exactly the "brain-like spatial representation" the MT-LNN roadmap is after.

What this kernel does
---------------------
  1. Generates rat-like 2D random-walk trajectories in a square arena and the
     corresponding velocity inputs + ground-truth place-cell activations.
  2. Trains TWO recurrent path integrators on the same data:
       (A) a vanilla GRU  -- the canonical, known-to-work baseline; validates
           the whole analysis pipeline end-to-end.
       (B) the MT-LNN recurrent core (small config), fed velocity through
           `inputs_embeds` with `use_lnn_recurrence=True`, hidden state tapped
           off `final_norm` via a forward hook, plus a linear place-cell readout.
       (B) is best-effort: wrapped in try/except so a model-API hiccup on the
       Kaggle node never costs us the baseline result.
  3. Analyses the learned hidden units: spatial rate maps -> 2D spatial
     autocorrelogram -> GRID SCORE (6-fold rotational symmetry). Saves the
     top grid cells' rate-map / autocorr figures, a grid-score histogram, and
     a metrics.json summary to /kaggle/working.

Everything writes to /kaggle/working so it lands in the kernel output.
This experiment is independent of the text-LM pretraining and can run in
parallel on its own GPU session.
"""
import json
import math
import os
import subprocess
import sys
import time

t_start = time.time()
WORK = os.environ.get("WORK_DIR", "/kaggle/working")
os.makedirs(WORK, exist_ok=True)

# Hyperparameters (override via env for quick experiments) -------------------
SEED        = int(os.environ.get("GC_SEED", "0"))
ARENA       = float(os.environ.get("GC_ARENA", "2.2"))     # metres, square box
N_PLACE     = int(os.environ.get("GC_N_PLACE", "512"))     # place-cell targets
PLACE_SIGMA = float(os.environ.get("GC_PLACE_SIGMA", "0.12"))  # place-field width (m)
SEQ_LEN     = int(os.environ.get("GC_SEQ_LEN", "180"))     # steps per trajectory
DT          = float(os.environ.get("GC_DT", "0.02"))       # s per step
HIDDEN      = int(os.environ.get("GC_HIDDEN", "256"))      # GRU hidden / readout width
BATCH       = int(os.environ.get("GC_BATCH", "256"))
STEPS       = int(os.environ.get("GC_STEPS", "3000"))      # training iterations
LR          = float(os.environ.get("GC_LR", "1e-3"))
WD          = float(os.environ.get("GC_WD", "1e-4"))       # weight decay (metabolic cost)
N_BINS      = int(os.environ.get("GC_N_BINS", "32"))       # rate-map resolution
EVAL_TRAJ   = int(os.environ.get("GC_EVAL_TRAJ", "2000"))  # trajectories for rate maps
TOPK_FIG    = int(os.environ.get("GC_TOPK_FIG", "16"))     # how many cells to plot

print("=" * 70)
print("Grid-cell emergence -- minimal path-integration experiment")
print("=" * 70)
print(f"[cfg] arena={ARENA}m  n_place={N_PLACE}  seq_len={SEQ_LEN}  hidden={HIDDEN}")
print(f"[cfg] batch={BATCH}  steps={STEPS}  lr={LR}  wd={WD}  n_bins={N_BINS}")

# ---------------------------------------------------------------------------
# 1. Environment: clone repo (for the MT-LNN core) + torch GPU compatibility
# ---------------------------------------------------------------------------
REPO = "https://github.com/AwareLiquid/M1.git"
DIR = os.path.join(WORK, "M1")
HAVE_REPO = True
# Local-run guard: if WORK already sits *inside* a checkout of this repo (e.g.
# WORK_DIR=artifacts/gridcell_local during local smoke testing), cloning the
# repo into ${WORK}/M1 would create a self-referential copy living under the
# repo root -- and a hand-made symlink there (M1 -> repo root) makes os.walk /
# pytest recurse forever. Detect that case and import the enclosing checkout
# directly instead of self-cloning.
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
_work_abs = os.path.abspath(WORK)
_inside_repo = _work_abs == _repo_root or _work_abs.startswith(_repo_root + os.sep)
try:
    if _inside_repo and os.path.isdir(os.path.join(_repo_root, "mt_lnn")):
        # Running from a live checkout: use it, don't clone into ourselves.
        print(f"[env] WORK is inside the repo checkout at {_repo_root}; "
              f"using it directly (no self-clone).")
        sys.path.insert(0, _repo_root)
    else:
        if not os.path.exists(DIR):
            subprocess.run(["git", "clone", "--depth", "1", REPO, DIR], check=True)
        sys.path.insert(0, DIR)
except Exception as e:  # noqa: BLE001
    print(f"[env] repo clone failed ({e}); MT-LNN core (model B) will be skipped.")
    HAVE_REPO = False

import torch  # noqa: E402  (Kaggle pre-installs torch)

# Pascal (sm_60) compatibility: Kaggle's newest torch dropped Pascal kernels but
# still hands out P100s. Reinstall a Pascal-capable torch BEFORE any CUDA op.
_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
_dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"[env] torch {torch.__version__} | device={_dev} | capability={_cap}")
if _cap is not None and _cap[0] < 7:
    print(f"[env] {_dev} (sm_{_cap[0]}{_cap[1]}) unsupported by pre-installed torch "
          f"-- installing Pascal-compatible torch 2.6.0+cu124")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"],
        check=True,
    )
    os.execv(sys.executable, [sys.executable] + sys.argv)  # re-exec with new torch

# einops is a hard dep of the MT-LNN model; install quietly (no-op if present).
try:
    import einops  # noqa: F401
except Exception:  # noqa: BLE001
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops"], check=False)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 2. Trajectory + place-cell data generator
# ---------------------------------------------------------------------------
class TrajectoryGenerator:
    """Rat-like 2D random walk in a square box with smooth, boundary-aware turns.

    Velocity input per step is the 3-vector (speed, sin(heading), cos(heading)) --
    an egocentric, allocentric-free self-motion signal. The network must
    integrate it to recover allocentric position, read out as a population of
    Gaussian place cells (the supervised target).
    """

    def __init__(self, arena, n_place, place_sigma, dt, device):
        self.arena = arena
        self.dt = dt
        self.device = device
        self.border = 0.03 * arena
        self.place_sigma = place_sigma
        # Place-cell target = the single source of truth in mt_lnn.spatial.
        # The DoG (center-surround / Mexican-hat) recipe that triggers hexagonal
        # grid emergence (Sorscher et al. 2019) used to be re-implemented inline
        # here; it now lives ONCE in PlaceCellCode (validated, dog_amp?(0,1)
        # enforced). This script simply parameterises and calls it, so the
        # experiment and the library can never drift apart.
        #   GC_PLACE_DOG=1 -> DoG target (the grid-emergence trigger);
        #   default        -> the proven single-bump Gaussian code.
        use_dog = os.environ.get("GC_PLACE_DOG", "0") == "1"
        try:
            from mt_lnn.spatial import PlaceCellCode
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "TrajectoryGenerator now delegates the place-cell target to "
                "mt_lnn.spatial.PlaceCellCode, but the M1 repo could not be "
                "imported. Ensure the repo clone succeeded (HAVE_REPO)."
            ) from e
        # PlaceCellCode's default centre layout (seed=123, rand(n_place,2)*arena)
        # is byte-identical to the layout this script used before the refactor,
        # so historical runs remain directly comparable.
        self.place_code = PlaceCellCode(
            n_place=n_place, arena=arena, coord_dim=2, sigma=place_sigma,
            mode="dog" if use_dog else "gaussian",
            dog_surround=float(os.environ.get("GC_DOG_SURROUND", "2.0")),
            dog_amp=float(os.environ.get("GC_DOG_AMP", "0.5")),
            dog_temp=float(os.environ.get("GC_DOG_TEMP", "0.05")),
            seed=123,
        ).to(device)
        self.use_dog = use_dog
        self.place_centers = self.place_code.place_centers     # (N_place, 2)

    def _place_activity(self, pos):                        # pos (B, T, 2)
        # (B,T,2) -> (B,T,N_place) softmax target, computed by the shared module.
        return self.place_code(pos)

    def generate(self, batch, seq_len):
        dev = self.device
        a = self.arena
        pos = torch.rand(batch, 2, device=dev) * a
        head = torch.rand(batch, device=dev) * 2 * math.pi
        speed = torch.zeros(batch, device=dev)
        positions = torch.empty(batch, seq_len, 2, device=dev)
        vel_in = torch.empty(batch, seq_len, 3, device=dev)
        for t in range(seq_len):
            # Ornstein-Uhlenbeck-ish speed + random turn; bias turn away from walls.
            speed = (0.7 * speed + 0.3 * (0.13 + 0.08 * torch.randn(batch, device=dev))).clamp(0.0, 0.5)
            dhead = 0.2 * torch.randn(batch, device=dev)
            # Wall avoidance: if near a border, steer back toward the centre.
            to_center = torch.atan2(a / 2 - pos[:, 1], a / 2 - pos[:, 0])
            near_wall = ((pos < self.border) | (pos > a - self.border)).any(-1)
            ang_err = torch.atan2(torch.sin(to_center - head), torch.cos(to_center - head))
            dhead = torch.where(near_wall, 0.5 * ang_err, dhead)
            head = head + dhead
            dx = speed * torch.cos(head) * self.dt * 30.0
            dy = speed * torch.sin(head) * self.dt * 30.0
            pos = torch.stack([pos[:, 0] + dx, pos[:, 1] + dy], dim=-1).clamp(0.0, a)
            positions[:, t] = pos
            vel_in[:, t, 0] = speed
            vel_in[:, t, 1] = torch.sin(head)
            vel_in[:, t, 2] = torch.cos(head)
        place = self._place_activity(positions)
        # initial-position conditioning: the network gets the t0 place code so the
        # task is integration (relative), not absolute localisation from scratch.
        init_place = place[:, 0, :]
        return vel_in, positions, place, init_place


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------
class GRUIntegrator(torch.nn.Module):
    """Canonical baseline: GRU + linear place-cell readout. The hidden layer is
    a low-D bottleneck encouraged (by weight decay) toward an efficient code --
    the regime in which grid cells emerge (Sorscher et al. 2019)."""

    def __init__(self, n_place, hidden):
        super().__init__()
        self.init_h = torch.nn.Linear(n_place, hidden)     # encode t0 place code
        self.rnn = torch.nn.GRU(3, hidden, batch_first=True)
        # Dropout on the recurrent->readout path (Banino et al. 2018): forces a
        # distributed, redundant code and is a known grid-emergence regulariser.
        # Active only in train mode, so the eval-time rate maps see clean units.
        self.drop = torch.nn.Dropout(float(os.environ.get("GC_READOUT_DROPOUT", "0.0")))
        self.readout = torch.nn.Linear(hidden, n_place)

    def forward(self, vel_in, init_place):
        h0 = torch.tanh(self.init_h(init_place)).unsqueeze(0)   # (1,B,H)
        hidden, _ = self.rnn(vel_in, h0)                        # (B,T,H)
        logits = self.readout(self.drop(hidden))                # (B,T,N)
        return logits, hidden


class MTLNNIntegrator(torch.nn.Module):
    """MT-LNN recurrent core as a path integrator.

    Velocity (B,T,3) is projected to d_model and passed as `inputs_embeds` with
    `use_lnn_recurrence=True` so the liquid core threads its hidden state across
    the trajectory (true RNN mode). The post-`final_norm` hidden state (B,T,d) is
    captured with a forward hook and decoded to place-cell logits. The original
    vocab `lm_head` is unused."""

    def __init__(self, model, d_model, n_place, hidden):
        super().__init__()
        self.model = model
        self.vel_proj = torch.nn.Linear(3, d_model)
        self.init_proj = torch.nn.Linear(n_place, d_model)
        # Dropout before the readout MLP -- same Banino-style grid regulariser
        # as the GRU baseline (active only in train mode; eval rate maps clean).
        self.readout = torch.nn.Sequential(
            torch.nn.Dropout(float(os.environ.get("GC_READOUT_DROPOUT", "0.0"))),
            torch.nn.Linear(d_model, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_place),
        )
        self._tap = {}
        # Tap the normalised hidden state that feeds the (unused) lm_head.
        self.model.final_norm.register_forward_hook(
            lambda m, i, o: self._tap.__setitem__("h", o)
        )

    def forward(self, vel_in, init_place):
        emb = self.vel_proj(vel_in)                              # (B,T,d)
        # Inject the initial place code as an additive bias on every step's input
        # (a simple, differentiable way to condition on the start location).
        emb = emb + self.init_proj(init_place).unsqueeze(1)
        self.model(inputs_embeds=emb, use_lnn_recurrence=True)   # logits unused
        hidden = self._tap["h"]                                  # (B,T,d) via hook
        logits = self.readout(hidden)                            # (B,T,N)
        return logits, hidden


def build_mtlnn():
    """Small, valid MT-LNN core. Constraints (validated locally):
      - d_model multiple of n_protofilaments(13)*8 = 104  -> d_proto multiple of 8
      - d_model divisible by n_heads; d_head = d_model // n_heads
      - d_gw = d_model//8 divisible by gwtb_n_heads
    d_model=208 (=13*16) -> d_head=16, d_gw=26 (? gwtb_n_heads=2). ~0.83M params."""
    from mt_lnn.config import MTLNNConfig
    from mt_lnn.model import MTLNNModel
    cfg = MTLNNConfig(
        vocab_size=256, d_model=208, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=16, max_seq_len=max(SEQ_LEN + 8, 256), gwtb_n_heads=2,
        use_world_model=False, use_competitive_gwtb=False, use_hebbian=False,
    )
    return MTLNNModel(cfg), cfg.d_model


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
def train(model, gen, tag):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    losses = []
    for step in range(1, STEPS + 1):
        vel_in, _pos, place, init_place = gen.generate(BATCH, SEQ_LEN)
        logits, _hidden = model(vel_in, init_place)
        # cross-entropy against the (soft) place-cell target distribution.
        logp = torch.log_softmax(logits, dim=-1)
        loss = -(place * logp).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % max(1, STEPS // 20) == 0 or step == 1:
            recent = float(np.mean(losses[-50:]))
            print(f"[{tag}] step {step:5d}/{STEPS}  loss {loss.item():.4f}  "
                  f"(avg50 {recent:.4f})  {time.time()-t_start:6.1f}s")
    return losses


# ---------------------------------------------------------------------------
# 5. Analysis: rate maps, spatial autocorrelogram, grid score
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_rate_maps(model, gen, n_units):
    """Accumulate mean hidden activation per spatial bin -> (n_units, N_BINS, N_BINS)."""
    model.eval()
    a = gen.arena
    acc = np.zeros((n_units, N_BINS, N_BINS), dtype=np.float64)
    cnt = np.zeros((N_BINS, N_BINS), dtype=np.float64)
    done = 0
    while done < EVAL_TRAJ:
        b = min(BATCH, EVAL_TRAJ - done)
        vel_in, pos, _place, init_place = gen.generate(b, SEQ_LEN)
        _logits, hidden = model(vel_in, init_place)
        h = hidden.reshape(-1, hidden.shape[-1]).cpu().numpy()       # (b*T, U)
        p = pos.reshape(-1, 2).cpu().numpy()                          # (b*T, 2)
        bx = np.clip((p[:, 0] / a * N_BINS).astype(int), 0, N_BINS - 1)
        by = np.clip((p[:, 1] / a * N_BINS).astype(int), 0, N_BINS - 1)
        for u in range(n_units):
            np.add.at(acc[u], (by, bx), h[:, u])
        np.add.at(cnt, (by, bx), 1.0)
        done += b
    cnt_safe = np.where(cnt > 0, cnt, 1.0)
    rate = acc / cnt_safe[None]                                       # mean activity
    return rate, cnt


def autocorrelogram(rmap):
    """2D spatial autocorrelation of a rate map (mean-subtracted, normalised)."""
    r = rmap - np.nanmean(rmap)
    r = np.nan_to_num(r)
    F = np.fft.fft2(r, s=(2 * r.shape[0], 2 * r.shape[1]))
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    if ac.max() > 0:
        ac = ac / ac.max()
    return ac


def grid_score(ac):
    """Standard grid score: 6-fold vs 4-fold rotational symmetry of the
    autocorrelogram ring. score = min(corr@60,120) - max(corr@30,90,150)."""
    n = ac.shape[0]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    r_outer = 0.5 * c
    r_inner = 0.10 * c
    ring = (rr >= r_inner) & (rr <= r_outer)
    base = ac[ring]
    if base.std() < 1e-8:
        return -1.0
    from scipy.ndimage import rotate
    corrs = {}
    for ang in (30, 60, 90, 120, 150):
        rot = rotate(ac, ang, reshape=False, order=1)
        v = rot[ring]
        corrs[ang] = float(np.corrcoef(base, v)[0, 1])
    return min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])


def analyse(model, gen, tag, n_units):
    print(f"[{tag}] collecting rate maps over {EVAL_TRAJ} trajectories ...")
    rate, _cnt = collect_rate_maps(model, gen, n_units)
    scores = np.array([grid_score(autocorrelogram(rate[u])) for u in range(n_units)])
    order = np.argsort(-scores)
    summary = {
        "tag": tag,
        "n_units": int(n_units),
        "grid_score_max": float(np.nanmax(scores)),
        "grid_score_mean": float(np.nanmean(scores)),
        "grid_score_p90": float(np.nanpercentile(scores, 90)),
        "n_units_score_gt_0.3": int((scores > 0.3).sum()),
        "n_units_score_gt_0.0": int((scores > 0.0).sum()),
        "top_unit_scores": [float(scores[i]) for i in order[:TOPK_FIG]],
    }
    print(f"[{tag}] grid score: max={summary['grid_score_max']:.3f} "
          f"mean={summary['grid_score_mean']:.3f} "
          f"#(>0.3)={summary['n_units_score_gt_0.3']}/{n_units}")

    # Figure: top-K rate maps + autocorrelograms.
    k = min(TOPK_FIG, n_units)
    fig, axes = plt.subplots(2, k, figsize=(1.6 * k, 3.4))
    if k == 1:
        axes = axes.reshape(2, 1)
    for j in range(k):
        u = order[j]
        axes[0, j].imshow(rate[u], cmap="jet", origin="lower")
        axes[0, j].set_title(f"u{u}\n{scores[u]:.2f}", fontsize=7)
        axes[0, j].axis("off")
        axes[1, j].imshow(autocorrelogram(rate[u]), cmap="jet", origin="lower")
        axes[1, j].axis("off")
    fig.suptitle(f"{tag}: top {k} cells by grid score (row1 rate map, row2 autocorr)")
    fig.tight_layout()
    fig.savefig(os.path.join(WORK, f"gridcells_{tag}.png"), dpi=110)
    plt.close(fig)

    # Grid-score histogram.
    fig2, ax = plt.subplots(figsize=(5, 3))
    ax.hist(scores[np.isfinite(scores)], bins=30, color="#3b6ea5")
    ax.axvline(0.3, color="r", ls="--", lw=1, label="grid threshold 0.3")
    ax.set_xlabel("grid score"); ax.set_ylabel("# units"); ax.set_title(f"{tag}")
    ax.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(os.path.join(WORK, f"gridscore_hist_{tag}.png"), dpi=110)
    plt.close(fig2)
    return summary


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------
def main():
    gen = TrajectoryGenerator(ARENA, N_PLACE, PLACE_SIGMA, DT, DEVICE)
    results = {"config": {
        "arena": ARENA, "n_place": N_PLACE, "place_sigma": PLACE_SIGMA,
        "seq_len": SEQ_LEN, "hidden": HIDDEN, "batch": BATCH, "steps": STEPS,
        "lr": LR, "wd": WD, "n_bins": N_BINS, "eval_traj": EVAL_TRAJ,
        "device": _dev, "torch": torch.__version__,
    }, "models": {}}

    # ---- Model A: GRU baseline (always) ----
    print("\n" + "=" * 70 + "\n[A] GRU baseline\n" + "=" * 70)
    gru = GRUIntegrator(N_PLACE, HIDDEN)
    a_loss = train(gru, gen, "GRU")
    results["models"]["GRU"] = analyse(gru, gen, "GRU", HIDDEN)
    results["models"]["GRU"]["final_loss"] = float(np.mean(a_loss[-50:]))

    # ---- Model B: MT-LNN core (best-effort) ----
    if HAVE_REPO:
        print("\n" + "=" * 70 + "\n[B] MT-LNN recurrent core\n" + "=" * 70)
        try:
            core, d_model = build_mtlnn()
            mt = MTLNNIntegrator(core, d_model, N_PLACE, HIDDEN)
            b_loss = train(mt, gen, "MTLNN")
            results["models"]["MTLNN"] = analyse(mt, gen, "MTLNN", d_model)
            results["models"]["MTLNN"]["final_loss"] = float(np.mean(b_loss[-50:]))
            results["models"]["MTLNN"]["d_model"] = int(d_model)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[B] MT-LNN core failed: {e}")
            traceback.print_exc()
            results["models"]["MTLNN"] = {"error": repr(e)}
    else:
        results["models"]["MTLNN"] = {"error": "repo unavailable"}

    results["elapsed_s"] = round(time.time() - t_start, 1)
    out = os.path.join(WORK, "grid_cell_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 70)
    print(f"DONE in {results['elapsed_s']}s -> {out}")
    print(json.dumps(results["models"], indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
