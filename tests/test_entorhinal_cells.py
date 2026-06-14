"""
tests/test_entorhinal_cells.py — behavioural contract for the multi-module
entorhinal spatial code added to mt_lnn/spatial.py (2026-06-15).

The single-scale GridCellEncoding flattens all scales into one map with a shared
orientation and zero phase. Real entorhinal cortex is 4-5 discrete grid MODULES
(Stensola et al. 2012), each with its own scale (ratio ~1.4), orientation, and
phase, plus head-direction cells (heading) and border cells (distance to walls).
We pin the contracts of the new, strictly-additive cell types:

  • MultiScaleGridCellModules: geometric scale progression; per-module
    orientation/phase; a module's grid component is periodic at its wavelength;
    finer modules oscillate faster; deterministic & zero trainable params;
  • HeadDirectionCells: von Mises tuning peaks at the heading; rotating the
    heading rotates the active cell; accepts both vector and angle inputs;
  • BoundaryDistanceCells: a cell fires when its wall is at its preferred
    distance; near a wall the distance-0 cell of that wall lights up; bounded;
  • the existing GridCellEncoding / SpatialCoordEncoder are untouched (covered
    by their own tests) — these classes are purely additive (zero regression).

Run:  python -m pytest tests/test_entorhinal_cells.py -v
"""
import sys
import math

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import (                                                  # noqa: E402
    MultiScaleGridCellModules,
    HeadDirectionCells,
    BoundaryDistanceCells,
)


# --------------------------------------------------------------------------- #
# MultiScaleGridCellModules                                                    #
# --------------------------------------------------------------------------- #
def test_grid_modules_shape_and_zero_params():
    g = MultiScaleGridCellModules(n_modules=4)
    assert g.out_dim == 4 * 6
    assert sum(p.numel() for p in g.parameters()) == 0      # fixed feature map
    x = torch.rand(2, 7, 2)
    out = g(x)
    assert out.shape == (2, 7, 24)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-6                    # cos/sin bounded


def test_grid_modules_geometric_scale_progression():
    ratio = 1.4
    g = MultiScaleGridCellModules(n_modules=5, base_wavelength=1.0, scale_ratio=ratio)
    wl = g.module_wavelengths
    # Each successive module is ratio× finer.
    for i in range(len(wl) - 1):
        assert (wl[i] / wl[i + 1]).item() == pytest.approx(ratio, rel=1e-5)


def test_grid_module_component_is_periodic_at_its_wavelength():
    # Translating along a module's first lattice direction by exactly one
    # wavelength reproduces that direction's (cos, sin) component (phase += 2π).
    g = MultiScaleGridCellModules(n_modules=3, base_wavelength=1.0,
                                  scale_ratio=1.4, phase_jitter=True, seed=1)
    m = 1
    k0 = g.wave_vectors[m, 0]                                 # first direction's k
    lam = g.module_wavelengths[m].item()                     # = 2π / |k0|
    u = (k0 / k0.norm())                                      # unit dir
    x = torch.rand(2, 5, 2)
    x_shift = x + lam * u
    code = g.module_code(x, m)
    code_shift = g.module_code(x_shift, m)
    # Channel 0 = cos(dir0), channel 3 = sin(dir0) → invariant under +1 wavelength.
    assert torch.allclose(code[..., 0], code_shift[..., 0], atol=1e-4)
    assert torch.allclose(code[..., 3], code_shift[..., 3], atol=1e-4)


def test_finer_module_oscillates_faster():
    # Sweep a straight path; the finest module's cos component crosses zero more
    # often than the coarsest (smaller wavelength → higher spatial frequency).
    g = MultiScaleGridCellModules(n_modules=4, base_wavelength=1.0, scale_ratio=1.6,
                                  orientation_jitter=False, phase_jitter=False)
    t = torch.linspace(0, 1, 200)
    path = torch.stack([t, torch.zeros_like(t)], dim=-1)[None]    # (1, 200, 2)

    def zero_crossings(sig):
        s = torch.sign(sig)
        return int((s[1:] * s[:-1] < 0).sum())

    coarse = g.module_code(path, 0)[0, :, 0]                       # module 0 cos d0
    fine = g.module_code(path, g.n_modules - 1)[0, :, 0]           # finest cos d0
    assert zero_crossings(fine) > zero_crossings(coarse)


def test_grid_modules_orientation_jitter_distinct():
    # With orientation jitter the modules' lattice directions are not identical
    # (beyond a pure scale factor); without jitter they share orientation 0.
    g_jit = MultiScaleGridCellModules(n_modules=4, orientation_jitter=True, seed=3)
    # Normalised first-direction unit vectors per module.
    units = g_jit.wave_vectors[:, 0] / g_jit.wave_vectors[:, 0].norm(dim=-1, keepdim=True)
    # Not all modules point the same way.
    assert not torch.allclose(units[0], units[1], atol=1e-3)

    g_no = MultiScaleGridCellModules(n_modules=4, orientation_jitter=False)
    units0 = g_no.wave_vectors[:, 0] / g_no.wave_vectors[:, 0].norm(dim=-1, keepdim=True)
    assert torch.allclose(units0[0], units0[1], atol=1e-6)         # all aligned


def test_grid_modules_deterministic_by_seed():
    a = MultiScaleGridCellModules(n_modules=4, seed=7)
    b = MultiScaleGridCellModules(n_modules=4, seed=7)
    x = torch.rand(2, 6, 2)
    assert torch.equal(a(x), b(x))


def test_grid_modules_reject_bad_input():
    g = MultiScaleGridCellModules(n_modules=2)
    with pytest.raises(ValueError):
        g(torch.rand(2, 5, 3))            # coord_dim must be 2
    with pytest.raises(IndexError):
        g.module_code(torch.rand(1, 4, 2), 5)


# --------------------------------------------------------------------------- #
# HeadDirectionCells                                                           #
# --------------------------------------------------------------------------- #
def test_head_direction_peaks_at_heading():
    hd = HeadDirectionCells(n_cells=12, concentration=6.0)
    assert hd.out_dim == 12
    assert sum(p.numel() for p in hd.parameters()) == 0
    # Heading pointing exactly along preferred dir of cell 0 (angle 0 → +x).
    heading = torch.tensor([[[1.0, 0.0]]])                 # (1,1,2)
    r = hd(heading)[0, 0]
    assert r.shape == (12,)
    assert torch.argmax(r).item() == 0
    assert r.max().item() == pytest.approx(1.0, abs=1e-5)  # von Mises peak = 1
    assert (r > 0).all() and (r <= 1.0 + 1e-6).all()


def test_head_direction_rotates_with_heading():
    n = 12
    hd = HeadDirectionCells(n_cells=n, concentration=8.0)
    # Heading at the preferred angle of cell 3 → that cell should win.
    ang = 2.0 * math.pi * 3 / n
    heading = torch.tensor([[[math.cos(ang), math.sin(ang)]]])
    assert torch.argmax(hd(heading)[0, 0]).item() == 3


def test_head_direction_accepts_angle_input():
    hd = HeadDirectionCells(n_cells=8, concentration=6.0)
    ang = torch.tensor([[[2.0 * math.pi * 2 / 8]]])         # (1,1,1) angle form
    r = hd(ang)
    assert r.shape == (1, 1, 8)
    assert torch.argmax(r[0, 0]).item() == 2


def test_head_direction_rejects_bad_input():
    hd = HeadDirectionCells(n_cells=8)
    with pytest.raises(ValueError):
        hd(torch.rand(1, 4, 3))           # last dim must be 1 or 2


# --------------------------------------------------------------------------- #
# BoundaryDistanceCells                                                        #
# --------------------------------------------------------------------------- #
def test_boundary_cells_shape_and_zero_params():
    bc = BoundaryDistanceCells(arena=1.0, coord_dim=2, n_dist=4, sigma=0.1)
    assert bc.n_walls == 4
    assert bc.out_dim == 4 * 4
    assert sum(p.numel() for p in bc.parameters()) == 0
    pos = torch.rand(2, 5, 2)
    out = bc(pos)
    assert out.shape == (2, 5, 16)
    assert (out >= 0).all() and (out <= 1.0 + 1e-6).all()


def test_boundary_cell_fires_at_its_wall_distance():
    # A position right on the low-x wall (x=0) maximally drives that wall's
    # distance-0 cell, and weakly drives the far-distance cells.
    bc = BoundaryDistanceCells(arena=1.0, coord_dim=2, n_dist=4, sigma=0.08)
    pos = torch.tensor([[[0.0, 0.5]]])                     # on low-x wall
    out = bc(pos)[0, 0].reshape(bc.n_walls, bc.n_dist)
    # Wall ordering: [low_x, low_y, high_x, high_y]; pref_dist[0] = 0.
    low_x = out[0]
    assert torch.argmax(low_x).item() == 0                 # nearest-distance cell wins
    assert low_x[0].item() == pytest.approx(1.0, abs=1e-5)


def test_boundary_distinguishes_walls():
    # Near the low-x wall vs near the high-y wall light up different wall groups.
    bc = BoundaryDistanceCells(arena=1.0, coord_dim=2, n_dist=3, sigma=0.08)
    near_lowx = bc(torch.tensor([[[0.02, 0.5]]]))[0, 0].reshape(4, 3)
    near_highy = bc(torch.tensor([[[0.5, 0.98]]]))[0, 0].reshape(4, 3)
    # The defining border-cell signal is the nearest-distance cell (column 0):
    # whichever wall the agent is right next to lights up its distance-0 cell.
    # Wall ordering [low_x, low_y, high_x, high_y].
    assert torch.argmax(near_lowx[:, 0]).item() == 0      # hugging low-x wall
    assert torch.argmax(near_highy[:, 0]).item() == 3     # hugging high-y wall


def test_boundary_rejects_bad_input():
    bc = BoundaryDistanceCells(coord_dim=2)
    with pytest.raises(ValueError):
        bc(torch.rand(1, 4, 3))           # coord_dim mismatch
    with pytest.raises(ValueError):
        BoundaryDistanceCells(arena=[1.0, 1.0, 1.0], coord_dim=2)   # wrong arena len


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
