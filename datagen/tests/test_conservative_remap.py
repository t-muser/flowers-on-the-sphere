"""Unit tests for the conservative spherical-polygon area-overlap remap."""

import numpy as np
import pytest

from datagen.shock_caps.geometry import (
    COMPUTATIONAL_LOWER,
    COMPUTATIONAL_UPPER,
    cell_centers_latlon,
    compute_aux,
)
from datagen.shock_caps.remap import (
    apply_remap,
    build_remap_matrix,
    target_cell_areas,
)


# Test grid: small enough for fast precompute (~5s), large enough that
# the O(grid_size³) "GC-chord vs parallel" approximation error stays under
# the loosest test tolerance below (~1%). Production grids (Nx=512, Ny=256,
# Nlat=256, Nlon=512) have this same error at ~5e-7.
NX, NY = 40, 20
NLAT, NLON = 32, 64

# Third-order error scaling — see the docstring of _target_cell_areas.
# Rough magnitude: max(dlat, dlon)² · (1/2). At test grid ~ 5e-3.
GC_CHORD_TOL = 5e-3


@pytest.fixture(scope="module")
def grid_params():
    xlower, ylower = COMPUTATIONAL_LOWER
    xupper, yupper = COMPUTATIONAL_UPPER
    dx = (xupper - xlower) / NX
    dy = (yupper - ylower) / NY
    return xlower, ylower, dx, dy


@pytest.fixture(scope="module")
def W():
    return build_remap_matrix(NX, NY, NLAT, NLON)


# ---------------------------------------------------------------------------
# Sanity: shape, sphere area
# ---------------------------------------------------------------------------

def test_W_shape(W):
    assert W.shape == (NLAT * NLON, NX * NY)


def test_target_areas_total_4pi():
    A_t = target_cell_areas(NLAT, NLON)
    np.testing.assert_allclose(A_t.sum(), 4.0 * np.pi, rtol=1e-12)


def test_source_areas_total_4pi(grid_params):
    xlower, ylower, dx, dy = grid_params
    aux = compute_aux(NX, NY, xlower, ylower, dx, dy)
    kappa = aux[0]
    total = float(np.sum(kappa) * dx * dy)
    np.testing.assert_allclose(total, 4.0 * np.pi, rtol=1e-3)


# ---------------------------------------------------------------------------
# Row sums == 1: every target cell fully tiled by source overlaps
# ---------------------------------------------------------------------------

def test_row_sums_one(W):
    """Self-consistent target normalization → every row sums to 1.0 exactly."""
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Constant field round-trip
# ---------------------------------------------------------------------------

def test_constant_field_preserved(W):
    q_src = np.ones((NX, NY), dtype=float)
    q_tgt = apply_remap(W, q_src, NLAT, NLON)
    np.testing.assert_allclose(q_tgt, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Mass conservation: Σ q_tgt · A(T) ≈ Σ q_src · A(S)
# ---------------------------------------------------------------------------

def test_mass_conservation(W, grid_params):
    """Conservative remap: total mass preserved up to GC-chord truncation.

    The remap is exactly conservative against its internal source/target
    partition (both represented as GC-quadrilaterals). When measured against
    the *true* source area kappa·dx·dy and parallel-bounded target area,
    the mismatch scales as O(grid_size³). At the production grid this is
    ~5e-7; at the test grid (cells ~10x coarser) it sits ~5e-3.
    """
    xlower, ylower, dx, dy = grid_params
    aux = compute_aux(NX, NY, xlower, ylower, dx, dy)
    A_src = aux[0] * dx * dy
    A_tgt = target_cell_areas(NLAT, NLON)

    rng = np.random.default_rng(7)
    for _ in range(3):
        q_src = rng.standard_normal((NX, NY))
        q_tgt = apply_remap(W, q_src, NLAT, NLON)
        m_src = float(np.sum(q_src * A_src))
        m_tgt = float(np.sum(q_tgt * A_tgt))
        # rtol caps relative error when m_src is well away from zero;
        # atol guards against random fields where m_src happens to be
        # small (relative error blows up but absolute error stays bounded).
        np.testing.assert_allclose(
            m_tgt, m_src, rtol=GC_CHORD_TOL, atol=GC_CHORD_TOL * 4 * np.pi,
        )


# ---------------------------------------------------------------------------
# Pole smoothness: a zonally-symmetric source produces a zonally-symmetric
# target (no Voronoi pizza slices). f(lat) only at source -> output should be
# ~constant within each lat band of the target grid.
# ---------------------------------------------------------------------------

def test_pole_zonal_symmetry(W, grid_params):
    xlower, ylower, dx, dy = grid_params
    lat_cell, _ = cell_centers_latlon(NX, NY, xlower, ylower, dx, dy)
    q_src = np.sin(lat_cell)        # zonally constant
    q_tgt = apply_remap(W, q_src, NLAT, NLON)
    lat_band_std = q_tgt.std(axis=1)
    # Pizza-slice artefact would push stdev to O(0.1+); a clean conservative
    # remap leaves third-order residue at the test grid.
    assert lat_band_std.max() < 0.02, (
        f"target field should be zonally constant; max stdev={lat_band_std.max():.3e}"
    )


# ---------------------------------------------------------------------------
# Monotonicity: target values bounded by source min/max in their support.
# Globally: min(q_tgt) >= min(q_src), max(q_tgt) <= max(q_src).
# ---------------------------------------------------------------------------

def test_monotone_global_bounds(W):
    rng = np.random.default_rng(13)
    q_src = rng.uniform(-1.0, 1.0, size=(NX, NY))
    q_tgt = apply_remap(W, q_src, NLAT, NLON)
    assert q_tgt.min() >= q_src.min() - 1e-9
    assert q_tgt.max() <= q_src.max() + 1e-9
