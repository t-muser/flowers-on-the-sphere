"""Mapped-sphere geometry for the shallow-water shock-quadrants solver.

Thin wrapper around the Fortran ``sw_sphere_problem.setaux`` routine from
clawpack's ``shallow_sphere`` example. The Calhoun–Helzel single-patch mapped
sphere parameterises the unit sphere over the computational domain
``[-3, 1] × [-1, 1]``, using the same ``mapc2p`` formula as the upstream
Rossby-wave example.

Coordinate conventions
----------------------
- ``lat``:  geodetic latitude in radians, ``[-π/2, π/2]`` (positive north).
- ``lon``:  longitude in radians, ``[0, 2π)`` (positive east).
- 3-D momentum carried by the solver is in ambient Cartesian coordinates
  ``(x, y, z)`` with x-axis through (lat=0, lon=0), z-axis through north pole.
"""

from __future__ import annotations

import numpy as np
from clawpack.pyclaw.examples.shallow_sphere import sw_sphere_problem


RSPHERE: float = 1.0
COMPUTATIONAL_LOWER = (-3.0, -1.0)
COMPUTATIONAL_UPPER = (1.0, 1.0)
NUM_AUX = 16

# sw_sphere_problem.setaux has a hard-coded internal buffer and crashes when called
# with varying (mx, my) in the same process.  Fix: always call with the same fixed
# (mx, my) = (_SETAUX_CHUNK_X, _SETAUX_CHUNK_Y).  Grids that are not multiples are
# padded out; the padding cells are computed but discarded.
# (20, 10) is the size verified to work: sub-grids of the 40×20 test grid passed
# with max error 1.28e-7.  For production (512, 256): 26×26 = 676 identical calls.
_SETAUX_CHUNK_X = 20
_SETAUX_CHUNK_Y = 10


# ---------------------------------------------------------------------------
# Calhoun–Helzel single-patch mapping  (vectorised, copied from Rossby_wave.py)
# ---------------------------------------------------------------------------

def mapc2p_sphere(X: np.ndarray, Y: np.ndarray):
    """Map computational ``(X, Y)`` to physical ``[xp, yp, zp]`` on the sphere."""
    mx, my = X.shape
    sgnz = np.ones((mx, my))
    xc = X[:][:]
    yc = Y[:][:]

    ij2 = np.where(yc < -1.0)
    xc[ij2] = -xc[ij2] - 2.0
    yc[ij2] = -yc[ij2] - 2.0

    ij = np.where(xc < -1.0)
    xc[ij] = -2.0 - xc[ij]
    sgnz[ij] = -1.0

    xc1 = np.abs(xc)
    yc1 = np.abs(yc)
    d = np.maximum(xc1, yc1)
    d = np.maximum(d, 1e-10)
    D = RSPHERE * d * (2 - d) / np.sqrt(2)
    R = RSPHERE * np.ones_like(d)

    centers = D - np.sqrt(R ** 2 - D ** 2)
    xp = D / d * xc1
    yp = D / d * yc1

    ij = np.where(yc1 == d)
    yp[ij] = centers[ij] + np.sqrt(R[ij] ** 2 - xp[ij] ** 2)
    ij = np.where(xc1 == d)
    xp[ij] = centers[ij] + np.sqrt(R[ij] ** 2 - yp[ij] ** 2)

    xp = np.sign(xc) * xp
    yp = np.sign(yc) * yp
    zp = sgnz * np.sqrt(np.maximum(RSPHERE ** 2 - (xp ** 2 + yp ** 2), 0.0))
    return [xp, yp, zp]


# ---------------------------------------------------------------------------
# Cell-centre lat/lon from the mapped grid
# ---------------------------------------------------------------------------

def cell_centers_latlon(
    Nx: int, Ny: int, xlower: float, ylower: float, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lat, lon)`` of every cell centre in radians.

    Both output arrays have shape ``(Nx, Ny)``.
    """
    xc = xlower + (np.arange(Nx) + 0.5) * dx
    yc = ylower + (np.arange(Ny) + 0.5) * dy
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    xp, yp, zp = mapc2p_sphere(XC.copy(), YC.copy())
    lat = np.arcsin(np.clip(zp, -1.0, 1.0))
    lon = np.mod(np.arctan2(yp, xp), 2.0 * np.pi)
    return lat, lon


def subcell_centers_latlon(
    Nx: int, Ny: int, xlower: float, ylower: float, dx: float, dy: float,
    sub_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell sub-sample ``(lat, lon)`` of shape ``(Nx, Ny, S, S)``.

    Each cell is split into ``S × S`` uniformly-placed sub-points in
    computational ``(X, Y)`` (offsets ``(a + 0.5) / S`` for ``a = 0..S-1``).
    Sub-points are mapped through ``mapc2p_sphere`` and converted to
    ``(lat, lon)`` in radians. Used by IC fillers to anti-alias hard-
    bordered piecewise-constant ICs: every cell is classified at all S²
    sub-points and the conserved-variable cell average replaces the
    cell-centre staircase.
    """
    S = int(sub_samples)
    sub = (np.arange(S) + 0.5) / S
    xc_1d = xlower + (np.add.outer(np.arange(Nx), sub).ravel()) * dx
    yc_1d = ylower + (np.add.outer(np.arange(Ny), sub).ravel()) * dy
    XC, YC = np.meshgrid(xc_1d, yc_1d, indexing="ij")
    xp, yp, zp = mapc2p_sphere(XC.copy(), YC.copy())
    lat = np.arcsin(np.clip(zp, -1.0, 1.0)).reshape(Nx, S, Ny, S).transpose(0, 2, 1, 3)
    lon = np.mod(np.arctan2(yp, xp), 2.0 * np.pi).reshape(Nx, S, Ny, S).transpose(0, 2, 1, 3)
    return lat, lon


# ---------------------------------------------------------------------------
# Aux-array setup (wraps Fortran setaux)
# ---------------------------------------------------------------------------

def compute_full_aux(
    Nx: int, Ny: int, xlower: float, ylower: float, dx: float, dy: float,
    num_ghost: int = 2,
) -> np.ndarray:
    """Build the FULL extended aux array — interior cells + all ghost cells.

    Returns shape ``(NUM_AUX, Nx + 2*num_ghost, Ny + 2*num_ghost)``,
    Fortran order. The metric is static (depends only on the mapped sphere
    geometry), so we compute the entire ghost-extended array once at setup
    and reuse it for both ``state.aux`` (interior) and the
    ``user_aux_bc_*`` y-ghost fills.

    Why chunk: ``sw_sphere_problem.setaux`` has a compiled-in workspace,
    and calling it with *varying* ``(mx, my)`` in one process triggers a
    glibc double-free. We therefore tile the full extended grid with
    sub-grids of one fixed size ``(_SETAUX_CHUNK_X, _SETAUX_CHUNK_Y)``
    and discard padding cells in the last row/column of chunks.
    """
    NX_EXT = Nx + 2 * num_ghost
    NY_EXT = Ny + 2 * num_ghost
    cx, cy = _SETAUX_CHUNK_X, _SETAUX_CHUNK_Y
    nx_chunks = (NX_EXT + cx - 1) // cx
    ny_chunks = (NY_EXT + cy - 1) // cy

    # Sub-grid origin offset by num_ghost cells: the sub-grid INTERIOR span
    # covers the entire extended (interior + ghost) target range.
    x_ext_lo = xlower - num_ghost * dx
    y_ext_lo = ylower - num_ghost * dy

    atmp = np.ndarray(
        (NUM_AUX, cx + 2 * num_ghost, cy + 2 * num_ghost),
        dtype=float, order="F",
    )
    result = np.empty((NUM_AUX, NX_EXT, NY_EXT), dtype=float, order="F")
    for bx in range(nx_chunks):
        x0 = bx * cx
        x1 = min(x0 + cx, NX_EXT)
        xl = x_ext_lo + x0 * dx
        for by in range(ny_chunks):
            y0 = by * cy
            y1 = min(y0 + cy, NY_EXT)
            yl = y_ext_lo + y0 * dy
            atmp = sw_sphere_problem.setaux(
                cx, cy, num_ghost, cx, cy, xl, yl, dx, dy, atmp, RSPHERE,
            )
            result[:, x0:x1, y0:y1] = atmp[
                :, num_ghost:num_ghost + (x1 - x0),
                   num_ghost:num_ghost + (y1 - y0),
            ]
    return result


def compute_aux(
    Nx: int, Ny: int, xlower: float, ylower: float, dx: float, dy: float,
    num_ghost: int = 2,
) -> np.ndarray:
    """Interior-only aux array, shape ``(NUM_AUX, Nx, Ny)``."""
    full = compute_full_aux(Nx, Ny, xlower, ylower, dx, dy, num_ghost=num_ghost)
    return full[:, num_ghost:-num_ghost, num_ghost:-num_ghost]


def make_aux_bc(side: str, full_aux: np.ndarray):
    """Return an aux-BC callable that copies y-ghost rows from a cache.

    ``full_aux`` is the precomputed extended aux array of shape
    ``(NUM_AUX, mx + 2*num_ghost, my + 2*num_ghost)`` — same shape as
    ``auxbc`` — so the BC fill is just a slab copy.
    """
    if side not in ("lower", "upper"):
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

    def _fill(state, dim, t, qbc, auxbc, num_ghost):
        if side == "lower":
            auxbc[:, :, :num_ghost] = full_aux[:, :, :num_ghost]
        else:
            auxbc[:, :, -num_ghost:] = full_aux[:, :, -num_ghost:]

    return _fill


# ---------------------------------------------------------------------------
# East/north ↔ 3-D Cartesian momentum helpers
# ---------------------------------------------------------------------------

def en_to_xyz(
    lat: np.ndarray, lon: np.ndarray,
    u_east: np.ndarray, u_north: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project local east/north velocity into 3-D Cartesian components.

    At point ``(lat, lon)`` on the unit sphere:
      ``ê_lon = (−sin λ, cos λ, 0)``
      ``ê_lat = (−sin φ cos λ, −sin φ sin λ, cos φ)``
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    ux = -sin_lon * u_east - sin_lat * cos_lon * u_north
    uy = cos_lon * u_east - sin_lat * sin_lon * u_north
    uz = cos_lat * u_north
    return ux, uy, uz


def xyz_to_en(
    lat: np.ndarray, lon: np.ndarray,
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3-D Cartesian velocity back to local east/north components."""
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    u_east = -sin_lon * ux + cos_lon * uy
    u_north = -sin_lat * cos_lon * ux - sin_lat * sin_lon * uy + cos_lat * uz
    return u_east, u_north
