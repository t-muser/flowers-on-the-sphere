"""Random spherical-cap Riemann initial conditions for shallow water.

``K`` random caps are placed on the unit sphere — each cap is a geodesic
disk with a random centre, a random angular radius, and its own uniform
piecewise-constant ``{h, u, v}`` state. Outside every cap, a separate
``background`` state takes over. Where caps overlap, the higher-index
cap wins (painter's algorithm: caps painted in order ``0 → K-1``).

The result is a varied non-grid-aligned hard-bordered IC across the
sphere — expanding bores, multi-shock collisions, and antipodal focus —
without a separate ``SO(3)`` tilt step, because cap centres are already
uniform on ``S²``.

The flow-strength axis ``delta`` ∈ ``[0, 1]`` scales the velocity
envelope only: ``u, v ∈ [-0.5·delta, 0.5·delta]``. Depth jumps and cap
geometry are independent of ``delta``, so even ``delta = 0`` (rest-state
caps) produces full-amplitude pressure-driven Riemann fans.

Sub-cell antialiased classification: at IC time every solver cell is
sub-sampled ``S × S`` times in computational coords; the cell IC is the
mean of primitives ``(h, u_east, u_north)`` across sub-points, with
momentum projected to 3-D Cartesian at the cell centre. Boundary cells
get a clean one-cell soft transition.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from datagen.shock_quadrants.geometry import (
    en_to_xyz,
    subcell_centers_latlon,
)


# h strictly positive (keeps h far from 0 to prevent blow-up at shock fronts).
# u, v ranges are the delta=1 envelope; the delta arg multiplies them at
# draw time. Max Fr = 0.5·delta / sqrt(9.806 * 0.5) ≈ 0.23·delta.
H_RANGE: Tuple[float, float] = (0.5, 2.0)
VELOCITY_HALFWIDTH: float = 0.5

# Angular cap radius range in radians (~17° to ~57°). Mean ~37° ⇒ each cap
# covers ~17% of the sphere; the expected total cap coverage ~50–70% (at
# K=4 painted in order) leaves background pockets that interact with
# expanding/collapsing cap fronts.
CAP_RADIUS_RANGE: Tuple[float, float] = (0.3, 1.0)


def _sample_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Uniform on ``S²`` via the ``(z, φ)`` parameterisation."""
    z = 2.0 * float(rng.uniform()) - 1.0
    phi = 2.0 * math.pi * float(rng.uniform())
    s = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([s * math.cos(phi), s * math.sin(phi), z], dtype=np.float64)


def _draw_state(rng: np.random.Generator, delta: float) -> Dict[str, float]:
    """One ``(h, u, v)`` triple. ``u, v`` are scaled by ``delta``; at
    ``delta = 0`` they are exactly zero (the multiplication is the only
    use of the random draws, so this is a hard zero, not float slack).
    """
    h_lo, h_hi = H_RANGE
    h = float(rng.uniform(h_lo, h_hi))
    u = delta * float(rng.uniform(-VELOCITY_HALFWIDTH, VELOCITY_HALFWIDTH))
    v = delta * float(rng.uniform(-VELOCITY_HALFWIDTH, VELOCITY_HALFWIDTH))
    return {"h": h, "u": u, "v": v}


def sample_caps(seed: int, K: int, delta: float) -> Tuple[
    List[Dict[str, float]],   # per-cap (h, u, v) states, length K
    List[np.ndarray],          # cap centres (unit vectors), length K
    List[float],               # cap radii (rad), length K
    Dict[str, float],          # background (h, u, v)
]:
    """Draw cap centres, radii, per-cap states, and background state.

    Draw order (deterministic from ``seed``):
        background (h, u, v)
        cap_0..cap_{K-1} centres (z, φ each)
        cap_0..cap_{K-1} radii
        cap_0..cap_{K-1} (h, u, v) states

    ``delta`` scales velocity amplitudes only (depth ranges are fixed).
    ``K`` controls the number of caps; different ``(seed, K)`` pairs
    draw independently.
    """
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    background = _draw_state(rng, delta)
    centers = [_sample_unit_vector(rng) for _ in range(K)]
    r_lo, r_hi = CAP_RADIUS_RANGE
    radii = [float(rng.uniform(r_lo, r_hi)) for _ in range(K)]
    states = [_draw_state(rng, delta) for _ in range(K)]
    return states, centers, radii, background


def cap_label(
    lat: np.ndarray, lon: np.ndarray,
    centers: List[np.ndarray], radii: List[float],
) -> np.ndarray:
    """Painter's-algorithm label per point.

    Returns an integer array with the same shape as ``lat`` / ``lon``:
    ``0 ≤ k ≤ K-1`` for cap-``k``, ``K`` for background. Caps are painted
    in order ``0 → K-1`` so a higher-index cap overwrites lower-index ones
    in overlap regions.
    """
    cos_lat = np.cos(lat)
    px = cos_lat * np.cos(lon)
    py = cos_lat * np.sin(lon)
    pz = np.sin(lat)
    K = len(centers)
    label = np.full(lat.shape, K, dtype=np.int8)
    for k in range(K):
        c = centers[k]
        cos_r = math.cos(radii[k])
        dot = px * c[0] + py * c[1] + pz * c[2]
        label[dot >= cos_r] = k
    return label


def fill_ic(
    h_out: np.ndarray,
    momx_out: np.ndarray,
    momy_out: np.ndarray,
    momz_out: np.ndarray,
    *,
    seed: int,
    K: int,
    delta: float,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    xlower: float,
    ylower: float,
    dx: float,
    dy: float,
    sub_samples: int = 4,
) -> None:
    """Fill the four conserved-state arrays in place at every cell.

    Sub-cell antialiased: every cell is sub-sampled ``S²`` times in
    computational ``(X, Y)``; each sub-point is classified into a cap
    (or background) by ``cap_label``; the cell IC is the mean of the
    primitives ``(h, u_east, u_north)`` across sub-points, with momentum
    projected to 3-D Cartesian at the cell centre.
    """
    states, centers, radii, background = sample_caps(seed, K, delta)
    # state_table[K] is the background; state_table[k<K] is cap k.
    state_table = states + [background]
    Nx, Ny = lat_centers.shape

    lat_s, lon_s = subcell_centers_latlon(
        Nx, Ny, xlower, ylower, dx, dy, sub_samples,
    )
    label = cap_label(lat_s, lon_s, centers, radii)  # (Nx, Ny, S, S)

    h_sub = np.empty(label.shape, dtype=np.float64)
    u_sub = np.empty(label.shape, dtype=np.float64)
    v_sub = np.empty(label.shape, dtype=np.float64)
    for k, s in enumerate(state_table):
        mask = label == k
        h_sub[mask] = s["h"]
        u_sub[mask] = s["u"]
        v_sub[mask] = s["v"]

    h_cell = h_sub.mean(axis=(-2, -1))
    u_cell = u_sub.mean(axis=(-2, -1))
    v_cell = v_sub.mean(axis=(-2, -1))

    ux, uy, uz = en_to_xyz(lat_centers, lon_centers, u_cell, v_cell)

    h_out[...] = h_cell
    momx_out[...] = h_cell * ux
    momy_out[...] = h_cell * uy
    momz_out[...] = h_cell * uz
